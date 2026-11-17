from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.trainer.states import TrainerFn
from ratiopath.masks.mask_builders import MaskBuilder, MaxAggregator, MeanAggregator

from lymph_nodes.data.datasets.meta_dataset import column_values


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyvips
    from lightning import LightningModule, Trainer
    from rationai.mlkit.lightning.loggers import MLFlowLogger

    from lymph_nodes.data.datasets.embedding import EmbeddingClassificationDataset
    from lymph_nodes.typedefs import Batch, Outputs


# Slide columns (written by preprocessing/tiling.py) describing the tiling level's grid.
_GEOMETRY = (
    "extent_x",
    "extent_y",
    "tile_extent_x",
    "tile_extent_y",
    "stride_x",
    "stride_y",
    "mpp_x",
    "mpp_y",
)
# PredictionMasks' combine: the artifact directory of each kind of mask.
_DIRECTORIES = {
    "centre": "prediction_masks",
    "mean": "prediction_masks_mean",
    "max": "prediction_masks_max",
}


class PredictionMasks(Callback):
    """A prediction mask of every slide of the final evaluations, logged to MLflow.

    During ``validate`` and ``test`` (not the validation inside ``fit``), each evaluated
    slide gets a BigTIFF of the predicted cancer probability scaled to 0-255 (0 also
    where no tile was evaluated), logged as ``prediction_masks/<cohort>/<slide_name>.tiff``
    with a max-pooled pyramid (``_write_mask``: a flag stays visible zoomed out).

    The mask has one pixel per tile: the stride x stride cell at the tile's grid
    position, so its mpp is the tiling level's times the stride. It covers the slide
    and no more: xOpat draws an overlay stretched to the slide's size, not placed by its
    mpp, so the grid's overhang past the slide's edge (the last tiles reach up to a
    tile beyond it) would shrink the drawn mask by as much — shifting it by up to
    ~0.4% of the distance from the slide's corner, a few hundred um across a lymph node
    slide. The mask is cropped to the slide, rounded to whole pixels. Tiles overlap (stride < tile extent), and the cells of the grid cover
    the slide exactly once. A cell starts at its tile's corner rather than being
    centered on the tile, which shifts the mask by (tile extent - stride) / 2 — 51 px of
    the tiling level for the 224/122 lymph node grid, under half a mask pixel. Scaling
    the mask up to shift it exactly would cost a full-resolution image per slide, and a
    MaskBuilder over the full overlapping tiles would have to work at a resolution of
    gcd(tile extent, stride): 2 px for that grid.

    Per-patch outputs (B, g, g), as from SegmentationMetaArch, give a mask with one
    pixel per patch instead (tile extent / g px of the level: 14 for the foundation
    models' patches), assembled by MaskBuilder: from the full tiles without overlap
    (stride = tile extent), and from each tile's centre, (tile extent - stride) / 2
    clipped off each side, when the overlap is a whole number of patches on each side
    (stride 112 of 224 px: the central 8 x 8 of 16 x 16 patches), so that every patch
    comes from the tile in which it has the most context (eval_stride of
    configs/experiment/tiling/mmci_hdab_segmentation.yaml). Other overlaps are averaged.

    ``combine``: which masks to log, by how the overlapping tiles make a pixel —
    ``centre`` (as above, ``prediction_masks/``), and the ``mean`` or ``max`` of the
    full outputs of every tile covering it (``prediction_masks_mean/``,
    ``prediction_masks_max/``; up to four tiles at half a tile's stride): the centre
    takes each point from the tile in which it has the most context, the mean smooths
    over the tiles' disagreement, the max keeps whatever any of them flagged. Mean and
    max for per-patch outputs only, where the overlapping tiles' patches coincide
    (stride x g a multiple of the tile extent): all three on one pixel grid.

    Needs the steps to return their logits and the samples' meta to carry ``slide``
    (position in the dataset's ``slides``), ``tile_x`` and ``tile_y``.

    Each slide's masks are written as soon as its last tile is in (the evaluation reads
    the slides one after another), and logged at the end: only the slides still being
    read are held. Holding every tile's outputs to the end took 2 KB a tile at 32 x 32
    patches (6.7 GB for the 3.3M tiles of a lymph node evaluation, twice that while
    they were joined), and would take 8 KB at 64 x 64: 27 GB, 54 GB joined.
    """

    def __init__(self, combine: Sequence[str] = ("centre",)) -> None:
        unknown = set(combine) - set(_DIRECTORIES)
        if unknown:
            raise ValueError(
                f"combine: unknown {sorted(unknown)}, not of {list(_DIRECTORIES)}"
            )
        self.combine = tuple(combine)
        self._run: _Run | None = None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._collect(trainer.datamodule.val, outputs, batch)  # type: ignore[attr-defined]

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._log_masks(trainer)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect(trainer.datamodule.test, outputs, batch)  # type: ignore[attr-defined]

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_masks(trainer)

    def _collect(
        self, dataset: EmbeddingClassificationDataset, outputs: Outputs, batch: Batch
    ) -> None:
        _, _, meta = batch
        if self._run is None:
            self._run = _Run(dataset)
        run = self._run
        slide = meta["slide"].cpu().numpy()
        tile_x = meta["tile_x"].cpu().numpy()
        tile_y = meta["tile_y"].cpu().numpy()
        probability = (
            torch.sigmoid(outputs.detach().float()).cpu().numpy().astype(np.float16)
        )
        for position in np.unique(slide).tolist():
            rows = slide == position
            run.pending.setdefault(position, []).append(
                (tile_x[rows], tile_y[rows], probability[rows])
            )
            run.received[position] += int(rows.sum())
            if run.received[position] >= run.expected[position]:
                self._write_slide(run, position)

    def _write_slide(self, run: _Run, position: int) -> None:
        """The masks of the slide at ``position`` (dataset.slides), into run's directory."""
        import pyvips

        tile_x, tile_y, probability = (
            np.concatenate(c) for c in zip(*run.pending.pop(position), strict=True)
        )
        cohort = str(run.cohorts[position]) if run.cohorts is not None else "unknown"
        geometry = {column: values[position] for column, values in run.geometry.items()}
        for combine in self.combine:
            built = _prediction_mask(geometry, tile_x, tile_y, probability, combine)
            if built is None:
                continue
            mask, (pixel_y, pixel_x) = built
            directory = Path(run.directory.name, _DIRECTORIES[combine], cohort)
            directory.mkdir(parents=True, exist_ok=True)
            _write_mask(
                pyvips.Image.new_from_array(mask),
                path=directory / f"{run.names[position]}.tiff",
                mpp_x=float(geometry["mpp_x"] * pixel_x),
                mpp_y=float(geometry["mpp_y"] * pixel_y),
            )

    def _log_masks(self, trainer: Trainer) -> None:
        run, self._run = self._run, None
        if run is None:
            return
        try:
            # Slides with fewer tiles evaluated than they have (limit_*_batches).
            for position in sorted(run.pending):
                self._write_slide(run, position)
            for directory in sorted(Path(run.directory.name).iterdir()):
                cast("MLFlowLogger", trainer.logger).log_artifacts(
                    str(directory), directory.name
                )
        finally:
            run.directory.cleanup()


class _Run:
    """PredictionMasks' state through one evaluation of ``dataset``.

    ``expected``: each slide's number of tiles (the dataset's per-slide datasets, in
    the order of its ``slides``); ``received``: how many of them are in so far;
    ``pending``: per slide still being read, its batches' tile_x, tile_y and
    probabilities; ``directory``: the masks written so far.
    """

    def __init__(self, dataset: EmbeddingClassificationDataset) -> None:
        self.expected = np.diff(dataset.cumulative_sizes, prepend=0)
        self.received = np.zeros_like(self.expected)
        self.pending: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self.directory = TemporaryDirectory()
        slides = dataset.slides
        self.names = column_values(slides, "slide_name")
        self.cohorts = (
            column_values(slides, "cohort_id")
            if "cohort_id" in slides.column_names
            else None
        )
        self.geometry = {column: column_values(slides, column) for column in _GEOMETRY}


def _write_mask(image: pyvips.Image, path: Path, mpp_x: float, mpp_y: float) -> None:
    """Ratiopath's write_big_tiff, its pyramid max-pooled rather than averaged.

    An averaged pyramid loses a flag of a cell or two (a pixel or two of the mask)
    within a few levels: averaged with its unflagged neighbours it falls below the
    viewer's threshold, so zoomed out xOpat drew nothing where the model flagged
    (measured on the 3.4 um masks: a flag at 0.9 is under 0.35 from the 14 um level
    on). Max-pooled, every level keeps each region's highest probability, so a flag
    stays drawn at any zoom. The full-resolution level, which the metrics and the
    reports read, is the same either way. The resolution in pixels per cm, as libvips
    wrote it by default before 8.18 (which writes inches) and as the reports read it.
    """
    import pyvips

    image.tiffsave(
        path,
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=512,
        tile_height=512,
        xres=1000 / mpp_x,
        yres=1000 / mpp_y,
        resunit=pyvips.enums.ForeignTiffResunit.CM,
        pyramid=True,
        region_shrink=pyvips.enums.RegionShrink.MAX,
    )


def _prediction_mask(
    slide: dict[str, Any],
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    probability: np.ndarray,
    combine: str = "centre",
) -> tuple[np.ndarray, np.ndarray] | None:
    """The uint8 probability mask of one slide (see PredictionMasks), or None.

    Also returns the size of a mask pixel in pixels of the tiling level, (y, x). None
    for a ``combine`` of ``mean`` or ``max`` the outputs don't allow (PredictionMasks).
    """
    stride = (int(slide["stride_y"]), int(slide["stride_x"]))
    full = (int(slide["tile_extent_y"]), int(slide["tile_extent_x"]))
    output_extent: int | tuple[int, ...]
    clipping: tuple[int, int] = (0, 0)
    if probability.ndim == 1:
        if combine != "centre":
            return None  # full tiles would need a grid of gcd(tile extent, stride)
        # One mask pixel per tile: a "tile" of stride x stride source pixels.
        tile_extent, output_extent, values = stride, 1, probability[:, np.newaxis]
    else:
        # One mask pixel per patch of the full tile; of its centre if tiles overlap.
        tile_extent = full
        output_extent, values = probability.shape[1:], probability[:, np.newaxis]
        if combine == "centre":
            clipping = _centre_clipping(tile_extent, stride, probability.shape[1:])
        elif any(
            (step * n) % extent
            for step, n, extent in zip(stride, probability.shape[1:], full, strict=True)
        ):
            return None  # the overlapping tiles' patches don't coincide

    builder = MaskBuilder(
        source_extents=(int(slide["extent_y"]), int(slide["extent_x"])),
        source_tile_extent=tile_extent,
        output_tile_extent=output_extent,
        stride=stride,
        aggregation=MaxAggregator if combine == "max" else MeanAggregator,
    )
    try:
        builder.update_batch(
            values.astype(np.float32),
            np.stack([tile_y, tile_x], axis=1).astype(np.int64),
            edge_clipping=clipping,
        )
        result = builder.finalize()
        mask = result["mask"] if isinstance(result, dict) else result
        mask = np.round(mask[0] * 255).astype(np.uint8)
        pixel = builder.span / builder.mask_extents
        # The slide's extent in mask pixels (see PredictionMasks: no overhang).
        extent = np.array([int(slide["extent_y"]), int(slide["extent_x"])])
        rows, cols = np.maximum(np.round(extent / pixel), 1).astype(int)
        return mask[:rows, :cols], pixel
    finally:
        builder.cleanup()


def _centre_clipping(
    tile_extent: tuple[int, int], stride: tuple[int, int], patches: tuple[int, ...]
) -> tuple[int, int]:
    """Patches to clip off each side of a tile to keep its centre, (y, x).

    Half the overlap, (tile extent - stride) / 2, in patches; 0 where that is not a
    whole number of patches (MaskBuilder then averages the overlapping tiles).
    """
    clipping = []
    for extent, step, n in zip(tile_extent, stride, patches, strict=True):
        overlap = (extent - step) * n  # in patches, times the tile extent
        whole = overlap > 0 and overlap % (2 * extent) == 0
        clipping.append(overlap // (2 * extent) if whole else 0)
    return clipping[0], clipping[1]
