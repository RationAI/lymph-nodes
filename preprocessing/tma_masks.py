"""TMA cores and their cytokeratin ground truth, per slide (preprocessing/tma.py).

Corrects the cytokeratin masks of MLflow run 8ad173ea (experiment 68; ``reference_dir``):
a per-image automatic DAB threshold whose outlines follow the epithelium tightly, but
which labelled whole CK-negative cores positive (some TNBC slides) and left faintly
stained cells negative (analysis/tma_faint_cells.py, analysis/tma_gt_check.py).

Per slide:

1. its cores, on a white-balanced image at ~``coarse_mpp`` um/px (find_cores);
2. every core read at ~``mpp`` um/px in ``chunk``-pixel pieces (with a margin for the
   cell rule), white-balanced like the slide, with the reference mask's piece (its full
   resolution is the slide's level at ~0.5 um/px), labelled by corrected_mask:
   255 positive, 128 ignore, 0 negative.

Logs:

- ``<artifact_path>/<slide>.tiff``: the ground truth at the level read (uint8 pyramid,
  0 outside the cores); tiled like the old masks (cytokeratin_coverage/_grid with
  min_label 255, so ignore counts as negative until the training masks it out);
- ``<cores_artifact_path>/<slide>.tiff``: the cores at ~``coarse_mpp`` (uint16, a
  core's index, 0 outside cores);
- ``cores.csv``: every core — slide, index (its label in the core mask), row and column
  in the array, its id (``<slide>:r<row>c<col>``, ``-<index>`` added where two share a
  position), centre in level-0 pixels, radius, the core mask's pixel size, tissue area,
  number of pieces, positive and ignore shares, and ``ignored_<name>``: the share each
  of ``ignore_dirs``' masks covers (the pathologists' ignore masks, the custom
  cytokeratin ignore annotations; the tiling drops or filters their tiles).
  preprocessing/tma_split.py splits the cores between folds from it.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
import pyvips
import ray
import tifffile
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from PIL import Image
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.openslide import OpenSlide
from scipy import ndimage

from preprocessing.tma import (
    IGNORE,
    POSITIVE,
    balance,
    core_region,
    corrected_mask,
    find_cores,
    white_point,
)


def read_reference(path: Path, x: int, y: int, w: int, h: int) -> np.ndarray:
    """The reference mask's (x, y, w, h) at full resolution, >= 128, False outside it."""
    image = pyvips.Image.new_from_file(str(path), access="random")
    out = np.zeros((h, w), bool)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(x + w, image.width), min(y + h, image.height)
    if x1 > x0 and y1 > y0:
        crop = image.crop(x0, y0, x1 - x0, y1 - y0).numpy()
        crop = crop[..., 0] if crop.ndim == 3 else crop
        out[y0 - y : y1 - y, x0 - x : x1 - x] = crop >= 128
    return out


def ignore_shares(path: Path, labels: np.ndarray) -> dict[int, float]:
    """Per core (index in ``labels``), the share of it an ignore mask covers (>= 128).

    The mask read at the smallest pyramid level still as fine as ``labels`` (both cover
    the whole slide), averaged onto the core grid.
    """
    with tifffile.TiffFile(path) as tiff:
        levels = tiff.series[0].levels
        fine = [lv for lv in levels if lv.shape[0] >= labels.shape[0]] or levels[:1]
        mask = fine[-1].asarray()
    mask = mask[..., 0] if mask.ndim == 3 else mask
    covered = Image.fromarray(((mask >= 128) * 255).astype(np.uint8))
    grid = np.asarray(covered.resize(labels.shape[::-1], Image.BOX)) / 255
    indices = [i for i in np.unique(labels) if i > 0]
    means = ndimage.mean(grid, labels, indices) if indices else []
    return {int(i): float(m) for i, m in zip(indices, means, strict=True)}


def composite(rgba: np.ndarray) -> np.ndarray:
    """RGB of an RGBA read, transparent (unscanned) areas white, float32."""
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    return rgba[..., :3] * alpha + 255 * (1 - alpha)


def process_slide(
    slide_path: str,
    reference: Path,
    ignores: dict[str, Path],
    mpp: float,
    coarse_mpp: float,
    output_path: Path,
    chunk: int,
    pad: int,
    rule: dict,
    cores_rule: dict,
) -> None:
    name = Path(slide_path).stem
    with OpenSlide(slide_path) as slide:
        level, coarse_level = slide.closest_level(mpp), slide.closest_level(coarse_mpp)
        mpp_x, mpp_y = (float(v) for v in slide.slide_resolution(level))
        cmpp_x, cmpp_y = (float(v) for v in slide.slide_resolution(coarse_level))
        level_mpp = (mpp_x + mpp_y) / 2
        down = round(slide.level_downsamples[level])
        factor = round(slide.level_downsamples[coarse_level]) // down
        width, height = slide.level_dimensions[level]

        coarse = composite(
            np.asarray(
                slide.read_region(
                    (0, 0), coarse_level, slide.level_dimensions[coarse_level]
                )
            )
        )
        white = white_point(coarse)
        labels, cores = find_cores(
            balance(coarse, white), (cmpp_x + cmpp_y) / 2, **cores_rule
        )

        covered = {name: ignore_shares(path, labels) for name, path in ignores.items()}
        mask = pyvips.Image.black(width, height)
        rows = []
        for core in cores:
            box = ndimage.find_objects((labels == core.index).astype(np.int32))[0]
            y0, x0 = box[0].start * factor, box[1].start * factor
            y1 = min(box[0].stop * factor, height)
            x1 = min(box[1].stop * factor, width)
            own = np.repeat(
                np.repeat(core_region(labels[box], core.index), factor, 0), factor, 1
            )
            own = np.pad(own[: y1 - y0, : x1 - x0], pad)
            out = np.zeros((y1 - y0, x1 - x0), np.uint8)
            for cy in range(y0, y1, chunk):
                for cx in range(x0, x1, chunk):
                    h, w = min(chunk, y1 - cy), min(chunk, x1 - cx)
                    rgb = balance(
                        composite(
                            np.asarray(
                                slide.read_region(
                                    ((cx - pad) * down, (cy - pad) * down),
                                    level,
                                    (w + 2 * pad, h + 2 * pad),
                                )
                            )
                        ),
                        white,
                    )
                    oy, ox = cy - y0, cx - x0
                    core_mask = own[oy : oy + h + 2 * pad, ox : ox + w + 2 * pad]
                    od = -np.log((rgb + 1) / 256).sum(-1)
                    core_mask = core_mask & (ndimage.gaussian_filter(od, 2) > 0.1)
                    old = read_reference(
                        reference, cx - pad, cy - pad, w + 2 * pad, h + 2 * pad
                    )
                    labelled = corrected_mask(rgb, level_mpp, core_mask, old, **rule)
                    out[oy : oy + h, ox : ox + w] = labelled[
                        pad : pad + h, pad : pad + w
                    ]
            mask = mask.insert(pyvips.Image.new_from_array(out), x0, y0)
            tissue = max(1, int((own[pad:-pad, pad:-pad] if pad else own).sum()))
            rows.append(
                {
                    "slide": name,
                    "index": core.index,
                    "row": core.row,
                    "col": core.col,
                    "x": (core.x * factor + 0.5) * down,
                    "y": (core.y * factor + 0.5) * down,
                    "radius_um": core.radius * (cmpp_x + cmpp_y) / 2,
                    # the core label mask's pixel, for preprocessing.tiling_blocks.CoreSelection
                    "label_mpp_x": cmpp_x,
                    "label_mpp_y": cmpp_y,
                    "tissue_mm2": core.tissue_mm2,
                    "pieces": core.pieces,
                    "positive_share": float((out == POSITIVE).sum() / tissue),
                    "ignore_share": float((out == IGNORE).sum() / tissue),
                    # covered by the ignore masks (ignore_dirs), e.g. excluded cores
                    **{
                        f"ignored_{name}": shares.get(core.index, 0.0)
                        for name, shares in covered.items()
                    },
                }
            )

    for folder in ("masks", "cores"):
        (output_path / folder).mkdir(parents=True, exist_ok=True)
    write_big_tiff(
        mask, path=output_path / "masks" / f"{name}.tiff", mpp_x=mpp_x, mpp_y=mpp_y
    )
    write_big_tiff(
        pyvips.Image.new_from_array(labels.astype(np.uint16)),
        path=output_path / "cores" / f"{name}.tiff",
        mpp_x=cmpp_x,
        mpp_y=cmpp_y,
    )
    table = pd.DataFrame(rows)
    if not table.empty:
        position = (
            table.slide + ":r" + table.row.astype(str) + "c" + table.col.astype(str)
        )
        shared = position.duplicated(keep=False)
        table["core_id"] = position.where(
            ~shared, position + "-" + table["index"].astype(str)
        )
    table.to_csv(output_path / "cores" / f"{name}.csv", index=False)


@ray.remote(memory=8 * 1024**3)
def process_item(item: tuple[str, Path, dict[str, Path]], **kwargs) -> None:
    process_slide(*item, **kwargs)


@with_cli_args(["+preprocessing=tma_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = list(hydra.utils.instantiate(config.dataset.slides))
    references = {
        path.stem: path
        for path in Path(download_artifacts(config.reference_dir)).rglob("*.tiff")
    }
    missing = [s for s in slides if Path(s).stem not in references]
    if missing:
        raise SystemExit(f"no reference mask for: {', '.join(map(str, missing))}")
    ignore_masks = {
        name: {p.stem: p for p in Path(download_artifacts(uri)).rglob("*.tiff")}
        for name, uri in (config.get("ignore_dirs") or {}).items()
        if uri
    }

    with TemporaryDirectory() as output_dir:
        output = Path(output_dir)
        process_items(
            [
                (
                    str(s),
                    references[Path(s).stem],
                    {
                        name: masks[Path(s).stem]
                        for name, masks in ignore_masks.items()
                        if Path(s).stem in masks
                    },
                )
                for s in slides
            ],
            process_item=process_item,
            fn_kwargs={
                "mpp": config.mpp,
                "coarse_mpp": config.coarse_mpp,
                "output_path": output,
                "chunk": config.chunk,
                "pad": config.pad,
                "rule": dict(config.rule),
                "cores_rule": dict(config.cores_rule),
            },
            max_concurrent=config.max_concurrent,
        )
        tables = [
            pd.read_csv(p)
            for p in sorted((output / "cores").glob("*.csv"))
            if p.stat().st_size > 1
        ]
        cores = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        (output / "summary").mkdir()
        cores.to_csv(output / "summary" / "cores.csv", index=False)
        logger.log_artifacts(
            local_dir=str(output / "masks"), artifact_path=config.artifact_path
        )
        logger.log_artifacts(
            local_dir=str(output / "cores"), artifact_path=config.cores_artifact_path
        )
        logger.log_artifacts(local_dir=str(output / "summary"))
        if not cores.empty:
            logger.log_hyperparams(
                {"cores": len(cores), "slides": cores.slide.nunique()}
            )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
