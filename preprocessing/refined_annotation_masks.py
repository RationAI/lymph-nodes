"""Refined annotation masks: the pathologists' annotations narrowed to stained cells.

The annotations are rough: drawn generously around a focus, they take in the lymphoid
tissue around and between the stained cells, and now and then cut through a stained
region at a tight edge. The refined mask of a slide keeps the stained cells
(preprocessing.staining.stained_cells: cytokeratin-positive by colour deconvolution,
their nuclei filled)
whose connected region overlaps the annotation, whole — beyond the outline too, up to
``margin_um`` from it — and nothing else. Tiled like any other mask
(configs/experiment/tiling/mmci_hdab_segmentation.yaml: ``annotation_brown_*``), it
gives the refined evaluation labels (configs/lymph_nodes.yaml, ``ln_refined_labels``).

Written per slide under ``<artifact_path>/<the annotation mask's subfolder>/`` at
``output_downsample`` times the staining level's resolution, uint8 (the fraction of
each pixel covered, * 255), with the stained cells near the annotations (before the
selection) under ``stained_cells/`` for review.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pyvips
import ray
import tifffile
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.openslide import OpenSlide
from scipy import ndimage

from preprocessing.staining import stained_cells


def regions_overlapping(
    stained: np.ndarray, annotation: np.ndarray, reach: np.ndarray
) -> np.ndarray:
    """The connected regions of ``stained`` that overlap ``annotation``, within ``reach``."""
    labels, _ = ndimage.label(stained, structure=np.ones((3, 3), dtype=bool))
    hit = np.unique(labels[annotation & stained])
    return np.isin(labels, hit[hit > 0]) & reach


def annotation_boxes(
    annotation: np.ndarray, margin: int, coarse: int = 8
) -> list[tuple[slice, slice]]:
    """Boxes (annotation px) around the annotated regions, ``margin`` px wider.

    Found on a ``coarse`` x coarser view, so that a whole-slide mask needs no
    full-resolution labelling.
    """
    h, w = annotation.shape
    cropped = annotation[: h // coarse * coarse, : w // coarse * coarse]
    view = cropped.reshape(h // coarse, coarse, w // coarse, coarse).any(axis=(1, 3))
    regions, _ = ndimage.label(
        ndimage.binary_dilation(view, iterations=margin // coarse + 2)
    )
    return [
        (
            slice(box[0].start * coarse, min(box[0].stop * coarse, h)),
            slice(box[1].start * coarse, min(box[1].stop * coarse, w)),
        )
        for box in ndimage.find_objects(regions)
        if box is not None
    ]


@ray.remote(memory=6 * 1024**3)
def process_slide(
    item: tuple[str, str, str],
    mpp: float,
    output_path: Path,
    output_downsample: int,
    margin_um: float,
    chunk: int,
    stain: dict[str, float],
) -> None:
    annotation_path, slide_path, subfolder = item
    with tifffile.TiffFile(annotation_path) as tiff:
        annotation = tiff.pages[0].asarray() >= 128

    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        # Pixels need not be square (the IHC test scans: 0.19% taller than wide); the
        # stain is measured in the mean, the masks tagged per axis.
        mpp_x, mpp_y = (float(value) for value in slide.slide_resolution(level))
        level_mpp = (mpp_x + mpp_y) / 2
        downsample = round(slide.level_downsamples[level])
        width, height = slide.level_dimensions[level]
        # Annotation px -> level px; the annotation masks cover the slide at a whole
        # downsample of level 0 (their resolution tags can't be relied on).
        scale = round(slide.dimensions[0] / annotation.shape[1]) // downsample
        margin = round(margin_um / (level_mpp * scale))
        grow = stain.get("grow_um", 2.0) / level_mpp
        hole = np.sqrt(stain.get("hole_area_um2", 200.0)) / level_mpp
        pad = int(np.ceil(grow + hole)) + 8

        out_w, out_h = width // output_downsample, height // output_downsample
        images = {
            "refined": pyvips.Image.black(out_w, out_h),
            "stained_cells": pyvips.Image.black(out_w, out_h),
        }
        # Box edges (annotation px) on the output grid as well.
        align = max(1, output_downsample // scale)
        for rows, cols in annotation_boxes(annotation, margin):
            ay0, ax0 = rows.start // align * align, cols.start // align * align
            ay1 = min(rows.stop, out_h * output_downsample // scale)
            ax1 = min(cols.stop, out_w * output_downsample // scale)
            ay1, ax1 = (
                ay0 + (ay1 - ay0) // align * align,
                ax0 + (ax1 - ax0) // align * align,
            )
            if ay1 <= ay0 or ax1 <= ax0:
                continue
            inside = annotation[ay0:ay1, ax0:ax1]
            reach = ndimage.distance_transform_edt(~inside) <= margin
            y0, x0, y1, x1 = ay0 * scale, ax0 * scale, ay1 * scale, ax1 * scale

            stained = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            for cy in range(y0, y1, chunk):
                for cx in range(x0, x1, chunk):
                    h, w = min(chunk, y1 - cy), min(chunk, x1 - cx)
                    rgba = np.asarray(
                        slide.read_region(
                            ((cx - pad) * downsample, (cy - pad) * downsample),
                            level,
                            (w + 2 * pad, h + 2 * pad),
                        )
                    )
                    cells = stained_cells(rgba[..., :3], level_mpp, **stain)
                    cells &= rgba[..., 3] > 0
                    stained[cy - y0 : cy - y0 + h, cx - x0 : cx - x0 + w] = cells[
                        pad : pad + h, pad : pad + w
                    ]

            def upsample(mask: np.ndarray) -> np.ndarray:
                return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)

            kept = regions_overlapping(stained, upsample(inside), upsample(reach))
            for name, values in (("refined", kept), ("stained_cells", stained)):
                block = values.reshape(
                    (y1 - y0) // output_downsample,
                    output_downsample,
                    (x1 - x0) // output_downsample,
                    output_downsample,
                ).mean(axis=(1, 3))
                patch = pyvips.Image.new_from_array(
                    np.round(block * 255).astype(np.uint8)
                )
                images[name] = images[name].insert(
                    patch, x0 // output_downsample, y0 // output_downsample
                )

    for name, image in images.items():
        path = output_path / name / subfolder / f"{Path(slide_path).stem}.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_big_tiff(
            image.cast("uchar"),
            path=path,
            mpp_x=mpp_x * output_downsample,
            mpp_y=mpp_y * output_downsample,
        )


@with_cli_args(["+preprocessing=refined_annotation_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    """Refined masks for the annotation masks of ``annotation_mask_dir``.

    Every mask whose slide is in one of ``slide_dirs`` (by file stem), keeping the
    annotation masks' subfolders.
    """
    annotation_dir = Path(download_artifacts(config.annotation_mask_dir))
    slides = {
        path.stem: path
        for directory in config.slide_dirs
        for path in Path(directory).glob(f"*.{config.slide_suffix}")
    }
    items = [
        (
            str(mask),
            str(slides[mask.stem]),
            str(mask.parent.relative_to(annotation_dir)),
        )
        for mask in sorted(annotation_dir.rglob("*.tiff"))
        if mask.stem in slides
    ]
    print(f"{len(items)} annotated slides")

    with TemporaryDirectory() as output_dir:
        process_items(
            items,
            process_item=process_slide,
            fn_kwargs={
                "mpp": config.mpp,
                "output_path": Path(output_dir),
                "output_downsample": config.output_downsample,
                "margin_um": config.margin_um,
                "chunk": config.chunk,
                "stain": dict(config.stain),
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(
            local_dir=str(Path(output_dir) / "refined"),
            artifact_path=config.artifact_path,
        )
        logger.log_artifacts(
            local_dir=str(Path(output_dir) / "stained_cells"),
            artifact_path="stained_cells",
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
