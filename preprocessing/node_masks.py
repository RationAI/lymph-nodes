"""Lymph node masks: the node itself, as opposed to all tissue on the slide.

The tissue masks (preprocessing/tissue_masks.py) take in everything on the glass: the
node, but also the fat, fibrous and vascular tissue around it, where models trained on
cytokeratin-positive epithelium find many of their false positives (stained debris,
macrophages, pigment). This mask outlines the node, from a low-resolution level
(~8 um/px):

- node tissue is dense in haematoxylin: smoothed over ``smooth_um``, above the slide's
  Otsu threshold over its tissue (H + DAB above ``tissue``; the threshold at least
  ``min_density``), joined by paler connected tissue (above ``low`` x the threshold:
  medulla, sinuses);
- near-black pixels (ink, debris: every channel under ``dark``) are left out, as they
  would raise the threshold;
- closed over ``closing_um``, holes filled, pieces under ``min_area_mm2`` dropped;
- DAB-dense regions (above ``dab``) within ``margin_um`` of the node added: a
  metastasis at its edge replaces node tissue rather than looking like it;
- grown by ``margin_um``: the capsule and the subcapsular sinus, where isolated tumour
  cells arrive.

On the MMCI validation slides (MLflow run 6fbbe481) predictions outside it were 70-80%
of the false positives, and it missed 5 of the 862 annotated cell groups.

Written per slide as uint8 pyramid TIFFs (255 inside) under ``<artifact_path>/``, and
the mask's edge (``edge_um`` wide, for viewer overlays) under ``<edge_artifact_path>/``.
Tiled as ``node_coverage`` / ``node_grid`` (configs/experiment/tiling/*.yaml).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.openslide import OpenSlide
from scipy import ndimage

from preprocessing.staining import hed_densities
from preprocessing.white_balance import estimate_white_point


def otsu(values: np.ndarray) -> float:
    """Otsu's threshold of ``values`` over a 256-bin histogram."""
    hist, edges = np.histogram(values, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def node_mask(
    rgb: np.ndarray,
    um: float,
    tissue: float = 0.004,
    min_density: float = 0.008,
    low: float = 0.55,
    dab: float = 0.05,
    dark: int = 90,
    smooth_um: float = 24.0,
    closing_um: float = 40.0,
    min_area_mm2: float = 0.5,
    margin_um: float = 200.0,
) -> np.ndarray:
    """The lymph node of ``rgb`` (H, W, 3; white where unscanned) at ``um`` per pixel."""
    rgb = np.asarray(rgb, np.float32)
    ink = ndimage.binary_dilation(rgb.max(axis=-1) < dark, iterations=2)
    stains = hed_densities(rgb)
    haematoxylin = ndimage.gaussian_filter(stains[..., 0], smooth_um / um)
    brown = ndimage.gaussian_filter(stains[..., 2], smooth_um / um)

    stained = ((haematoxylin + brown) > tissue) & ~ink
    threshold = (
        max(otsu(haematoxylin[stained]), min_density) if stained.any() else min_density
    )
    core = (haematoxylin > threshold) & ~ink
    pale, _ = ndimage.label((haematoxylin > low * threshold) & ~ink)
    joined = np.unique(pale[core & (pale > 0)])
    node = np.isin(pale, joined[joined > 0]) | core
    node = ndimage.binary_closing(node, iterations=max(1, round(closing_um / um)))
    node = ndimage.binary_fill_holes(node)
    labels, n = ndimage.label(node)
    if n:
        areas = np.bincount(labels.ravel())[1:] * (um / 1000) ** 2
        node = np.isin(labels, np.flatnonzero(areas >= min_area_mm2) + 1)

    margin = max(1, round(margin_um / um))
    near = ndimage.binary_dilation(node, iterations=margin)
    regions, k = ndimage.label((brown > dab) & ~ink)
    if k:
        touching = np.unique(regions[near & (regions > 0)])
        node |= np.isin(regions, touching[touching > 0])
    return ndimage.binary_dilation(ndimage.binary_fill_holes(node), iterations=margin)


@ray.remote(memory=3 * 1024**3)
def process_slide(
    slide_path: str,
    mpp: float,
    output_path: Path,
    edge_path: Path,
    edge_um: float,
    white_balance: bool,
    rule: dict,
) -> None:
    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        mpp_x, mpp_y = slide.slide_resolution(level)
        rgba = np.asarray(
            slide.read_region((0, 0), level, slide.level_dimensions[level])
        )
        white = estimate_white_point(slide) if white_balance else None

    alpha = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3] * alpha + 255 * (1 - alpha)  # unscanned (transparent) = glass
    if white is not None:  # preprocessing.white_balance.apply_white_point, in floats
        rgb = np.clip(rgb * (255.0 / np.maximum(np.asarray(white), 100.0)), 0, 255)

    um = (mpp_x + mpp_y) / 2
    mask = node_mask(rgb, um, **rule)
    edge = mask & ~ndimage.binary_erosion(mask, iterations=max(1, round(edge_um / um)))
    name = Path(slide_path).with_suffix(".tiff").name
    for directory, array in ((output_path, mask), (edge_path, edge)):
        image = pyvips.Image.new_from_array(array.astype(np.uint8) * 255)
        write_big_tiff(image, path=directory / name, mpp_x=mpp_x, mpp_y=mpp_y)


@with_cli_args(["+preprocessing=node_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = hydra.utils.instantiate(config.dataset.slides)

    with TemporaryDirectory() as output_dir:
        masks, edges = Path(output_dir, "masks"), Path(output_dir, "edges")
        masks.mkdir()
        edges.mkdir()
        process_items(
            slides,
            process_item=process_slide,
            fn_kwargs={
                "mpp": config.mpp,
                "output_path": masks,
                "edge_path": edges,
                "edge_um": config.edge_um,
                "white_balance": config.get("white_balance", False),
                "rule": dict(config.rule),
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(local_dir=str(masks), artifact_path=config.artifact_path)
        logger.log_artifacts(
            local_dir=str(edges), artifact_path=config.edge_artifact_path
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
