"""Training-time tile augmentations for TileSegmentationDataset.

TileAugmentation varies what the lymph node evaluation varies and the training tiles
don't: orientation, stain intensity, the diffuse DAB background of some labs' slides
and focus. CellPasting adds what the training has no example of: tumour cells alone in
lymphoid tissue (isolated tumour cells), cut from the TMA tumour tiles.

Both work on optical densities split into haematoxylin, eosin and DAB by Ruifrok &
Johnston's stain vectors (as skimage.color.rgb2hed and
preprocessing/refined_annotation_masks.py), on the same scale: a DAB density here is
dab_density's (stained cells from ~0.022 on).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage


if TYPE_CHECKING:
    from collections.abc import Sequence


_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)
# rgb2hed's optical density scale: log(rgb) / log(1e-6).
_OD_SCALE = np.float32(-np.log(1e-6))
H, E, DAB = 0, 1, 2


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    """Stain densities (..., 3: H, E, DAB) of uint8 RGB pixels, unclipped (exact inverse of hed_to_rgb)."""
    od = -np.log(np.maximum(rgb.astype(np.float32), 1.0) / 255) / _OD_SCALE
    return od @ _HED_FROM_RGB


def hed_to_rgb(hed: np.ndarray) -> np.ndarray:
    """uint8 RGB pixels of stain densities (..., 3)."""
    od = hed @ _RGB_FROM_HED
    return np.clip(255 * np.exp(-od * _OD_SCALE), 0, 255).round().astype(np.uint8)


def tissue_mask(
    rgb: np.ndarray, min_density: float = 0.15, window: int = 15
) -> np.ndarray:
    """Tissue pixels by their optical density (summed over R, G, B), averaged over ``window`` px.

    A fallback for tiles without a tissue grid: on MMCI lymph node tiles it keeps 84 %
    of the tissue grid's tissue (the lymphoid tissue is pale, from OD ~0.1) and 10 % of
    its glass. Averaging counts the pale cytoplasm between nuclei as tissue.
    """
    od = -np.log(np.maximum(rgb.astype(np.float32), 1.0) / 255).sum(axis=-1)
    return ndimage.uniform_filter(od, window) > min_density


def cellular_mask(
    hed: np.ndarray, min_density: float = 0.01, window: int = 31
) -> np.ndarray:
    """Pixels with nuclei around: haematoxylin density averaged over ``window`` px.

    Where a pasted cell may go: on MMCI lymph node tiles it keeps 70 % of the tissue
    grid's tissue and 3 % of its glass — what it leaves out of the tissue is mostly the
    near-blank fat and glass at the tissue masks' edges.
    """
    return ndimage.uniform_filter(hed[..., H], window) > min_density


class TileAugmentation:
    """Random orientation, stain, DAB background and blur of a tile and its target grid.

    - a random quarter turn and flip (``flip``), of the target grid alike;
    - stain jitter (Tellez et al., 2018): each stain's density scaled by a random
      factor within 1 ± ``stain_scale`` (H, E, DAB) and, on tissue, shifted within
      ± ``stain_shift``;
    - with probability ``dab_background_p``, a diffuse DAB background over the tissue:
      smooth and faintly granular, up to ``dab_background`` density. FN Brno's negative
      lymph node tissue reads ~0.03 throughout (median of its tiles), MMCI's ~0.003,
      and models trained on MMCI flag it: background staining should mean nothing;
    - with probability ``blur_p``, a Gaussian blur of ``blur_sigma`` px (a range),
      for scans less sharp than the training ones (FN Brno's).

    Called with the tile (H, W, 3) uint8, its target grid (g, g), a numpy Generator
    and, optionally, the tiling's tissue (H, W) bool, narrowed to tissue_mask's (the
    tissue masks take in near-blank fat and glass at their edges); returns the tile and
    grid, contiguous.
    """

    def __init__(
        self,
        flip: bool = True,
        stain_scale: Sequence[float] = (0.15, 0.15, 0.3),
        stain_shift: Sequence[float] = (0.005, 0.005, 0.005),
        dab_background: float = 0.04,
        dab_background_p: float = 0.3,
        blur_p: float = 0.2,
        blur_sigma: Sequence[float] = (0.5, 1.5),
    ) -> None:
        self.flip = flip
        self.stain_scale = np.asarray(list(stain_scale), dtype=np.float32)
        self.stain_shift = np.asarray(list(stain_shift), dtype=np.float32)
        self.dab_background = dab_background
        self.dab_background_p = dab_background_p
        self.blur_p = blur_p
        self.blur_sigma = (float(blur_sigma[0]), float(blur_sigma[1]))

    def __call__(
        self,
        tile: np.ndarray,
        grid: np.ndarray,
        rng: np.random.Generator,
        tissue: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tissue = tissue_mask(tile) if tissue is None else tissue & tissue_mask(tile)
        if self.flip:
            turns = int(rng.integers(4))
            tile, grid = np.rot90(tile, turns), np.rot90(grid, turns)
            tissue = np.rot90(tissue, turns)
            if rng.random() < 0.5:
                tile, grid, tissue = tile[:, ::-1], grid[:, ::-1], tissue[:, ::-1]

        hed = rgb_to_hed(tile)
        # Feathered: no blocks of the tissue grid or pinholes of the pixel rule.
        weight = ndimage.gaussian_filter(tissue.astype(np.float32), 4.0)
        scale = rng.uniform(1 - self.stain_scale, 1 + self.stain_scale)
        shift = rng.uniform(-self.stain_shift, self.stain_shift)
        hed = hed * scale + shift * weight[..., np.newaxis]
        if rng.random() < self.dab_background_p:
            hed[..., DAB] += self._background(tile.shape[:2], rng) * weight
        tile = hed_to_rgb(hed)

        if rng.random() < self.blur_p:
            sigma = rng.uniform(*self.blur_sigma)
            tile = (
                ndimage.gaussian_filter(tile.astype(np.float32), (sigma, sigma, 0))
                .round()
                .astype(np.uint8)
            )
        return np.ascontiguousarray(tile), np.ascontiguousarray(grid)

    def _background(
        self, shape: tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """A DAB density field: smooth (4 x 4 control points), faintly granular."""
        h, w = shape
        smooth = ndimage.zoom(
            rng.uniform(0.3, 1.0, size=(4, 4)).astype(np.float32),
            (h / 4, w / 4),
            order=1,
        )[:h, :w]
        grain = ndimage.gaussian_filter(
            rng.standard_normal((h, w)).astype(np.float32), 2.0
        )
        grain /= grain.std() + 1e-6
        amplitude = rng.uniform(0, self.dab_background)
        return amplitude * smooth * np.clip(1 + 0.15 * grain, 0, None)


class CellPasting:
    """Synthetic isolated tumour cells: TMA tumour cells pasted into negative lymph node tiles.

    The training's positives are all TMA tumour — sheets of epithelium — while lymph
    node metastases are mostly isolated tumour cells, one to a few cytokeratin-positive
    cells among lymphocytes, which the models trained so far miss. A negative lymph node
    tile gets, with probability ``p``, cells cut from a TMA tumour tile (``donor``):

    - 1 to ``max_cells`` single cells, each a disk of radius ``cell_radius_um`` (a
      range; slightly irregular and elliptic) around a nucleus with stained cytoplasm
      around it (mean DAB density above ``donor_dab``), or, with probability
      ``cluster_p``, one small group: a disk of ``cluster_radius_um`` around a stained
      point;
    - each pasted at a random place on the tissue with nuclei around (cellular_mask),
      blended in optical density (soft edge), with the donor's haematoxylin scaled to
      the tile's;
    - the target grid becomes each patch's covered fraction, as the cytokeratin grids.

    With probability ``negative_p``, the brownest spot of another negative lymph node
    tile (``brown``: benign brown — pigment, background stain, dendritic cells) is
    pasted the same way and stays negative, so that the pasting itself tells nothing.

    Called (after a donor tile is read) with the tile (H, W, 3) uint8, its target grid
    (g, g), the slide's resolution, a numpy Generator and, optionally, the tiling's
    tissue (H, W) bool, narrowed to tissue_mask's as in TileAugmentation; returns the
    tile and grid.
    """

    def __init__(
        self,
        p: float = 0.3,
        negative_p: float = 0.2,
        max_cells: int = 3,
        cluster_p: float = 0.3,
        cell_radius_um: Sequence[float] = (5.0, 8.0),
        cluster_radius_um: Sequence[float] = (10.0, 20.0),
        donor_dab: float = 0.05,
        donor_cohorts: Sequence[str] = ("mmci_tmas",),
    ) -> None:
        self.p = p
        self.negative_p = negative_p
        self.max_cells = max_cells
        self.cluster_p = cluster_p
        self.cell_radius_um = (float(cell_radius_um[0]), float(cell_radius_um[1]))
        self.cluster_radius_um = (
            float(cluster_radius_um[0]),
            float(cluster_radius_um[1]),
        )
        self.donor_dab = donor_dab
        self.donor_cohorts = [str(cohort) for cohort in donor_cohorts]

    def __call__(
        self,
        tile: np.ndarray,
        grid: np.ndarray,
        mpp: float,
        rng: np.random.Generator,
        donor: np.ndarray | None = None,
        brown: np.ndarray | None = None,
        tissue: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tissue = tissue_mask(tile) if tissue is None else tissue & tissue_mask(tile)
        hed = rgb_to_hed(tile)
        tissue &= cellular_mask(hed)
        if brown is not None:
            cut = self._cut_brown(rgb_to_hed(brown), tissue_mask(brown), mpp, rng)
            if cut is not None:
                _paste(hed, *cut, tissue, rng)
        if donor is not None:
            pasted = np.zeros(tile.shape[:2], dtype=bool)
            donor_hed, donor_tissue = rgb_to_hed(donor), tissue_mask(donor)
            cluster = rng.random() < self.cluster_p
            count = 1 if cluster else int(rng.integers(1, self.max_cells + 1))
            reference = np.percentile(hed[..., H][tissue], 90) if tissue.any() else None
            for _ in range(count):
                cut = self._cut_cell(donor_hed, donor_tissue, mpp, rng, cluster)
                if cut is None:
                    break
                window, alpha = cut
                if reference is not None:
                    level = np.percentile(window[..., H][alpha > 0.5], 90)
                    window[..., H] *= np.clip(reference / max(level, 1e-6), 0.6, 1.5)
                placed = _paste(hed, window, alpha, tissue, rng)
                if placed is not None:
                    pasted |= placed
            if pasted.any():
                g = grid.shape[0]
                h, w = tile.shape[:2]
                coverage = pasted.reshape(g, h // g, g, w // g).mean(axis=(1, 3))
                grid = np.maximum(grid, coverage.astype(np.float32))
        return hed_to_rgb(hed), grid

    def _cut_cell(
        self,
        hed: np.ndarray,
        tissue: np.ndarray,
        mpp: float,
        rng: np.random.Generator,
        cluster: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A cell (or group) of the donor: its densities and blending weights, or None."""
        radius = (
            rng.uniform(*(self.cluster_radius_um if cluster else self.cell_radius_um))
            / mpp
        )
        radius = min(radius, 0.3 * min(hed.shape[:2]))  # fits a tile at any resolution
        sigma = 1.0 / mpp  # ~1 um
        dab = ndimage.gaussian_filter(hed[..., DAB], sigma)
        if cluster:
            candidates = dab > 2 * self.donor_dab
        else:
            nuclei = ndimage.gaussian_filter(hed[..., H], sigma)
            size = round(8 / mpp) | 1  # one peak per ~8 um
            peaks = (nuclei == ndimage.maximum_filter(nuclei, size=size)) & (
                nuclei > np.percentile(nuclei, 75)
            )
            around = ndimage.uniform_filter(dab, size=round(2 * radius) | 1)
            candidates = peaks & (around > self.donor_dab)
        return _cut(hed, tissue, candidates, radius, rng)

    def _cut_brown(
        self, hed: np.ndarray, tissue: np.ndarray, mpp: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The donor's brownest spot (DAB density above ``donor_dab``), a disk of a cell to a small group's radius."""
        radius = (
            rng.uniform(self.cell_radius_um[0], self.cluster_radius_um[0] * 1.5) / mpp
        )
        radius = min(radius, 0.3 * min(hed.shape[:2]))
        dab = ndimage.gaussian_filter(hed[..., DAB], 2.0 / mpp)
        r = int(np.ceil(radius * 1.3)) + 2
        inner = np.zeros(dab.shape, dtype=bool)
        inner[r:-r, r:-r] = True
        if not inner.any() or dab[inner].max() < self.donor_dab:
            return None
        best = np.unravel_index(np.argmax(np.where(inner, dab, -np.inf)), dab.shape)
        candidates = np.zeros(dab.shape, dtype=bool)
        candidates[best] = True
        return _cut(hed, tissue, candidates, radius, rng)


def _cut(
    hed: np.ndarray,
    tissue: np.ndarray,
    candidates: np.ndarray,
    radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """A disk of ``radius`` (irregular, elliptic) around a random candidate pixel.

    Returns the window of densities (a copy) and its blending weights (soft edge, on
    the donor's ``tissue`` only: no glass pasted), or None if no candidate is far
    enough from the edge.
    """
    r = int(np.ceil(radius * 1.3)) + 2
    h, w = candidates.shape
    inner = np.zeros_like(candidates)
    inner[r : h - r, r : w - r] = True
    ys, xs = np.nonzero(candidates & inner)
    if len(ys) == 0:
        return None
    i = int(rng.integers(len(ys)))
    cy, cx = int(ys[i]), int(xs[i])
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    elongation = rng.uniform(0.8, 1.25)
    angle = rng.uniform(0, np.pi)
    u = xx * np.cos(angle) + yy * np.sin(angle)
    v = -xx * np.sin(angle) + yy * np.cos(angle)
    theta = np.arctan2(v, u)
    wobble = sum(
        rng.normal(0, 0.06) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
        for k in (2, 3, 5)
    )
    inside = np.hypot(u / elongation, v * elongation) <= radius * (1 + wobble)
    inside &= tissue[cy - r : cy + r + 1, cx - r : cx + r + 1]
    alpha = np.clip(ndimage.gaussian_filter(inside.astype(np.float32), 0.8), 0, 1)
    if not (alpha > 0.5).any():
        return None
    window = hed[cy - r : cy + r + 1, cx - r : cx + r + 1].copy()
    return window, alpha


def _paste(
    hed: np.ndarray,
    window: np.ndarray,
    alpha: np.ndarray,
    tissue: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 20,
) -> np.ndarray | None:
    """Blends ``window`` into ``hed`` in place at a random place on the tissue.

    Returns the pasted pixels (alpha above 0.5) as a mask of the tile, or None if no
    place (mostly tissue, within the tile) was found.
    """
    size = window.shape[0]
    h, w = tissue.shape
    if size >= min(h, w):
        return None
    inside = alpha > 0.5
    # Centres on tissue, the window within the tile.
    centres = np.zeros_like(tissue)
    centres[
        size // 2 : h - size + size // 2 + 1, size // 2 : w - size + size // 2 + 1
    ] = True
    ys, xs = np.nonzero(centres & tissue)
    if len(ys) == 0:
        return None
    for _ in range(attempts):
        i = int(rng.integers(len(ys)))
        y, x = int(ys[i]) - size // 2, int(xs[i]) - size // 2
        if tissue[y : y + size, x : x + size][inside].mean() < 0.8:
            continue
        region = hed[y : y + size, x : x + size]
        region += alpha[..., np.newaxis] * (window - region)
        mask = np.zeros((h, w), dtype=bool)
        mask[y : y + size, x : x + size] = inside
        return mask
    return None
