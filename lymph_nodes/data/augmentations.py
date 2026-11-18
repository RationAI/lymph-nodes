"""Training-time tile augmentations for TileSegmentationDataset.

TileAugmentation varies what the lymph node evaluation varies and the training tiles
don't: orientation, stain intensity, faint staining, the diffuse DAB background of some
labs' slides and focus. CellPasting adds what the training has no example of: tumour cells alone in
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
    from collections.abc import Mapping, Sequence


_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)
# rgb2hed's optical density scale: log(rgb) / log(1e-6).
_OD_SCALE = np.float32(-np.log(1e-6))
H, E, DAB = 0, 1, 2
# DAB density of stained cells (preprocessing/staining.py's low threshold).
_STAINED = 0.022


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    """Stain densities (..., 3: H, E, DAB) of uint8 RGB pixels, unclipped (exact inverse of hed_to_rgb)."""
    od = -np.log(np.maximum(rgb.astype(np.float32), 1.0) / 255) / _OD_SCALE
    return od @ _HED_FROM_RGB


def hed_to_rgb(hed: np.ndarray) -> np.ndarray:
    """uint8 RGB pixels of stain densities (..., 3)."""
    od = hed @ _RGB_FROM_HED
    return np.clip(255 * np.exp(-od * _OD_SCALE), 0, 255).round().astype(np.uint8)


def _cell_starts(size: int, cells: int) -> np.ndarray:
    """The first pixel of each of ``cells`` cells across ``size`` pixels.

    Rounded edges, as the tiling's grids (preprocessing.tiling_blocks.grid_coverage):
    exact when ``size`` is a multiple of ``cells`` (224 px at grid 16 or 32), else cells
    of alternating widths (grid 64: 3.5 px -> 4, 3, 3, 4, ...).
    """
    return np.linspace(0, size, cells + 1).round().astype(int)[:-1]


def cells_to_pixels(cells: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A (g, g) grid as an (H, W) image: each pixel takes its cell's value."""
    rows, cols = (
        np.searchsorted(_cell_starts(n, g), np.arange(n), side="right") - 1
        for n, g in zip(shape[:2], cells.shape, strict=True)
    )
    return cells[rows[:, np.newaxis], cols[np.newaxis, :]]


def pixels_to_cells(pixels: np.ndarray, cells: int) -> np.ndarray:
    """The mean of an (H, W) image over each cell of a ``cells`` x ``cells`` grid."""
    h, w = pixels.shape[:2]
    rows, cols = _cell_starts(h, cells), _cell_starts(w, cells)
    sums = np.add.reduceat(
        np.add.reduceat(pixels.astype(np.float64), rows, axis=0), cols, axis=1
    )
    return sums / np.outer(np.diff(rows, append=h), np.diff(cols, append=w))


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
    - with probability ``dab_fade_p``, faint staining: the DAB faded so that the
      median DAB density of the stained pixels (of the positive patches, if the tile
      has any) falls within ``dab_fade_to``. The missed isolated tumour cells are the
      faint ones (median DAB ~0.031 of the faintest third of the annotated groups,
      against ~0.08 of the TMA breast tumour cells); the stained-cell rule starts at
      ~0.022, so the faded cells stay stained. Each pixel's optical density is scaled
      by ``1 - (1 - f) * s``, ``s`` its DAB share of the H and DAB densities, ``f`` at
      least ``min_fade``: a stained pixel turns a paler brown of its own hue and a
      nucleus stays. (Scaling the DAB density alone, along the fixed DAB vector, turns
      strong brown olive or yellow: these scanners' brown is off that vector.) Negative
      tiles fade alike (their pigment and brown background), so that faint staining
      tells nothing;
    - with probability ``dab_background_p``, a diffuse DAB background over the tissue:
      smooth and faintly granular, up to ``dab_background`` density. FN Brno's negative
      lymph node tissue reads ~0.03 throughout (median of its tiles), MMCI's ~0.003,
      and models trained on MMCI flag it: background staining should mean nothing;
    - with probability ``blur_p``, a Gaussian blur of ``blur_sigma`` px (a range),
      for scans less sharp than the training ones (FN Brno's) and slides partly out of
      focus; of the blurred tiles a share ``blur_partial_p`` only on one side of a soft
      straight edge at a random angle and place — the seam between a scanner's fields
      focused at different depths (IHC 2023 val: the clusters the models missed lay in
      fields ~0.4-0.6 um softer than their neighbours) — so that blur is no cue of the
      whole tile.

    Called with the tile (H, W, 3) uint8, its target grid (g, g), a numpy Generator
    and, optionally, the tiling's tissue (H, W) bool, narrowed to tissue_mask's (the
    tissue masks take in near-blank fat and glass at their edges), and an ignore grid
    (g, g), turned with the target grid; returns the tile, grid and ignore grid (None
    without one), contiguous.
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
        blur_partial_p: float = 0.0,
        dab_fade_p: float = 0.0,
        dab_fade_to: Sequence[float] = (0.03, 0.05),
        min_fade: float = 0.25,
    ) -> None:
        self.flip = flip
        self.stain_scale = np.asarray(list(stain_scale), dtype=np.float32)
        self.stain_shift = np.asarray(list(stain_shift), dtype=np.float32)
        self.dab_background = dab_background
        self.dab_background_p = dab_background_p
        self.blur_p = blur_p
        self.blur_sigma = (float(blur_sigma[0]), float(blur_sigma[1]))
        self.blur_partial_p = blur_partial_p
        self.dab_fade_p = dab_fade_p
        self.dab_fade_to = (float(dab_fade_to[0]), float(dab_fade_to[1]))
        self.min_fade = min_fade

    def __call__(
        self,
        tile: np.ndarray,
        grid: np.ndarray,
        rng: np.random.Generator,
        tissue: np.ndarray | None = None,
        ignore: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        tissue = tissue_mask(tile) if tissue is None else tissue & tissue_mask(tile)
        if self.flip:
            turns = int(rng.integers(4))
            tile, grid = np.rot90(tile, turns), np.rot90(grid, turns)
            tissue = np.rot90(tissue, turns)
            ignore = None if ignore is None else np.rot90(ignore, turns)
            if rng.random() < 0.5:
                tile, grid, tissue = tile[:, ::-1], grid[:, ::-1], tissue[:, ::-1]
                ignore = None if ignore is None else ignore[:, ::-1]

        hed = rgb_to_hed(tile)
        # Feathered: no blocks of the tissue grid or pinholes of the pixel rule.
        weight = ndimage.gaussian_filter(tissue.astype(np.float32), 4.0)
        scale = rng.uniform(1 - self.stain_scale, 1 + self.stain_scale)
        shift = rng.uniform(-self.stain_shift, self.stain_shift)
        hed = hed * scale + shift * weight[..., np.newaxis]
        if rng.random() < self.dab_fade_p:
            hed = rgb_to_hed(self._fade(hed_to_rgb(hed), grid, tissue, rng))
        if rng.random() < self.dab_background_p:
            hed[..., DAB] += self._background(tile.shape[:2], rng) * weight
        tile = hed_to_rgb(hed)

        if rng.random() < self.blur_p:
            sigma = rng.uniform(*self.blur_sigma)
            blurred = ndimage.gaussian_filter(
                tile.astype(np.float32), (sigma, sigma, 0)
            )
            # Drawn only when on: the random stream of the scenarios without it stays.
            if self.blur_partial_p and rng.random() < self.blur_partial_p:
                seam = _seam(tile.shape[:2], rng)[..., np.newaxis]
                blurred = seam * blurred + (1 - seam) * tile
            tile = blurred.round().astype(np.uint8)
        return (
            np.ascontiguousarray(tile),
            np.ascontiguousarray(grid),
            None if ignore is None else np.ascontiguousarray(ignore),
        )

    def _fade(
        self,
        tile: np.ndarray,
        grid: np.ndarray,
        tissue: np.ndarray,
        rng: np.random.Generator,
        min_pixels: int = 50,
        steps: int = 3,
    ) -> np.ndarray:
        """The tile with its stained pixels faded to ``dab_fade_to`` (unchanged if fainter)."""
        hed = rgb_to_hed(tile)
        dab = hed[..., DAB]
        stained = tissue & (dab > _STAINED)
        positive = cells_to_pixels(grid > 0.5, dab.shape)
        if (stained & positive).sum() >= min_pixels:
            stained &= positive
        if stained.sum() < min_pixels:
            return tile
        target = rng.uniform(*self.dab_fade_to)
        return fade_dab(tile, stained, target, self.min_fade, steps)

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


def fade_dab(
    tile: np.ndarray,
    stained: np.ndarray,
    target: float,
    min_fade: float = 0.25,
    steps: int = 3,
) -> np.ndarray:
    """``tile`` (RGB uint8) faded so that its ``stained`` pixels' median DAB density is ``target``.

    Unchanged if already as faint. Hue-preserving (TileAugmentation): each pixel's
    optical density scaled by ``1 - (1 - f) * s``, ``s`` its DAB share of the H and DAB
    densities, the factor ``f`` at least ``min_fade``.
    """
    hed = rgb_to_hed(tile)
    level = float(np.median(hed[..., DAB][stained]))
    if level <= target:
        return tile
    od = -np.log(np.maximum(tile.astype(np.float32), 1.0) / 255)
    densities = np.clip(hed, 0, None)
    share = densities[..., DAB] / np.maximum(
        densities[..., H] + densities[..., DAB], 1e-6
    )
    # Mixed pixels fade less than the factor: a few corrections of it.
    factor, faded = max(target / level, min_fade), tile
    for _ in range(steps):
        scaled = od * (1 - (1 - factor) * share)[..., np.newaxis]
        faded = np.clip(255 * np.exp(-scaled), 0, 255).round().astype(np.uint8)
        reached = float(np.median(rgb_to_hed(faded)[..., DAB][stained]))
        if reached <= target * 1.05 or factor <= min_fade:
            break
        factor = max(factor * target / reached, min_fade)
    return faded


def _seam(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Weights (H, W) in [0, 1]: 1 on one side of a straight edge, 0 on the other.

    The edge at a random angle through a random point of the middle half of the tile,
    soft over a random 2 to 15 px (scanner fields meet along straight lines; the
    focus changes over a few um).
    """
    h, w = shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    cy, cx = rng.uniform(0.25, 0.75) * h, rng.uniform(0.25, 0.75) * w
    softness = rng.uniform(2.0, 15.0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    distance = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    return (0.5 * (1 + np.tanh(distance / softness))).astype(np.float32)


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
      point, or, with probability ``fragment_p``, one larger fragment: the donor's
      stained epithelium (with the nuclei inside it) around a stained point, cut to an
      irregular ellipse ``fragment_um`` long and up to 3x narrower — nests, cords and
      branching strands of tens of cells, the shape the stained tissue has. Small
      groups alone left the models flagging only the edges and loose cells of larger
      clusters in lymph nodes: in lymph node tissue they had never seen a positive
      larger than a handful of cells;
    - each pasted at a random place on the tissue with nuclei around (cellular_mask,
      haematoxylin density above ``placement_density``: lower takes in sparser stroma
      too), blended in optical density (soft edge), with the donor's haematoxylin scaled
      to the tile's;
    - the target grid becomes each patch's covered fraction, as the cytokeratin grids.

    With probability ``negative_p``, the brownest spot of another negative lymph node
    tile (``brown``: benign brown — pigment, background stain, dendritic cells) is
    pasted the same way and stays negative, so that the pasting itself tells nothing.

    ``within_node``: both are pasted only within the lymph node (TileSegmentationDataset:
    the tiling's node_grid narrows the placement, and tiles without node coverage get
    none) — isolated tumour cells sit in the node, not in the fat or capsule around it.

    For more varied pastes (each off by default, the random stream then as before):

    - ``pastes``: how many pastes (a range) a tile gets, each of its own kind (cells,
      group or fragment) and from its own donor tile (TileSegmentationDataset reads one
      per paste);
    - ``min_gap_um``: everything pasted into a tile (the brown spot too) at least this
      far apart — no paste over another;
    - ``cell_fade_p``: each paste's DAB faded with this probability, alone, to a median
      DAB density in ``cell_fade_to`` (as TileAugmentation's fade, hue-preserving): faint
      cells in a tile otherwise as stained as it was, the missed isolated tumour cells'
      look (TileAugmentation's fade takes the whole tile);
    - ``edge_p``: where a tile holds the lymph node's outline, each paste placed with this
      probability within ``edge_um`` inside it — the subcapsular sinus, where metastatic
      cells arrive with the afferent lymph;
    - ``donor_weights``: the donor tiles drawn by group, a group's share by its weight,
      the groups slide-name prefixes (e.g. TMA kinds ``FIN``, ``TNBC``, ``KOS``; donors
      of no group never drawn) — TileSegmentationDataset draws them. By default every
      donor tile alike, so the kinds with the most tumour tiles dominate.

    Decoys, to tell whether a model flags the pasting rather than the cell: with ``cut``
    ``"unstained"``, single nucleated cells without any stain around them (DAB below the
    stained-cell level), cut from any tile of the donor cohorts (TileSegmentationDataset:
    every partition, not only the tumour tiles) — the same scanner, blending and edges
    as the tumour cells of those cohorts, no tumour; with ``mark`` false, pastes leave
    the target grid as it was (negatives). ``donor_pool`` narrows the donor tiles:
    ``"cancer"`` (the tumour tiles: unstained cells among tumour, CK-negative tumour cells
    possibly among them) or ``"no_cytokeratin"`` (tiles without any cytokeratin: stroma
    and benign cells); ``"default"``: the tumour tiles for tumour cells, every tile for
    unstained ones.

    Called (after a donor tile is read) with the tile (H, W, 3) uint8, its target grid
    (g, g), the slide's resolution, a numpy Generator, the donor tile (or one per
    paste), optionally a brown donor, the tiling's tissue (H, W) bool, narrowed to
    tissue_mask's as in TileAugmentation, and the lymph node (H, W) bool, for
    ``edge_p``; returns the tile and grid.
    """

    def __init__(
        self,
        p: float = 0.3,
        negative_p: float = 0.2,
        max_cells: int = 3,
        cluster_p: float = 0.3,
        cell_radius_um: Sequence[float] = (5.0, 8.0),
        cluster_radius_um: Sequence[float] = (10.0, 20.0),
        fragment_p: float = 0.0,
        fragment_um: Sequence[float] = (30.0, 80.0),
        donor_dab: float = 0.05,
        donor_cohorts: Sequence[str] = ("mmci_tmas",),
        placement_density: float = 0.01,
        within_node: bool = True,
        pastes: Sequence[int] = (1, 1),
        min_gap_um: float | None = None,
        cell_fade_p: float = 0.0,
        cell_fade_to: Sequence[float] = (0.03, 0.05),
        min_fade: float = 0.25,
        edge_p: float = 0.0,
        edge_um: float = 50.0,
        donor_weights: Mapping[str, float] | None = None,
        cut: str = "stained",
        mark: bool = True,
        donor_pool: str = "default",
    ) -> None:
        if cut not in ("stained", "unstained"):
            raise ValueError(f"cut: 'stained' or 'unstained', not {cut!r}")
        if donor_pool not in ("default", "cancer", "no_cytokeratin"):
            raise ValueError(
                f"donor_pool: 'default', 'cancer' or 'no_cytokeratin', not {donor_pool!r}"
            )
        self.p = p
        self.negative_p = negative_p
        self.max_cells = max_cells
        self.cluster_p = cluster_p
        self.fragment_p = fragment_p
        self.fragment_um = (float(fragment_um[0]), float(fragment_um[1]))
        self.cell_radius_um = (float(cell_radius_um[0]), float(cell_radius_um[1]))
        self.cluster_radius_um = (
            float(cluster_radius_um[0]),
            float(cluster_radius_um[1]),
        )
        self.donor_dab = donor_dab
        self.donor_cohorts = [str(cohort) for cohort in donor_cohorts]
        self.placement_density = placement_density
        self.within_node = within_node
        self.pastes = (int(pastes[0]), int(pastes[1]))
        self.min_gap_um = min_gap_um
        self.cell_fade_p = cell_fade_p
        self.cell_fade_to = (float(cell_fade_to[0]), float(cell_fade_to[1]))
        self.min_fade = min_fade
        self.edge_p = edge_p
        self.edge_um = edge_um
        self.donor_weights = None if donor_weights is None else dict(donor_weights)
        self.cut = cut
        self.mark = mark
        self.donor_pool = donor_pool

    def __call__(
        self,
        tile: np.ndarray,
        grid: np.ndarray,
        mpp: float,
        rng: np.random.Generator,
        donor: np.ndarray | Sequence[np.ndarray] | None = None,
        brown: np.ndarray | None = None,
        tissue: np.ndarray | None = None,
        node: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tissue = tissue_mask(tile) if tissue is None else tissue & tissue_mask(tile)
        hed = rgb_to_hed(tile)
        tissue &= cellular_mask(hed, self.placement_density)
        # What is pasted so far, kept apart from the next pastes with min_gap_um.
        occupied = (
            None if self.min_gap_um is None else np.zeros(tile.shape[:2], dtype=bool)
        )
        if brown is not None:
            cut = self._cut_brown(rgb_to_hed(brown), tissue_mask(brown), mpp, rng)
            if cut is not None:
                placed = _paste(
                    hed, *cut, tissue, rng, forbidden=self._apart(occupied, mpp)
                )
                if placed is not None and occupied is not None:
                    occupied |= placed
        donors = (
            []
            if donor is None
            else [donor]
            if isinstance(donor, np.ndarray)
            else list(donor)
        )
        if donors:
            pasted = np.zeros(tile.shape[:2], dtype=bool)
            reference = np.percentile(hed[..., H][tissue], 90) if tissue.any() else None
            edge = (
                _edge_band(node, self.edge_um / mpp)
                if self.edge_p and node is not None
                else None
            )
            for donor_tile in donors:
                donor_hed, donor_tissue = (
                    rgb_to_hed(donor_tile),
                    tissue_mask(donor_tile),
                )
                kind = rng.random()
                unstained = self.cut == "unstained"
                fragment = not unstained and kind < self.fragment_p
                cluster = (
                    not unstained
                    and not fragment
                    and kind < self.fragment_p + self.cluster_p
                )
                count = (
                    1
                    if cluster or fragment
                    else int(rng.integers(1, self.max_cells + 1))
                )
                where = tissue
                if edge is not None and rng.random() < self.edge_p:
                    near = tissue & edge
                    where = near if near.any() else tissue
                for _ in range(count):
                    cut = (
                        self._cut_fragment(donor_hed, donor_tissue, mpp, rng)
                        if fragment
                        else self._cut_cell(
                            donor_hed, donor_tissue, mpp, rng, cluster, unstained
                        )
                    )
                    if cut is None:
                        break
                    window, alpha = cut
                    if reference is not None:
                        level = np.percentile(window[..., H][alpha > 0.5], 90)
                        window[..., H] *= np.clip(
                            reference / max(level, 1e-6), 0.6, 1.5
                        )
                    if self.cell_fade_p and rng.random() < self.cell_fade_p:
                        window = self._fade_window(window, alpha, rng)
                    placed = _paste(
                        hed,
                        window,
                        alpha,
                        where,
                        rng,
                        forbidden=self._apart(occupied, mpp),
                    )
                    if placed is not None:
                        pasted |= placed
                        if occupied is not None:
                            occupied |= placed
            if pasted.any() and self.mark:
                coverage = pixels_to_cells(pasted, grid.shape[0])
                grid = np.maximum(grid, coverage.astype(np.float32))
        return hed_to_rgb(hed), grid

    def _apart(self, occupied: np.ndarray | None, mpp: float) -> np.ndarray | None:
        """Where a paste may not go: within ``min_gap_um`` of the pasted (None: anywhere)."""
        if occupied is None or not occupied.any():
            return None
        gap = max(1, round(float(self.min_gap_um or 0) / mpp))
        return ndimage.binary_dilation(occupied, iterations=gap)

    def _fade_window(
        self, window: np.ndarray, alpha: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """A cut's densities with its stained pixels faded to ``cell_fade_to``."""
        stained = (alpha > 0.5) & (window[..., DAB] > _STAINED)
        if stained.sum() < 10:
            return window
        rgb = hed_to_rgb(window)
        target = rng.uniform(*self.cell_fade_to)
        return rgb_to_hed(fade_dab(rgb, stained, target, self.min_fade))

    def _cut_cell(
        self,
        hed: np.ndarray,
        tissue: np.ndarray,
        mpp: float,
        rng: np.random.Generator,
        cluster: bool,
        unstained: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A cell (or group) of the donor: its densities and blending weights, or None.

        ``unstained``: a nucleated cell with no stain around it (a decoy).
        """
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
            if unstained:
                stain = ndimage.maximum_filter(dab, size=round(2 * radius) | 1)
                candidates = peaks & (stain < _STAINED)
            else:
                candidates = peaks & (around > self.donor_dab)
        return _cut(hed, tissue, candidates, radius, rng)

    def _cut_fragment(
        self, hed: np.ndarray, tissue: np.ndarray, mpp: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A fragment of the donor's stained epithelium: densities and blending weights, or None.

        The stained tissue (DAB density above ``donor_dab``, closed over ~2 um and its
        holes filled: the unstained nuclei inside the cells) within an irregular ellipse
        around a strongly stained point, ``fragment_um`` long (at most 0.6 of the tile)
        and 1 to 3x narrower; of it, the piece joined to that point, so that its outline
        is the epithelium's (or the donor tile's edge). Cropped to the piece.
        """
        length = rng.uniform(*self.fragment_um) / mpp
        length = min(length, 0.6 * min(hed.shape[:2]))  # _paste needs it within a tile
        width = length / rng.uniform(1.0, 3.0)
        dab = ndimage.gaussian_filter(hed[..., DAB], 1.0 / mpp)
        stained = ndimage.binary_closing(
            dab > self.donor_dab, iterations=max(1, round(2.0 / mpp))
        )
        stained = ndimage.binary_fill_holes(stained) & tissue
        ys, xs = np.nonzero(stained & (dab > 2 * self.donor_dab))
        if len(ys) == 0:
            return None
        # Padded, so that a point near the donor's edge is cut at the edge.
        r = int(np.ceil(length / 2 * 1.3)) + 2
        stained = np.pad(stained, r)
        hed = np.pad(hed, ((r, r), (r, r), (0, 0)), mode="edge")
        i = int(rng.integers(len(ys)))
        cy, cx = int(ys[i]) + r, int(xs[i]) + r
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
        angle = rng.uniform(0, np.pi)
        u = xx * np.cos(angle) + yy * np.sin(angle)
        v = -xx * np.sin(angle) + yy * np.cos(angle)
        theta = np.arctan2(v, u)
        wobble = sum(
            rng.normal(0, 0.1) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
            for k in (2, 3, 5)
        )
        inside = np.hypot(u / (length / 2), v / (width / 2)) <= 1 + wobble
        inside &= stained[cy - r : cy + r + 1, cx - r : cx + r + 1]
        pieces, _ = ndimage.label(inside)
        if pieces[r, r] == 0:
            return None
        inside = pieces == pieces[r, r]
        alpha = np.clip(ndimage.gaussian_filter(inside.astype(np.float32), 0.8), 0, 1)
        # Square (as _paste places it), around the piece with a pixel or two of margin;
        # padded first, so that the square fits however the piece lies in the window.
        rows, cols = np.nonzero(alpha > 0.01)
        half = max(rows.max() - rows.min(), cols.max() - cols.min()) // 2 + 2
        cy0 = (rows.min() + rows.max()) // 2 + half
        cx0 = (cols.min() + cols.max()) // 2 + half
        window = np.pad(
            hed[cy - r : cy + r + 1, cx - r : cx + r + 1],
            ((half, half), (half, half), (0, 0)),
            mode="edge",
        )
        alpha = np.pad(alpha, half)
        crop = np.s_[cy0 - half : cy0 + half + 1, cx0 - half : cx0 + half + 1]
        return window[crop].copy(), alpha[crop]

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


def _edge_band(node: np.ndarray, width: float) -> np.ndarray | None:
    """The lymph node's pixels within ``width`` px of its outline in the tile, or None.

    None where the tile holds no outline (all node or none): the tile's own edges are
    no outline.
    """
    if node.all() or not node.any():
        return None
    return node & (ndimage.distance_transform_edt(node) <= width)


def _paste(
    hed: np.ndarray,
    window: np.ndarray,
    alpha: np.ndarray,
    tissue: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 20,
    forbidden: np.ndarray | None = None,
) -> np.ndarray | None:
    """Blends ``window`` into ``hed`` in place at a random place on the tissue.

    Returns the pasted pixels (alpha above 0.5) as a mask of the tile, or None if no
    place (mostly tissue, within the tile, none of its pixels ``forbidden``) was found.
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
        if (
            forbidden is not None
            and forbidden[y : y + size, x : x + size][inside].any()
        ):
            continue
        region = hed[y : y + size, x : x + size]
        region += alpha[..., np.newaxis] * (window - region)
        mask = np.zeros((h, w), dtype=bool)
        mask[y : y + size, x : x + size] = inside
        return mask
    return None
