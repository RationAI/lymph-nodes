"""Tissue microarrays: their cores, and which of their cells are cytokeratin-positive.

Pure functions on images (numpy, scipy), so they can be checked anywhere; the cluster
job that reads the slides and writes the masks is preprocessing/tma_masks.py.

- ``find_cores``: the cores of a TMA slide, from a low-resolution image (~15 um/px),
  numbered by row and column of the array; ``core_region`` a core's region.
- ``corrected_mask``: the ground truth of a core at ~0.5 um/px: the old cytokeratin
  masks (MLflow run 8ad173ea) kept where DAB staining supports them, with an ignore
  label where the two disagree. A rule built from the staining alone (stained cells of
  the refined labels' rule, closed into epithelium) took in stroma and diffuse
  background staining that the old masks rightly leave out.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from preprocessing.staining import cells_from_density, hed_densities


POSITIVE, IGNORE, NEGATIVE = 255, 128, 0


def white_point(rgb: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """RGB of the empty glass: a high quantile of the low-saturation, bright pixels.

    TNBC scans show tinted glass (beige, grey), which colour deconvolution reads as
    stain; scaling it back to white removes most of it. Unscanned pure white or black
    is left out.
    """
    rgb = np.asarray(rgb, np.float32)
    top, bottom = rgb.max(-1), rgb.min(-1)
    saturation = (top - bottom) / np.maximum(top, 1)
    glass = (saturation < 0.1) & (top > 150) & (rgb.sum(-1) < 3 * 255) & (top > 0)
    if glass.sum() < 100:
        return np.full(3, 255.0, np.float32)
    bright = rgb[glass]
    cutoff = np.quantile(bright.sum(-1), 0.5)
    return np.quantile(bright[bright.sum(-1) >= cutoff], quantile, axis=0)


def balance(rgb: np.ndarray, white: np.ndarray) -> np.ndarray:
    """``rgb`` with each channel scaled so that ``white`` becomes 255 (float32)."""
    gain = 255.0 / np.maximum(np.asarray(white, np.float32), 100.0)
    return np.clip(np.asarray(rgb, np.float32) * gain, 0, 255)


@dataclass
class Core:
    index: int  # label in the core mask
    row: int
    col: int
    y: float  # centre, pixels of the low-resolution image
    x: float
    radius: float  # pixels
    tissue_mm2: float
    pieces: int


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def tissue(rgb: np.ndarray, um: float) -> np.ndarray:
    """Tissue of a white-balanced low-resolution image: optical density, smoothed."""
    od = -np.log((np.asarray(rgb, np.float32) + 1) / 256).sum(-1)
    od = ndimage.gaussian_filter(od, 1)
    stained = od[od > 0.03]
    level = _otsu(stained) / 2 if stained.size > 100 else 0.15
    mask = od > min(0.25, max(0.08, level))
    mask = ndimage.binary_closing(mask, iterations=max(1, round(30 / um)))
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_opening(mask, iterations=1)


def _grid_index(values: np.ndarray, spacing: float) -> np.ndarray:
    """1-D clustering: sorted values split where the gap exceeds half a spacing."""
    order = np.argsort(values)
    index = np.zeros(len(values), int)
    current = 0
    for previous, this in itertools.pairwise(order):
        if values[this] - values[previous] > spacing / 2:
            current += 1
        index[this] = current
    return index


def find_cores(
    rgb: np.ndarray,
    um: float,
    min_core_mm2: float = 0.15,
    max_core_mm2: float = 7.0,
    reach: float = 1.35,
    min_nuclei: float = 0.018,
    min_texture: float = 0.04,
) -> tuple[np.ndarray, list[Core]]:
    """The cores of a TMA slide image ``rgb`` at ``um`` per pixel (white-balanced).

    1. Tissue (``tissue``) and its connected pieces.
    2. The core radius: the median equivalent radius of round, solid pieces of a
       plausible area (``min_core_mm2``-``max_core_mm2``).
    3. Core centres: pieces of at least a quarter of a core's area; a piece of several
       touching cores split at the peaks of its distance transform (at least a radius
       apart); smaller pieces join the nearest centre within ``reach`` radii, or, if
       none, group into a core of their own when together large enough.
    4. Each core: the tissue within ``reach`` radii of its centre that is nearer to it
       than to any other centre. Rejected: candidates mostly outside that disc, and
       ones without patches of nuclei (90th percentile of haematoxylin under
       ``min_nuclei``: smudges, streaks, glue) or without texture (median local
       variation of optical density under ``min_texture``: dark blobs, scan stripes)
       — unless a disc of about a core's size (circularity >= 0.8, 0.6-1.3 cores'
       area) with some stain and colour: pale, unstained cores.
    5. Rows and columns (``_number``).

    Returns the core mask (int32, 0 outside cores, else a core's index) and the cores.
    """
    rgb = np.asarray(rgb, np.float32)
    mask = tissue(rgb, um)
    pieces, n = ndimage.label(mask)
    if not n:
        return np.zeros(mask.shape, np.int32), []
    px_mm2 = (um / 1000) ** 2
    areas = np.bincount(pieces.ravel())[1:] * px_mm2
    boxes = ndimage.find_objects(pieces)

    # 2. the core radius
    radii = []
    for i, box in enumerate(boxes):
        if not min_core_mm2 <= areas[i] <= max_core_mm2:
            continue
        piece = pieces[box] == i + 1
        h, w = piece.shape
        fill = piece.sum() / (np.pi * (max(h, w) / 2) ** 2)
        if 0.6 <= h / w <= 1.6 and fill > 0.6:
            radii.append(np.sqrt(piece.sum() / np.pi))
    radius = float(np.median(radii)) if radii else np.sqrt(0.8 / px_mm2 / np.pi)
    core_area = np.pi * radius**2

    # 3. centres
    distance = ndimage.distance_transform_edt(mask)
    centres: list[tuple[float, float]] = []
    small = []
    for i, box in enumerate(boxes):
        size = areas[i] / px_mm2
        if size < core_area / 4:
            small.append(i)
            continue
        piece = pieces[box] == i + 1
        if size > 1.6 * core_area:
            local = distance[box] * piece
            peaks = (
                local == ndimage.maximum_filter(local, size=int(2 * radius) | 1)
            ) & (local > radius / 2)
            ys, xs = np.nonzero(peaks)
            kept: list[tuple[float, float]] = []
            for y, x in sorted(zip(ys, xs, strict=True), key=lambda p: -local[p]):
                if all(np.hypot(y - ky, x - kx) >= radius for ky, kx in kept):
                    kept.append((y, x))
            centres += [(box[0].start + y, box[1].start + x) for y, x in kept]
        else:
            cy, cx = ndimage.center_of_mass(piece)
            centres.append((box[0].start + cy, box[1].start + cx))
    if small:
        lone = np.isin(pieces, np.array(small) + 1)
        if centres:
            yy, xx = np.array(centres).T
            near = np.zeros(mask.shape, bool)
            for y, x in zip(yy, xx, strict=True):
                r = reach * radius
                y0, y1 = int(max(0, y - r)), int(min(mask.shape[0], y + r + 1))
                x0, x1 = int(max(0, x - r)), int(min(mask.shape[1], x + r + 1))
                gy, gx = np.ogrid[y0:y1, x0:x1]
                near[y0:y1, x0:x1] |= (gy - y) ** 2 + (gx - x) ** 2 <= r**2
            lone &= ~near
        groups, k = ndimage.label(
            ndimage.binary_dilation(lone, iterations=max(1, round(radius / 2)))
        )
        for j in range(1, k + 1):
            members = lone & (groups == j)
            if members.sum() >= core_area / 4:
                cy, cx = ndimage.center_of_mass(members)
                centres.append((cy, cx))
    if not centres:
        return np.zeros(mask.shape, np.int32), []

    # 4. each core's tissue: nearest centre, within reach
    seeds = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(centres, start=1):
        seeds[round(y), round(x)] = i
    _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    nearest = seeds[iy, ix]
    dist = np.hypot(
        np.arange(mask.shape[0])[:, None] - iy, np.arange(mask.shape[1])[None, :] - ix
    )
    labels = np.where(mask & (dist <= reach * radius), nearest, 0)

    haematoxylin = hed_densities(rgb)[..., 0]
    od = -np.log((rgb + 1) / 256).sum(-1)
    texture = np.abs(od - ndimage.uniform_filter(od, 5))
    cores: list[Core] = []
    out = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(centres, start=1):
        own = labels == i
        piece_ids = np.unique(pieces[own])
        whole = np.isin(pieces, piece_ids[piece_ids > 0])
        inside = own.sum() / max(1, (whole & (nearest == i)).sum())
        if own.sum() < core_area / 4 or inside < 0.6:
            continue  # too little tissue, or mostly outside the disc: not a core
        nuclei = np.percentile(haematoxylin[own], 90)
        if nuclei < min_nuclei or np.median(texture[own]) < min_texture:
            filled = ndimage.binary_fill_holes(own)
            perimeter = (filled & ~ndimage.binary_erosion(filled)).sum()
            circularity = 4 * np.pi * filled.sum() / max(1, perimeter) ** 2
            pixels = rgb[own]
            saturation = np.median(
                (pixels.max(-1) - pixels.min(-1)) / np.maximum(pixels.max(-1), 1)
            )
            disc = circularity >= 0.8 and 0.6 <= filled.sum() / core_area <= 1.3
            if not (disc and nuclei >= 0.012 and saturation >= 0.025):
                continue  # no nuclei, or no texture, and not a pale core: not tissue
        out[own] = i
        cores.append(
            Core(
                index=i,
                row=0,
                col=0,
                y=float(y),
                x=float(x),
                radius=radius,
                tissue_mm2=float(own.sum() * px_mm2),
                pieces=int((piece_ids > 0).sum()),
            )
        )
    _number(cores, radius)
    return out, cores


def _number(cores: list[Core], radius: float, max_angle: float = 15.0) -> None:
    """Rows and columns of the array, numbered from the top left.

    Rows: the centres projected across a direction tilted by up to ``max_angle``
    degrees, clustered (gaps over a radius); the tilt is the one with the tightest
    rows. Columns alike, with their own tilt (arrays are often slightly sheared).
    """
    if not cores:
        return
    points = np.array([(c.y, c.x) for c in cores])

    def best(axis: int) -> np.ndarray:
        options = []
        for degrees in np.arange(-max_angle, max_angle + 0.01, 0.5):
            a = np.deg2rad(degrees)
            along = (
                points[:, 0] * np.cos(a) - points[:, 1] * np.sin(a)
                if axis == 0
                else points[:, 1] * np.cos(a) + points[:, 0] * np.sin(a)
            )
            index = _grid_index(along, 2 * radius)
            spread = np.mean(
                [
                    np.ptp(along[index == i])
                    for i in np.unique(index)
                    if (index == i).sum() > 1
                ]
                or [0.0]
            )
            options.append((spread, -index.max(), degrees, index))
        return min(options, key=lambda o: o[:3])[3]

    for core, row, col in zip(cores, best(0), best(1), strict=True):
        core.row, core.col = int(row), int(col)


def core_region(labels: np.ndarray, index: int) -> np.ndarray:
    """A core's region on the low-resolution grid: its tissue closed and holes filled.

    The cores' tissue at ~15 um/px misses pale patches and lumina; the region takes
    them in, and the fine tissue mask at the read level narrows it again.
    """
    own = ndimage.binary_closing(labels == index, iterations=2)
    return ndimage.binary_fill_holes(own) & ((labels == 0) | (labels == index))


def corrected_mask(
    rgb: np.ndarray,
    mpp: float,
    core: np.ndarray,
    reference: np.ndarray,
    support_um: float = 2.0,
    threshold: float = 0.03,
    low_threshold: float = 0.022,
    min_area_um2: float = 10.0,
    hole_area_um2: float = 200.0,
    min_purity: float = 0.45,
    dark: int = 80,
    dark_chroma: int = 20,
    edge_um: float = 15.0,
) -> np.ndarray:
    """A reference cytokeratin mask, kept where the staining supports it.

    ``reference`` (bool, same shape as ``rgb``): an earlier ground truth, here the
    cytokeratin masks of MLflow run 8ad173ea, whose outlines follow the epithelium
    tightly but which label whole CK-negative cores positive where their automatic
    threshold falls into the background, and leave faintly stained cells negative.
    The stained cells are the refined labels' (cells_from_density) on DAB that passes
    two colour checks: a DAB share of the stain densities of at least ``min_purity``
    (grey reads ~0.42, brown 0.9+, brown cytoplasm over a blue nucleus ~0.5), and not
    black (every channel under ``dark``, within ``dark_chroma`` of each other: ink,
    debris; strong DAB is as dark but brown).

    - POSITIVE: the reference within ``support_um`` of a stained cell.
    - IGNORE: where the two disagree — stained cells the reference leaves out (faint
      ones), reference positives without stained cells near (a negative core's
      background); positives within ``edge_um`` of the core's edge; black or
      colourless stained pixels.
    - NEGATIVE: the rest of the core and everything outside it.
    """
    stains = hed_densities(rgb)
    purity = stains[..., 2] / np.maximum(stains.sum(-1), 1e-6)
    rgb = np.asarray(rgb)
    top, bottom = rgb.max(-1), rgb.min(-1)
    blackish = ndimage.binary_dilation(
        (top < dark) & (top - bottom < dark_chroma), iterations=2
    )
    valid = (purity >= min_purity) & ~blackish
    cells = (
        cells_from_density(
            np.where(valid, stains[..., 2], 0),
            mpp,
            threshold=threshold,
            low_threshold=low_threshold,
            min_area_um2=min_area_um2,
            hole_area_um2=hole_area_um2,
            grow_um=0.0,
        )
        & core
    )
    reference = reference & core
    near = ndimage.distance_transform_edt(~cells) <= support_um / mpp
    positive = reference & near
    outline = ndimage.binary_fill_holes(core)  # lumina and gaps are no edge
    edge = core & ~ndimage.binary_erosion(
        outline, iterations=max(1, round(edge_um / mpp))
    )
    out = np.full(core.shape, NEGATIVE, np.uint8)
    invalid = core & ~valid & (stains[..., 2] > threshold)
    out[(cells & ~reference) | (reference & ~near) | invalid] = IGNORE
    out[positive] = POSITIVE
    out[positive & edge] = IGNORE
    return out
