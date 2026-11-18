r"""Is an IHC slide its H&E match's section, destained and restained? A cell-level score per pair.

analysis/he_ihc_pairs.py (2026-10-29, untracked, outputs in analysis/reports/he_ihc_pairs/
of the main worktree) pairs each IHC 2023 slide with the H&E slide(s) holding the same
blocks, from node outlines and the haematoxylin pattern at 62.5 um/px. That cannot tell the
same section restained from the next section of the same block (3-4 um deeper): both show
the same nodes in about the same shape. Nuclei can. A restained section has the very same
nuclei at the same places, carried by one rigid transform over the whole slide. The next
section has partly other nuclei, and was mounted (stretched, folded) on its own.

Per pair (IHC fixed, H&E moving; both through the haematoxylin channel, which both stains
share; everything compared on grids in the IHC frame, um of level 0):

1. Coarse: both slides at ~62 then ~31 um/px; the H&E in four orientations (as scanned,
   turned round, mirrored either way; a restained glass can only be the first two) and
   rotated by -12..12 degrees (2 degree steps, then 0.5 around the best). Masked normalised
   cross-correlation (Padfield 2012) of the haematoxylin over both tissue masks; the best
   gives a rigid transform T0.
2. Patches: up to PATCHES squares of PATCH_UM spread over the IHC node tissue (farthest-
   point sampling, >= EDGE_UM inside it). Each is found in the H&E near T0 by template
   matching at ~1.9 um/px within +-PATCH_MARGIN_UM, with its own rotation (+-4 degrees):
   neighbouring sections get a fair local fit too.
3. Fine: each patch again (FINE_PATCH_UM at ~0.49 um/px) within +-FINE_MARGIN_UM of its
   local fit (rotation +-0.5 degrees), then nuclei (haematoxylin peaks) in both: `share` =
   IHC nuclei with an H&E nucleus within MATCH_UM; `chance` = the same with the IHC nuclei
   shifted by CHANCE_UM in 8 directions (nuclei are dense in a node, so chance is high);
   kappa = (share - chance) / (1 - chance).
4. Global: a robust similarity fit (RANSAC) of all the fine patch correspondences. Its
   residuals (um) say whether one transform carries the whole slide (the same glass) or
   not (sections mounted apart).

A restain should show a high kappa and residuals of ~1-2 um across the slide; the thresholds
are to be set on the data (mirrored pairs cannot be restains: negative controls).

Writes to --out: pairs.csv (one row per pair), patches.csv (one row per patch), and
qc/<ihc>__<he>.jpg (the global overlay, and three fine patches: IHC, the H&E warped onto it,
and an overlay with the nuclei). With --mlflow they go into a run of experiment 4.

    python scripts/registration/restain_score.py --pairs scripts/registration/pairs.csv \
        --out restain [--workers 8] [--only SNB_IHC_CASE_134_SLIDE_2-1 ...] [--mlflow]
"""

from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import fft, ndimage
from scipy.spatial import cKDTree
from skimage.feature import match_template, peak_local_max


IHC_DIR = "/mnt/data/MOU/lymph_nodes/dataset1-ihc-2023"
HE_DIR = "/mnt/data/MOU/lymph_nodes/dataset1-he-2023"
# Ruifrok's H, E, DAB optical density vectors (as analysis/slide_audit.py): column 0 of the
# inverse is the haematoxylin, which both stains have.
HED_FROM_RGB = np.linalg.inv(
    np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
).astype(np.float32)

# 1. Coarse.
COARSE_UM = (62.0, 31.0)
COARSE_ANGLES = (np.arange(-12.0, 12.1, 2.0), np.arange(-2.0, 2.01, 0.5))
ORIENTATIONS = {
    "as is": (1, 1),
    "turned": (-1, -1),
    "mirrored lr": (-1, 1),
    "mirrored ud": (1, -1),
}
TISSUE_OD, TISSUE_OD_MIN = 0.25, 0.08  # as analysis/he_ihc_pairs.py
MIN_OVERLAP = 0.5  # of the smaller tissue area
NODE_MIN_MM2 = 0.5
# 2. Patches.
PATCHES, PATCH_UM, PATCH_MARGIN_UM, MID_UM = 36, 400.0, 250.0, 1.94
PATCH_ANGLES = np.arange(-4.0, 4.01, 1.0)
EDGE_UM = 200.0
# 3. Fine.
FINE_UM, FINE_PATCH_UM, FINE_MARGIN_UM = 0.486, 160.0, 20.0
FINE_ANGLES = np.arange(-0.5, 0.51, 0.25)
NUCLEUS_SMOOTH_UM, NUCLEUS_MIN_DIST_UM, MIN_NUCLEI = 1.0, 2.5, 20
MATCH_UM, CHANCE_UM = 2.0, 10.0
# 4. Global.
INLIER_UM, RANSAC_ITERS = 5.0, 2000


class Source(Protocol):
    extent: tuple[float, float, float, float]  # x, y, width, height (um)

    def read(
        self, x: float, y: float, w: float, h: float, um: float
    ) -> tuple[np.ndarray, float, float, float]: ...


class Slide:
    """A whole-slide image in um of level 0 (OpenSlide)."""

    def __init__(self, path: str) -> None:
        import openslide

        self.slide = openslide.OpenSlide(path)
        p = self.slide.properties
        self.mpp = float(p[openslide.PROPERTY_NAME_MPP_X])
        w, h = self.slide.dimensions
        bounds = [
            int(p.get(f"openslide.bounds-{k}", d))
            for k, d in (("x", 0), ("y", 0), ("width", w), ("height", h))
        ]
        x, y, bw, bh = (b * self.mpp for b in bounds)
        self.extent: tuple[float, float, float, float] = (x, y, bw, bh)
        self.levels = [d * self.mpp for d in self.slide.level_downsamples]

    def read(
        self, x: float, y: float, w: float, h: float, um: float
    ) -> tuple[np.ndarray, float, float, float]:
        """RGB of the box at the coarsest level not coarser than um: image, x0, y0, um/px."""
        level = max((i for i, u in enumerate(self.levels) if u <= um * 1.01), default=0)
        u = self.levels[level]
        x0, y0 = int(np.floor(x / self.mpp)), int(np.floor(y / self.mpp))
        size = (int(np.ceil(w / u)) + 2, int(np.ceil(h / u)) + 2)
        rgba = np.asarray(self.slide.read_region((x0, y0), level, size))
        rgb = rgba[..., :3].copy()
        rgb[rgba[..., 3] == 0] = 255
        return rgb, x0 * self.mpp, y0 * self.mpp, u


@dataclass
class ArraySlide:
    """A slide (or a part) held in memory: a pyramid of (RGB, um/px) from its corner (x0, y0)."""

    pyramid: list[tuple[np.ndarray, float]]
    x0: float = 0.0
    y0: float = 0.0
    extent: tuple[float, float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        rgb, u = self.pyramid[0]
        self.extent = (self.x0, self.y0, rgb.shape[1] * u, rgb.shape[0] * u)

    @classmethod
    def of(cls, source: Source, um: float) -> ArraySlide:
        rgb, x0, y0, u = source.read(*source.extent, um)
        return cls([(rgb, u)], x0, y0)

    def read(
        self, x: float, y: float, w: float, h: float, um: float
    ) -> tuple[np.ndarray, float, float, float]:
        k = max(
            (i for i, (_, u) in enumerate(self.pyramid) if u <= um * 1.01), default=0
        )
        rgb, u = self.pyramid[k]
        c0, r0 = int(np.floor((x - self.x0) / u)), int(np.floor((y - self.y0) / u))
        c1, r1 = c0 + int(np.ceil(w / u)) + 2, r0 + int(np.ceil(h / u)) + 2
        out = np.full((r1 - r0, c1 - c0, 3), 255, np.uint8)
        a, b = max(r0, 0), min(r1, rgb.shape[0])
        c, d = max(c0, 0), min(c1, rgb.shape[1])
        if a < b and c < d:
            out[a - r0 : b - r0, c - c0 : d - c0] = rgb[a:b, c:d]
        return out, self.x0 + c0 * u, self.y0 + r0 * u, u


# --- geometry: 3 x 3 homogeneous matrices mapping (x, y) um -------------------------


def rotation(degrees: float) -> np.ndarray:
    t = np.deg2rad(degrees)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def affine(linear: np.ndarray, shift: Any = (0.0, 0.0)) -> np.ndarray:
    m = np.eye(3)
    m[:2, :2], m[:2, 2] = linear, shift
    return m


def about(centre: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """X -> centre + linear (x - centre)."""
    return affine(linear, centre - linear @ centre)


def apply(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ m[:2, :2].T + m[:2, 2]


def warp(
    source: Source,
    t: np.ndarray,
    x0: float,
    y0: float,
    shape: tuple[int, int],
    um: float,
) -> np.ndarray:
    """RGB on a grid of um pixels from (x0, y0) of the fixed frame, read through t (fixed -> source)."""
    xs = x0 + (np.arange(shape[1]) + 0.5) * um
    ys = y0 + (np.arange(shape[0]) + 0.5) * um
    gx, gy = np.meshgrid(xs, ys)
    u = t[0, 0] * gx + t[0, 1] * gy + t[0, 2]
    v = t[1, 0] * gx + t[1, 1] * gy + t[1, 2]
    pad = 3 * um
    rgb, rx, ry, ru = source.read(
        u.min() - pad, v.min() - pad, np.ptp(u) + 2 * pad, np.ptp(v) + 2 * pad, um
    )
    rows, cols = (v - ry) / ru - 0.5, (u - rx) / ru - 0.5
    scale = um / ru
    out = np.empty((*shape, 3), np.uint8)
    for k in range(3):
        channel = rgb[..., k].astype(np.float32)
        if scale > 1.2:  # anti-alias a finer level
            channel = ndimage.gaussian_filter(channel, 0.4 * scale)
        out[..., k] = np.clip(
            ndimage.map_coordinates(channel, [rows, cols], order=1, cval=255.0), 0, 255
        )
    return out


# --- stains and masks ------------------------------------------------------------------


def optical_density(rgb: np.ndarray) -> np.ndarray:
    return -np.log((rgb.astype(np.float32) + 1.0) / 256.0)


def haematoxylin(rgb: np.ndarray) -> np.ndarray:
    return np.maximum(optical_density(rgb) @ HED_FROM_RGB[:, 0], 0.0)


def otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def tissue(rgb: np.ndarray) -> np.ndarray:
    od = ndimage.gaussian_filter(optical_density(rgb).sum(-1), 1.0)
    some = od[od > 0.02]
    level = (
        TISSUE_OD
        if some.size < 10
        else float(np.clip(0.5 * otsu(some), TISSUE_OD_MIN, TISSUE_OD))
    )
    return ndimage.binary_opening(od > level)


def node_mask(rgb: np.ndarray, um: float) -> np.ndarray:
    """Haematoxylin-dense tissue (the nodes), holes filled, small pieces dropped."""
    t = tissue(rgb)
    h = ndimage.gaussian_filter(haematoxylin(rgb), 1.0)
    if t.sum() < 10:
        return t
    node = (h > otsu(h[t])) & t
    node = ndimage.binary_closing(node, iterations=max(1, round(100 / um)))
    node = ndimage.binary_fill_holes(node)
    labels, n = ndimage.label(node)
    if n:
        areas = ndimage.sum(node, labels, range(1, n + 1)) * (um / 1000) ** 2
        node = np.isin(labels, 1 + np.flatnonzero(areas >= NODE_MIN_MM2))
    return node if node.any() else t


# --- matching ----------------------------------------------------------------------------


def masked_ncc(
    a: np.ndarray, ma: np.ndarray, b: np.ndarray, mb: np.ndarray, min_overlap: float
) -> tuple[float, tuple[int, int]]:
    """Best masked normalised cross-correlation over all shifts (Padfield 2012).

    Returns the score and the shift (du, dv) with a[i, j] ~ b[i + du, j + dv], over the
    overlap of both masks.
    """
    shape = [
        fft.next_fast_len(a.shape[k] + b.shape[k] - 1, real=True) for k in range(2)
    ]
    ma, mb = ma.astype(np.float64), mb.astype(np.float64)
    fa, fb = a * ma, b * mb
    ca = [np.conj(fft.rfft2(x, shape)) for x in (ma, fa, fa * fa)]
    cb = [fft.rfft2(x, shape) for x in (mb, fb, fb * fb)]

    def xcorr(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return fft.irfft2(p * q, shape)

    n = np.round(xcorr(ca[0], cb[0]))
    sa, sb = xcorr(ca[1], cb[0]), xcorr(ca[0], cb[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        num = xcorr(ca[1], cb[1]) - sa * sb / n
        den = (xcorr(ca[2], cb[0]) - sa**2 / n) * (xcorr(ca[0], cb[2]) - sb**2 / n)
        ncc = num / np.sqrt(den)
    ncc[(n < max(min_overlap, 1)) | ~np.isfinite(ncc)] = -np.inf
    k = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    du = k[0] if k[0] < b.shape[0] else k[0] - shape[0]
    dv = k[1] if k[1] < b.shape[1] else k[1] - shape[1]
    return float(ncc[k]), (int(du), int(dv))


@dataclass
class Coarse:
    t: np.ndarray
    orientation: str
    angle: float
    ncc: float


def coarse(ihc: Source, he: Source) -> Coarse:
    """The rigid transform IHC -> H&E (um) that best overlays the haematoxylin of the tissue."""
    best: Coarse | None = None
    for stage, (um, angles) in enumerate(zip(COARSE_UM, COARSE_ANGLES, strict=True)):
        x0, y0, w, h = ihc.extent
        a_rgb = warp(ihc, np.eye(3), x0, y0, (int(h / um), int(w / um)), um)
        a, ma = haematoxylin(a_rgb), tissue(a_rgb)
        corners = np.array(
            [
                [he.extent[0] + dx * he.extent[2], he.extent[1] + dy * he.extent[3]]
                for dx in (0, 1)
                for dy in (0, 1)
            ]
        )
        if stage == 0:
            candidates = [(o, float(t)) for o in ORIENTATIONS for t in angles]
        else:
            assert best is not None
            candidates = [(best.orientation, best.angle + float(t)) for t in angles]
        for orientation, angle in candidates:
            linear = rotation(angle) @ np.diag(ORIENTATIONS[orientation])
            p = corners @ np.linalg.inv(linear).T
            px0, py0 = p.min(0)
            pw, ph = np.ptp(p, axis=0)
            b_rgb = warp(he, affine(linear), px0, py0, (int(ph / um), int(pw / um)), um)
            b, mb = haematoxylin(b_rgb), tissue(b_rgb)
            score, (du, dv) = masked_ncc(
                a, ma, b, mb, MIN_OVERLAP * min(ma.sum(), mb.sum())
            )
            if best is None or score > best.ncc:
                delta = np.array([px0 - x0 + dv * um, py0 - y0 + du * um])
                best = Coarse(affine(linear, linear @ delta), orientation, angle, score)
        assert best is not None
        if stage == 0:  # the next stage compares at another scale: start afresh
            best = Coarse(best.t, best.orientation, best.angle, -np.inf)
    assert best is not None
    return best


@dataclass
class Located:
    t: np.ndarray  # fixed -> moving around the patch
    angle: float  # relative to the transform searched around
    score: float  # normalised cross-correlation of the haematoxylin
    fixed_rgb: np.ndarray
    moving_rgb: np.ndarray  # the moving slide warped onto the fixed patch


def subpixel(m: np.ndarray, k: tuple[int, ...]) -> np.ndarray:
    out = np.array(k, dtype=float)
    for axis in range(2):
        if 0 < k[axis] < m.shape[axis] - 1:
            lo, hi = list(k), list(k)
            lo[axis] -= 1
            hi[axis] += 1
            y0, y1, y2 = m[tuple(lo)], m[k], m[tuple(hi)]
            den = y0 - 2 * y1 + y2
            if den < 0:
                out[axis] += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return out


def locate(
    fixed: Source,
    moving: Source,
    t: np.ndarray,
    centre: np.ndarray,
    size_um: float,
    margin_um: float,
    um: float,
    angles: np.ndarray,
    smooth_um: float,
) -> Located:
    """Find the fixed square around centre in the moving slide near t, rotation included."""
    n = round(size_um / um)
    nw = round((size_um + 2 * margin_um) / um)
    origin = centre - n * um / 2
    window = centre - nw * um / 2
    fixed_rgb = warp(fixed, np.eye(3), origin[0], origin[1], (n, n), um)
    a = ndimage.gaussian_filter(haematoxylin(fixed_rgb), smooth_um / um)
    # Read the moving slide around the window once (any of the rotations fits in it).
    mc = apply(t, centre[None])[0]
    r = nw * um / np.sqrt(2) * np.sqrt(abs(np.linalg.det(t[:2, :2]))) + 5 * um
    rgb, rx, ry, ru = moving.read(mc[0] - r, mc[1] - r, 2 * r, 2 * r, um)
    moving = ArraySlide([(rgb, ru)], rx, ry)
    best: tuple[float, float, np.ndarray] | None = None
    for angle in angles:
        ta = t @ about(centre, rotation(float(angle)))
        b = ndimage.gaussian_filter(
            haematoxylin(warp(moving, ta, window[0], window[1], (nw, nw), um)),
            smooth_um / um,
        )
        m = match_template(b, a)
        k = tuple(int(i) for i in np.unravel_index(int(np.argmax(m)), m.shape))
        if best is None or m[k] > best[0]:
            oi, oj = subpixel(m, k)
            delta = window + np.array([oj, oi]) * um - origin
            best = (float(m[k]), float(angle), ta @ affine(np.eye(2), delta))
    assert best is not None
    score, angle, tl = best
    return Located(
        tl, angle, score, fixed_rgb, warp(moving, tl, origin[0], origin[1], (n, n), um)
    )


def nuclei(rgb: np.ndarray, um: float) -> np.ndarray:
    """Nucleus centres (x, y um within the patch): haematoxylin peaks above Otsu's level."""
    s = ndimage.gaussian_filter(haematoxylin(rgb), NUCLEUS_SMOOTH_UM / um)
    stained = s[s > 0.03]
    if stained.size < 50:
        return np.empty((0, 2))
    peaks = peak_local_max(
        s,
        min_distance=max(1, round(NUCLEUS_MIN_DIST_UM / um)),
        threshold_abs=otsu(stained),
        exclude_border=False,
    )
    return (peaks[:, ::-1] + 0.5) * um


def agreement(fixed: np.ndarray, moving: np.ndarray, size_um: float) -> dict[str, Any]:
    """Share of fixed nuclei with a moving nucleus within MATCH_UM, its chance level and kappa."""
    nan = {"share": np.nan, "chance": np.nan, "kappa": np.nan}
    if len(fixed) < MIN_NUCLEI or len(moving) < MIN_NUCLEI:
        return nan
    tree = cKDTree(moving)
    lo, hi = CHANCE_UM + MATCH_UM, size_um - CHANCE_UM - MATCH_UM

    def share(points: np.ndarray) -> float:
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        if inside.sum() < MIN_NUCLEI:
            return np.nan
        d, _ = tree.query(points[inside])
        return float(np.mean(d <= MATCH_UM))

    turns = np.deg2rad(np.arange(0, 360, 45))
    shifts = CHANCE_UM * np.c_[np.cos(turns), np.sin(turns)]
    actual = share(fixed)
    chance = float(np.nanmean([share(fixed + s) for s in shifts]))
    if not np.isfinite(actual) or not np.isfinite(chance) or chance >= 1:
        return nan
    return {
        "share": actual,
        "chance": chance,
        "kappa": (actual - chance) / (1 - chance),
    }


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation or reflection, uniform scale, shift): Umeyama."""
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    r = u @ vt
    scale = sig.sum() / max((s**2).sum() / len(src), 1e-12)
    return affine(scale * r, md - scale * r @ ms)


def robust_similarity(
    src: np.ndarray, dst: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """RANSAC (pairs of points) then least squares on the inliers: matrix, residuals, inliers."""
    if len(src) < 3:
        return None
    best: np.ndarray | None = None
    best_cost = np.inf
    for _ in range(RANSAC_ITERS):
        i = rng.choice(len(src), 2, replace=False)
        if np.linalg.norm(src[i[0]] - src[i[1]]) < 50:
            continue
        r = np.linalg.norm(apply(similarity(src[i], dst[i]), src) - dst, axis=1)
        inliers = r < INLIER_UM
        cost = -inliers.sum() + np.minimum(r, INLIER_UM).sum() / (
            INLIER_UM * len(src) + 1
        )
        if cost < best_cost:
            best, best_cost = inliers, cost
    if best is None or best.sum() < 2:
        return None
    m = similarity(src[best], dst[best])
    r = np.linalg.norm(apply(m, src) - dst, axis=1)
    return m, r, r < INLIER_UM


def spread(mask: np.ndarray, um: float, x0: float, y0: float, n: int) -> np.ndarray:
    """Up to n points of the mask, each as far from the others as possible, >= PATCH_UM apart."""
    inner = ndimage.binary_erosion(mask, iterations=max(1, round(EDGE_UM / um)))
    if not inner.any():
        inner = mask
    ii, jj = np.nonzero(inner)
    if not len(ii):
        return np.empty((0, 2))
    points = np.c_[x0 + (jj + 0.5) * um, y0 + (ii + 0.5) * um]
    depth = ndimage.distance_transform_edt(inner)[ii, jj]
    chosen = [int(np.argmax(depth))]
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        if d[k] < PATCH_UM:
            break
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(points - points[k], axis=1))
    return points[chosen]


# --- one pair ----------------------------------------------------------------------------


def score(
    ihc: Source, he: Source, seed: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Pair row, patch rows, and images for the QC sheet."""
    rng = np.random.default_rng(seed)
    ihc_mem, he_mem = ArraySlide.of(ihc, COARSE_UM[1]), ArraySlide.of(he, COARSE_UM[1])
    c = coarse(ihc_mem, he_mem)
    row: dict[str, Any] = {
        "orientation": c.orientation,
        "angle": c.angle,
        "coarse_ncc": c.ncc,
    }

    um = ihc_mem.pyramid[0][1]
    node = node_mask(ihc_mem.pyramid[0][0], um)
    centres = spread(node, um, ihc_mem.x0, ihc_mem.y0, PATCHES)
    patches: list[dict[str, Any]] = []
    fine: list[Located] = []
    for i, centre in enumerate(centres):
        mid = locate(
            ihc, he, c.t, centre, PATCH_UM, PATCH_MARGIN_UM, MID_UM, PATCH_ANGLES, 2.0
        )
        f = locate(
            ihc,
            he,
            mid.t,
            centre,
            FINE_PATCH_UM,
            FINE_MARGIN_UM,
            FINE_UM,
            FINE_ANGLES,
            0.5,
        )
        a, b = nuclei(f.fixed_rgb, FINE_UM), nuclei(f.moving_rgb, FINE_UM)
        mapped = apply(f.t, centre[None])[0]
        patches.append(
            {
                "patch": i,
                "x_um": centre[0],
                "y_um": centre[1],
                "he_x_um": mapped[0],
                "he_y_um": mapped[1],
                "mid_ncc": mid.score,
                "mid_angle": mid.angle,
                "fine_ncc": f.score,
                "fine_angle": mid.angle + f.angle,
                "nuclei_ihc": len(a),
                "nuclei_he": len(b),
                **agreement(a, b, FINE_PATCH_UM),
            }
        )
        fine.append(f)

    table = pd.DataFrame(patches)
    row["patches"] = len(table)
    if len(table):
        for k in (
            "mid_ncc",
            "fine_ncc",
            "share",
            "chance",
            "kappa",
            "nuclei_ihc",
            "nuclei_he",
        ):
            row[f"{k}_median"] = float(table[k].median())
        row["kappa_p25"] = float(table["kappa"].quantile(0.25))
        row["kappa_valid"] = int(table["kappa"].notna().sum())
        row["angle_spread"] = float(table["fine_angle"].std())
    fit = (
        robust_similarity(
            table[["x_um", "y_um"]].to_numpy(),
            table[["he_x_um", "he_y_um"]].to_numpy(),
            rng,
        )
        if len(table)
        else None
    )
    global_t = c.t
    if fit is not None:
        global_t, residuals, inliers = fit
        table["residual_um"] = residuals
        table["inlier"] = inliers
        lin = global_t[:2, :2]
        row |= {
            "residual_median_um": float(np.median(residuals)),
            "residual_p90_um": float(np.quantile(residuals, 0.9)),
            "inlier_share": float(inliers.mean()),
            "inlier_residual_median_um": float(np.median(residuals[inliers])),
            "scale": float(np.sqrt(abs(np.linalg.det(lin)))),
            "global_angle": float(np.degrees(np.arctan2(lin[1, 0], lin[0, 0]))),
        }
    images = {"ihc": ihc_mem, "he": he_mem, "t": global_t, "fine": fine}
    return row, table.to_dict("records"), images


def qc_sheet(
    images: dict[str, Any], patches: list[dict[str, Any]], title: str, file: Path
) -> None:
    """Global overlay (IHC magenta, H&E green) and three fine patches (best, median, worst kappa)."""
    tile = round(FINE_PATCH_UM / FINE_UM)
    ihc, he = images["ihc"], images["he"]
    x0, y0, w, h = ihc.extent
    um = COARSE_UM[1]
    shape = (int(h / um), int(w / um))
    a = haematoxylin(warp(ihc, np.eye(3), x0, y0, shape, um))
    b = haematoxylin(warp(he, images["t"], x0, y0, shape, um))

    def norm(x: np.ndarray) -> np.ndarray:
        return np.clip(
            255 * x / max(float(np.quantile(x, 0.995)), 1e-6), 0, 255
        ).astype(np.uint8)

    overview = Image.fromarray(np.dstack([norm(a), norm(b), norm(a)]))
    overview.thumbnail((3 * tile, 2 * tile))
    rows = [overview]
    order = sorted(
        (
            p
            for p in range(len(patches))
            if np.isfinite(patches[p].get("kappa", np.nan))
        ),
        key=lambda p: -patches[p]["kappa"],
    )
    picks = [order[0], order[len(order) // 2], order[-1]] if order else []
    for p in picks:
        f = images["fine"][p]
        ha, hb = haematoxylin(f.fixed_rgb), haematoxylin(f.moving_rgb)
        overlay = Image.fromarray(np.dstack([norm(ha), norm(hb), norm(ha)]))
        draw = ImageDraw.Draw(overlay)
        for x, y in nuclei(f.fixed_rgb, FINE_UM) / FINE_UM:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 64, 64))
        for x, y in nuclei(f.moving_rgb, FINE_UM) / FINE_UM:
            draw.line((x - 2, y, x + 2, y), fill=(0, 255, 255))
            draw.line((x, y - 2, x, y + 2), fill=(0, 255, 255))
        strip = Image.new("RGB", (3 * tile, tile + 14), "white")
        for k, im in enumerate(
            (Image.fromarray(f.fixed_rgb), Image.fromarray(f.moving_rgb), overlay)
        ):
            strip.paste(im, (k * tile, 14))
        q = patches[p]
        ImageDraw.Draw(strip).text(
            (2, 1),
            f"patch {q['patch']}: kappa {q['kappa']:.2f} share {q['share']:.2f} "
            f"chance {q['chance']:.2f} ncc {q['fine_ncc']:.2f}",
            fill="black",
        )
        rows.append(strip)
    sheet = Image.new("RGB", (3 * tile, 16 + sum(r.height for r in rows)), "white")
    ImageDraw.Draw(sheet).text((2, 2), title, fill="black")
    y = 16
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    file.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(file, quality=85)


def run_pair(
    ihc_name: str, he_name: str, out: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.time()
    base: dict[str, Any] = {"ihc": ihc_name, "he": he_name}
    try:
        ihc = Slide(f"{IHC_DIR}/{ihc_name}.mrxs")
        he = Slide(f"{HE_DIR}/{he_name}.mrxs")
        row, patches, images = score(ihc, he)
        qc_sheet(
            images,
            patches,
            f"{ihc_name} -> {he_name}: {row['orientation']} {row['angle']:+.1f} deg, "
            f"kappa {row.get('kappa_median', np.nan):.2f}, "
            f"residual {row.get('residual_median_um', np.nan):.1f} um",
            out / "qc" / f"{ihc_name}__{he_name}.jpg",
        )
        row = base | row | {"mpp_ihc": ihc.mpp, "mpp_he": he.mpp}
        patches = [base | p for p in patches]
    except Exception:  # one unreadable slide must not stop the others
        row, patches = base | {"error": traceback.format_exc(limit=4)}, []
    row["seconds"] = round(time.time() - start, 1)
    return row, patches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pairs", type=Path, required=True, help="CSV with ihc, he columns"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--only", nargs="*", help="IHC slide names to score (default all)"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="log --out into experiment 4"
    )
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    if args.only:
        pairs = pairs[pairs["ihc"].isin(args.only)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, patches = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {
            pool.submit(run_pair, r.ihc, r.he, args.out): (r.ihc, r.he)
            for r in pairs.itertuples()
        }
        for k, job in enumerate(as_completed(jobs), 1):
            row, p = job.result()
            rows.append(row)
            patches.extend(p)
            print(
                f"[{k}/{len(jobs)}] {row['ihc']} -> {row['he']}: "
                + (
                    "ERROR"
                    if "error" in row
                    else f"kappa {row.get('kappa_median', np.nan):.2f} "
                    f"residual {row.get('residual_median_um', np.nan):.1f} um"
                )
                + f" ({row['seconds']} s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(args.out / "pairs.csv", index=False)
    pd.DataFrame(patches).to_csv(args.out / "patches.csv", index=False)

    if args.mlflow:
        import mlflow

        constants = {
            k: v
            for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, tuple))
        }
        with mlflow.start_run(experiment_id="4", run_name="H&E-IHC restain score"):
            mlflow.set_tag("analysis", "he_registration/restain_score")
            mlflow.log_params({k: str(v)[:250] for k, v in constants.items()})
            mlflow.log_artifacts(str(args.out), "restain_score")


if __name__ == "__main__":
    main()
