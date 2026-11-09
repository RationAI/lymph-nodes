from __future__ import annotations

import numpy as np


class HardNegativeBank:
    """Hard negatives mined during training: the negative tiles the model itself scores highest.

    Every training step scores its tiles anyway; lymph_nodes.callbacks.HardNegativeMining
    hands this bank each negative lymph node tile's score (its most confident cell)
    after the step. StratifiedEpochSampler draws its ``healthy_hard`` partition from the
    bank's pool each epoch from ``start_epoch`` on (before, from the healthy tiles): the
    ``pool_size`` highest-scoring tiles of at least ``min_score``, at most a share
    ``slide_cap`` of them from one slide (no single slide's artefact takes over). A
    tile's score is a running mean (``decay``: the new score's weight), refreshed
    whenever it is drawn again, so tiles the model has learnt drop out of the pool while
    the random draws of the other partitions keep finding new ones. Unlike a table mined
    once by another model, it follows this model's own mistakes, on every fold.

    Tiles with a vivid DAB-stained spot (meta ``vivid``, TileSegmentationDataset's
    ``vivid_dab``) are never banked: a "negative" slide may hold an isolated tumour cell
    the pathologist missed, and mining it would teach it as benign.
    """

    def __init__(
        self,
        pool_size: int = 10_000,
        min_score: float = 0.1,
        decay: float = 0.5,
        slide_cap: float = 0.05,
        start_epoch: int = 1,
    ) -> None:
        self.pool_size = pool_size
        self.min_score = min_score
        self.decay = decay
        self.slide_cap = slide_cap
        self.start_epoch = start_epoch
        self._score = np.empty(0, dtype=np.float32)
        self._eligible = np.empty(0, dtype=bool)
        self._slide_of = np.empty(0, dtype=np.int64)
        self._vivid_seen = 0

    def setup(self, eligible: np.ndarray, slide_of: np.ndarray) -> None:
        """Per dataset index: whether it may be banked (a negative lymph node tile), its slide."""
        self._eligible = np.asarray(eligible, dtype=bool)
        self._slide_of = np.asarray(slide_of, dtype=np.int64)
        self._score = np.full(len(self._eligible), np.nan, dtype=np.float32)

    def update(
        self, indices: np.ndarray, scores: np.ndarray, vivid: np.ndarray
    ) -> None:
        """Record the scores of a step's tiles (dataset indices); vivid ones leave the bank."""
        indices = np.asarray(indices, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        vivid = np.asarray(vivid, dtype=bool)
        keep = self._eligible[indices]
        indices, scores, vivid = indices[keep], scores[keep], vivid[keep]
        self._vivid_seen += int(vivid.sum())
        self._score[indices[vivid]] = np.nan
        indices, scores = indices[~vivid], scores[~vivid]
        old = self._score[indices]
        self._score[indices] = np.where(
            np.isnan(old), scores, self.decay * scores + (1 - self.decay) * old
        )

    def pool(self) -> np.ndarray:
        """The dataset indices to draw hard negatives from (highest-scoring first)."""
        candidates = np.flatnonzero(self._score >= self.min_score)
        if len(candidates) == 0:
            return candidates
        candidates = candidates[np.argsort(-self._score[candidates], kind="stable")]
        slides = self._slide_of[candidates]
        cap = max(1, int(self.slide_cap * self.pool_size))
        # Each tile's rank among the candidates of its slide (0 for its highest).
        order = np.argsort(slides, kind="stable")
        sorted_slides = slides[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_slides)) + 1]
        lengths = np.diff(np.r_[starts, len(sorted_slides)])
        rank = np.empty(len(candidates), dtype=np.int64)
        rank[order] = np.arange(len(candidates)) - np.repeat(starts, lengths)
        return candidates[rank < cap][: self.pool_size]

    def stats(self) -> dict[str, float]:
        """For logging: tiles banked, the pool's size, score and slides, vivid tiles seen."""
        pool = self.pool()
        return {
            "banked": float(np.count_nonzero(~np.isnan(self._score))),
            "pool": float(len(pool)),
            "pool_score_mean": float(self._score[pool].mean()) if len(pool) else 0.0,
            "pool_score_min": float(self._score[pool].min()) if len(pool) else 0.0,
            "pool_slides": float(len(np.unique(self._slide_of[pool]))),
            "vivid_seen": float(self._vivid_seen),
        }
