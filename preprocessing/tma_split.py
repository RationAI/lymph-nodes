"""TMA cores split between the folds and validation, instead of whole slides.

A TMA slide holds dozens of cores, most from different patients, so splitting whole
slides puts thousands of tiles of one tumour type into a fold at once (the 12 MMCI TMA
slides gave folds of one or two slides each). This splits the cores
of preprocessing/tma_masks.py (its ``cores.csv``) instead:

- cores with less tissue than ``min_tissue_mm2`` are left out, and cores an ignore mask
  covers more than ``max_ignored`` of (the ``ignored_<name>`` columns): the tiling drops
  or filters out their tiles, so they would only unbalance the splits;
- cores that must stay together form a group: by default each core is its own group;
  ``slide_groups`` puts all cores of the listed slides into one group (two sections of
  one block, e.g. FIN-HR2-19-ERA and FIN-HR2-19-PGRA, until the lab's TMA map tells
  which cores match); ``core_groups_uri``, a table of ``core_id`` and ``group`` (the
  TMA map: replicate cores of one patient), overrides both;
- groups are stratified by TMA kind (the slide name's leading letters: KOS, FIN, TNBC)
  and by whether most of their cores are positive (``positive_share`` at least
  ``positive_min``), and within each stratum, shuffled (``seed``), each assigned to the
  split furthest below its share of the stratum's cores (largest groups first).

Splits: ``splits`` maps a name to its ``fraction``; one with ``n_folds`` is divided into
that many folds, ``<name>_fold_<k>``, as preprocessing/data_exploration/dataset_split.py
names them. Writes per split ``slides.csv`` — the slides table's rows of the slides with
cores in it, plus ``cores`` (their indices in the core mask, ``;``-separated, for
preprocessing.tiling_blocks.CoreSelection) — and ``cores.csv`` with every core's split.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing.data_source import read_slides_csv


def buckets(splits: dict) -> dict[str, float]:
    """Split name -> share of the cores, folds expanded."""
    out = {}
    for name, cfg in splits.items():
        n_folds = int(cfg.get("n_folds") or 0)
        if n_folds:
            for k in range(n_folds):
                out[f"{name}_fold_{k}"] = float(cfg["fraction"]) / n_folds
        else:
            out[str(name)] = float(cfg["fraction"])
    total = sum(out.values())
    return {name: share / total for name, share in out.items()}


def assign_cores(
    cores: pd.DataFrame,
    shares: dict[str, float],
    seed: int,
    positive_min: float = 0.05,
    slide_groups: list[list[str]] | None = None,
    core_groups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``cores`` with ``group``, ``kind``, ``positive``, ``stratum`` and ``split``."""
    cores = cores.copy()
    cores["group"] = cores["core_id"]
    for slides in slide_groups or []:
        together = cores["slide"].isin(slides)
        cores.loc[together, "group"] = "slides:" + "+".join(sorted(slides))
    if core_groups is not None and not core_groups.empty:
        mapping = dict(zip(core_groups["core_id"], core_groups["group"], strict=True))
        mapped = cores["core_id"].map(mapping)
        cores["group"] = mapped.fillna(cores["group"])
    cores["kind"] = cores["slide"].str.extract(r"^([A-Za-z]+)")[0]
    cores["positive"] = cores["positive_share"] >= positive_min
    # A group's stratum: its kind and whether most of its cores are positive.
    by_group = cores.groupby("group").agg(
        kind=("kind", lambda k: k.mode().iloc[0]),
        positive=("positive", "mean"),
        n=("core_id", "size"),
    )
    by_group["stratum"] = by_group["kind"] + np.where(
        by_group["positive"] >= 0.5, ":positive", ":negative"
    )
    rng = np.random.default_rng(seed)
    names = list(shares)
    assignment: dict[str, str] = {}
    for _, groups in by_group.groupby("stratum"):
        order = groups.sample(frac=1, random_state=rng.integers(2**31))
        order = order.sort_values("n", ascending=False, kind="stable")
        total = order["n"].sum()
        filled = dict.fromkeys(names, 0)
        for group, row in order.iterrows():
            # the split furthest below its share of this stratum (ties: config order)
            deficit = {n: shares[n] * total - filled[n] for n in names}
            target = max(names, key=lambda n: deficit[n])
            assignment[group] = target
            filled[target] += row["n"]
    cores["stratum"] = cores["group"].map(by_group["stratum"])
    cores["split"] = cores["group"].map(assignment)
    return cores


def split_slides(slides: pd.DataFrame, cores: pd.DataFrame, split: str) -> pd.DataFrame:
    """The slides table's rows of the slides with cores in ``split``, with ``cores``."""
    part = cores[cores["split"] == split]
    listed = part.groupby("slide")["index"].apply(
        lambda indices: ";".join(str(int(i)) for i in sorted(indices))
    )
    counts = part.groupby("slide").size()
    out = slides[slides["slide_name"].isin(listed.index)].copy()
    out["cores"] = out["slide_name"].map(listed)
    out["n_cores"] = out["slide_name"].map(counts)
    return out.reset_index(drop=True)


@with_cli_args(["+preprocessing=tma_split"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = read_slides_csv(download_artifacts(config.slides_uri))
    cores = pd.read_csv(download_artifacts(config.cores_uri))
    ignored = cores.filter(like="ignored_").max(axis=1).fillna(0.0)
    kept = cores[
        (cores["tissue_mm2"] >= config.min_tissue_mm2) & (ignored <= config.max_ignored)
    ]
    print(
        f"{len(kept)} of {len(cores)} cores with >= {config.min_tissue_mm2} mm2 tissue"
        f" and at most {config.max_ignored} under an ignore mask"
    )
    core_groups = (
        pd.read_csv(download_artifacts(config.core_groups_uri))
        if config.get("core_groups_uri")
        else None
    )
    shares = buckets(dict(config.splits))
    assigned = assign_cores(
        kept,
        shares,
        seed=config.seed,
        positive_min=config.positive_min,
        slide_groups=[list(g) for g in config.get("slide_groups") or []],
        core_groups=core_groups,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir)
        assigned.to_csv(out / "cores.csv", index=False)
        for name in shares:
            (out / name).mkdir()
            part = split_slides(slides, assigned, name)
            part.to_csv(out / name / "slides.csv", index=False)
            in_split = assigned[assigned["split"] == name]
            mlflow.log_metrics(
                {
                    f"{name}_n_cores": len(in_split),
                    f"{name}_n_slides": len(part),
                    f"{name}_n_positive_cores": int(in_split["positive"].sum()),
                    **{
                        f"{name}_n_{kind}_cores": int(n)
                        for kind, n in in_split["kind"].value_counts().items()
                    },
                }
            )
        logger.log_artifacts(tmp_dir)
    print(pd.crosstab(assigned["stratum"], assigned["split"]).to_string())


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
