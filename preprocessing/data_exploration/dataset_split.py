import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import hydra
import mlflow
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from ratiopath.model_selection import train_test_split
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedGroupKFold


# ── types ─────────────────────────────────────────────────────────────────────

SplitResult = tuple[pd.DataFrame, pd.DataFrame]
FoldedSplitResult = list[dict[str, SplitResult]]


# ── helpers ───────────────────────────────────────────────────────────────────

def _add_derived_case_columns(patients_df: pd.DataFrame) -> pd.DataFrame:
    df = patients_df.copy()
    for count_col, bool_col in [
        ("n_positive_slides", "has_positive_slides"),
        ("n_tma_control_slides", "has_tma_control_slides"),
        ("n_confounding_slides", "has_confounding_slides"),
    ]:
        if count_col in df.columns:
            df[bool_col] = df[count_col] > 0
    return df


def _rebuild_patients_df(slides_df: pd.DataFrame) -> pd.DataFrame:
    if slides_df.empty:
        return pd.DataFrame(
            columns=["case_id", "n_slides", "n_positive_slides", "n_negative_slides",
                     "n_tma_control_slides", "n_confounding_slides"]
        )
    return (
        slides_df.groupby("case_id")
        .agg(
            n_slides=("slide_path", "count"),
            n_positive_slides=("tumor", "sum"),
            n_negative_slides=("tumor", lambda x: (~x).sum()),
            n_tma_control_slides=("has_tma_control", "sum"),
            n_confounding_slides=("has_confounding_structures", "sum"),
        )
        .reset_index()
    )


def _apply_slide_filters(slides: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    for col in exclude_cols:
        if col in slides.columns:
            slides = slides[~slides[col].astype(bool)]
    return slides


# ── core split logic ──────────────────────────────────────────────────────────

def _recursive_split(
    slides: pd.DataFrame,
    splits: list[str],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Recursively binary-split slides across N named splits.

    Uses ratiopath.model_selection.train_test_split with case_id grouping
    and tumor-label stratification. Fractions are re-normalised at each level.
    """
    if len(splits) == 1:
        return {splits[0]: slides}

    p1, rest = splits[0], splits[1:]
    total = sum(fractions[p] for p in splits)
    train_size = fractions[p1] / total

    case_ids = slides["case_id"].values
    if len(np.unique(case_ids)) < 2:
        largest = max(splits, key=lambda p: fractions[p])
        return {p: (slides if p == largest else slides.iloc[:0].copy()) for p in splits}

    y = slides["tumor"].values
    stratify = y if len(np.unique(y)) > 1 else None

    slides_p1, slides_rest = train_test_split(
        slides.reset_index(drop=True),
        train_size=train_size,
        test_size=round(1.0 - train_size, 10),
        random_state=seed,
        stratify=stratify,
        groups=case_ids,
    )
    return {p1: slides_p1, **_recursive_split(slides_rest, rest, fractions, seed)}


def _apply_kfold(
    slides: pd.DataFrame,
    n_folds: int,
    train_exclude_slides: list[str],
    seed: int,
) -> FoldedSplitResult:
    """Apply K-fold CV to slides, producing K (train, val) pairs.

    Train sub-split: slide-level filters applied (same as a regular split).
    Val sub-split: unfiltered — validate against all available slides.
    """
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = slides["tumor"].values
    groups = slides["case_id"].values

    folds: FoldedSplitResult = []
    for train_idx, val_idx in skf.split(slides, y=y, groups=groups):
        train_slides = _apply_slide_filters(
            slides.iloc[train_idx].copy(), train_exclude_slides
        ).reset_index(drop=True)
        val_slides = slides.iloc[val_idx].reset_index(drop=True)

        folds.append({
            "train": (train_slides, _rebuild_patients_df(train_slides)),
            "val": (val_slides, _rebuild_patients_df(val_slides)),
        })
    return folds


def create_splits(
    slides_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    split_configs: DictConfig,
    seed: int,
) -> dict[str, SplitResult | FoldedSplitResult]:
    """Distribute slides across named splits, respecting case grouping and eligibility.

    Splits with n_folds produce a FoldedSplitResult (list of K {train, val} dicts).
    Splits without n_folds produce a plain SplitResult (slides_df, patients_df).
    """
    slides_df = slides_df[~slides_df["damaged"]].copy()
    active_cases = set(slides_df["case_id"].unique())
    patients_df = _add_derived_case_columns(
        patients_df[patients_df["case_id"].isin(active_cases)].copy()
    )
    patients_index = patients_df.set_index("case_id")

    pinned: dict[str, set[str]] = {}
    all_pinned: set[str] = set()
    for name, cfg in split_configs.items():
        pin_list = set(cfg.get("pinned_cases") or [])
        pinned[name] = pin_list
        all_pinned.update(pin_list)

    pool_cases = [c for c in active_cases if c not in all_pinned]
    split_order = list(split_configs.keys())
    fractions = {name: float(cfg.fraction) for name, cfg in split_configs.items()}

    eligibility: dict[str, frozenset[str]] = {
        case_id: frozenset(
            name
            for name, cfg in split_configs.items()
            if all(
                not bool(patients_index.loc[case_id].get(col, False))
                for col in (cfg.get("exclude_cases_where") or [])
            )
        )
        for case_id in pool_cases
    }

    sig_groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for case_id, sig in eligibility.items():
        sig_groups[sig].append(case_id)

    slides_per_split: dict[str, list[pd.DataFrame]] = {name: [] for name in split_configs}

    for sig, case_ids in sig_groups.items():
        if not sig:
            continue
        group_slides = slides_df[slides_df["case_id"].isin(case_ids)].reset_index(drop=True)
        ordered = [p for p in split_order if p in sig]
        for name, part in _recursive_split(group_slides, ordered, fractions, seed).items():
            slides_per_split[name].append(part)

    for name, pinned_set in pinned.items():
        if pinned_set:
            slides_per_split[name].append(slides_df[slides_df["case_id"].isin(pinned_set)])

    result: dict[str, SplitResult | FoldedSplitResult] = {}
    for name, cfg in split_configs.items():
        frames = slides_per_split[name]
        part_slides = pd.concat(frames, ignore_index=True) if frames else slides_df.iloc[:0].copy()
        part_slides = part_slides.reset_index(drop=True)

        n_folds = cfg.get("n_folds")
        if n_folds:
            result[name] = _apply_kfold(
                part_slides,
                n_folds=int(n_folds),
                train_exclude_slides=list(cfg.get("exclude_slides_where") or []),
                seed=seed,
            )
        else:
            part_slides = _apply_slide_filters(part_slides, list(cfg.get("exclude_slides_where") or []))
            result[name] = (part_slides.reset_index(drop=True), _rebuild_patients_df(part_slides))

    return result



def _log_split_metrics(name: str, slides: pd.DataFrame, prefix: str = "") -> None:
    mlflow.log_metrics({
        f"{prefix}{name}_n_slides": len(slides),
        f"{prefix}{name}_n_cases": int(slides["case_id"].nunique()),
        f"{prefix}{name}_n_positive_slides": int(slides["tumor"].sum()),
        f"{prefix}{name}_n_negative_slides": int((~slides["tumor"]).sum()),
    })


@with_cli_args(["+preprocessing=dataset_split"])
@hydra.main(
    config_path="../../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    artifact_dir = Path(download_artifacts(config.patient_dataset_uri))
    slides_df = pd.read_csv(artifact_dir / "slides.csv")
    patients_df = pd.read_csv(artifact_dir / "patients.csv")

    splits = create_splits(slides_df, patients_df, config.splits, config.seed)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, split_result in splits.items():
            if isinstance(split_result, list):
                for k, fold in enumerate(split_result):
                    for fold_split_name, (part_slides, part_patients) in fold.items():
                        out_dir = Path(tmp_dir) / f"fold_{k}" / fold_split_name
                        out_dir.mkdir(parents=True)
                        part_slides.to_csv(out_dir / "slides.csv", index=False)
                        part_patients.to_csv(out_dir / "patients.csv", index=False)
                        _log_split_metrics(fold_split_name, part_slides, prefix=f"fold_{k}_")
            else:
                part_slides, part_patients = split_result
                out_dir = Path(tmp_dir) / name
                out_dir.mkdir()
                part_slides.to_csv(out_dir / "slides.csv", index=False)
                part_patients.to_csv(out_dir / "patients.csv", index=False)
                _log_split_metrics(name, part_slides)

        logger.log_artifacts(tmp_dir)

    for name, split_result in splits.items():
        if not isinstance(split_result, list):
            part_slides, _ = split_result
            mlflow.log_input(
                mlflow.data.from_pandas(part_slides, name=f"{config.dataset.name}_{name}"),
                context=name,
            )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
