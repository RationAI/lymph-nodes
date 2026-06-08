import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from openslide import OpenSlide
from preprocessing.data_exploration.name_parsers import ParsedFilename
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm


def _read_slide_metadata(slide_path: str) -> dict:
    with OpenSlide(slide_path) as slide:
        return {
            "mpp_x": float(slide.properties.get("openslide.mpp-x", float("nan"))),
            "mpp_y": float(slide.properties.get("openslide.mpp-y", float("nan"))),
            "n_levels": slide.level_count,
            "vendor": slide.properties.get("openslide.vendor"),
        }


def _process_slide(
    path: str,
    name_parser: Callable[[str], ParsedFilename],
    tma_control_slides: frozenset[str],
    demaged_slides: frozenset[str],
    confounding_structure_slides: frozenset[str],
) -> dict:
    slide_name = Path(path).stem
    parsed = name_parser(Path(path).name)

    damaged = slide_name in demaged_slides
    meta = (
        {"mpp_x": float("nan"), "mpp_y": float("nan"), "n_levels": None, "vendor": None}
        if damaged
        else _read_slide_metadata(path)
    )

    return {
        "slide_path": path,
        "slide_name": slide_name,
        **parsed.__dict__,
        **meta,
        "has_tma_control": slide_name in tma_control_slides,
        "damaged": damaged,
        "has_confounding_structures": slide_name in confounding_structure_slides,
    }


def build_slides_df(
    slide_paths: list[str],
    name_parser: Callable[[str], ParsedFilename],
    tma_control_slides: frozenset[str],
    demaged_slides: frozenset[str],
    confounding_structure_slides: frozenset[str],
    max_workers: int,
) -> pd.DataFrame:
    process = partial(
        _process_slide,
        name_parser=name_parser,
        tma_control_slides=tma_control_slides,
        demaged_slides=demaged_slides,
        confounding_structure_slides=confounding_structure_slides,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(tqdm(executor.map(process, slide_paths), total=len(slide_paths), desc="Reading slides"))

    return pd.DataFrame(rows)


def build_patients_df(slides_df: pd.DataFrame) -> pd.DataFrame:
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


@with_cli_args(["+preprocessing=patient_dataset"])
@hydra.main(
    config_path="../../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = hydra.utils.instantiate(config.dataset.slides)

    slides_df = build_slides_df(
        list(slides),
        name_parser=config.dataset.name_parser,
        tma_control_slides=frozenset(config.dataset.get("tma_control_slides") or []),
        demaged_slides=frozenset(config.dataset.get("demaged_slides") or []),
        confounding_structure_slides=frozenset(config.dataset.get("confounding_structure_slides") or []),
        max_workers=config.max_workers,
    )
    patients_df = build_patients_df(slides_df)

    readable = slides_df[~slides_df["damaged"]]

    mlflow.log_metrics({
        "n_slides_total": len(slides_df),
        "n_slides_positive": int(slides_df["tumor"].sum()),
        "n_slides_negative": int((~slides_df["tumor"]).sum()),
        "n_cases_total": int(slides_df["case_id"].nunique()),
        "n_cases_positive": int((patients_df["n_positive_slides"] > 0).sum()),
        "n_cases_negative": int((patients_df["n_positive_slides"] == 0).sum()),
        "n_slides_tma_control": int(slides_df["has_tma_control"].sum()),
        "n_slides_damaged": int(slides_df["damaged"].sum()),
        "n_slides_confounding": int(slides_df["has_confounding_structures"].sum()),
        "mpp_x_min": float(readable["mpp_x"].min()),
        "mpp_x_max": float(readable["mpp_x"].max()),
        "mpp_x_mean": float(readable["mpp_x"].mean()),
        "mpp_y_min": float(readable["mpp_y"].min()),
        "mpp_y_max": float(readable["mpp_y"].max()),
        "mpp_y_mean": float(readable["mpp_y"].mean()),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        slides_df.to_csv(f"{tmp_dir}/slides.csv", index=False)
        patients_df.to_csv(f"{tmp_dir}/patients.csv", index=False)
        logger.log_artifacts(tmp_dir)

    mlflow.log_input(
        mlflow.data.from_pandas(slides_df, name=config.dataset.name),
        context="slides",
    )

    mlflow.log_input(
        mlflow.data.from_pandas(patients_df, name=f"{config.dataset.name}_patients"),
        context="patients",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
