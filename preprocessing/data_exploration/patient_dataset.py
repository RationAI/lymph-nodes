import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm

from preprocessing.data_exploration.name_parsers import ParsedFilename
from preprocessing.data_exploration.patients import build_patients_df


def _read_slide_metadata(slide_path: str) -> dict[str, Any]:
    with OpenSlide(slide_path) as slide:
        return {
            "base_mpp_x": float(slide.properties.get("openslide.mpp-x", float("nan"))),
            "base_mpp_y": float(slide.properties.get("openslide.mpp-y", float("nan"))),
            "n_levels": slide.level_count,
            "vendor": slide.properties.get("openslide.vendor"),
        }


def _process_slide(
    path: str,
    name_parser: Callable[[str], ParsedFilename],
    tma_control_slides: frozenset[str],
    damaged_slides: frozenset[str],
    confounding_structure_slides: frozenset[str],
) -> dict[str, Any]:
    slide_name = Path(path).stem
    parsed = name_parser(Path(path).name)

    damaged = slide_name in damaged_slides
    meta = (
        {"base_mpp_x": float("nan"), "base_mpp_y": float("nan"), "n_levels": None, "vendor": None}
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
    damaged_slides: frozenset[str],
    confounding_structure_slides: frozenset[str],
    max_workers: int,
) -> pd.DataFrame:
    process = partial(
        _process_slide,
        name_parser=name_parser,
        tma_control_slides=tma_control_slides,
        damaged_slides=damaged_slides,
        confounding_structure_slides=confounding_structure_slides,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(tqdm(executor.map(process, slide_paths), total=len(slide_paths), desc="Reading slides"))

    return pd.DataFrame(rows)


@with_cli_args(["+preprocessing=patient_dataset"])
@hydra.main(
    config_path="../../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = hydra.utils.instantiate(config.dataset.slides)
    name_parser = hydra.utils.instantiate(config.dataset.name_parser, _partial_=True)

    slides_df = build_slides_df(
        list(slides),
        name_parser=name_parser,
        tma_control_slides=frozenset(config.dataset.get("tma_control_slides") or []),
        damaged_slides=frozenset(config.dataset.get("damaged_slides") or []),
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
        "base_mpp_x_min": float(readable["base_mpp_x"].min()),
        "base_mpp_x_max": float(readable["base_mpp_x"].max()),
        "base_mpp_x_mean": float(readable["base_mpp_x"].mean()),
        "base_mpp_y_min": float(readable["base_mpp_y"].min()),
        "base_mpp_y_max": float(readable["base_mpp_y"].max()),
        "base_mpp_y_mean": float(readable["base_mpp_y"].mean()),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        slides_df.to_csv(f"{tmp_dir}/slides.csv", index=False)
        patients_df.to_csv(f"{tmp_dir}/patients.csv", index=False)
        logger.log_artifacts(tmp_dir)

    mlflow.log_input(
        mlflow.data.from_pandas(slides_df, name=config.dataset.name),  # type: ignore[attr-defined]
        context="slides",
    )

    mlflow.log_input(
        mlflow.data.from_pandas(  # type: ignore[attr-defined]
            patients_df, name=f"{config.dataset.name}_patients"
        ),
        context="patients",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
