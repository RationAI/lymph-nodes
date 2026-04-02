import re
import tempfile
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def parse_mmci_filename(filename: str) -> dict[str, Any]:
    # Legacy MMCI pattern breakdown:
    # ^SNB_: Starts with 'SNB_'
    # ([A-Z]+): Group 1 (Staining - one or more capital letters)
    # _CASE_: Literal string '_CASE_'
    # (\d+): Group 2 (Case ID - one or more digits)
    # _SLIDE_: Literal string '_SLIDE_'
    # ([A-Z0-9-]+): Group 3 (Slice ID - one or more capital letters, digits, or hyphens)
    # -(0|1): Group 4 (Tumor Indicator - '0' or '1')
    # \.mrxs$: Matches the literal '.mrxs' extension at the end
    base_name = Path(filename).name
    legacy_pattern = r"^SNB_([A-Z]+)_CASE_(\d+)_SLIDE_([A-Z0-9-]+)-(0|1)\.mrxs$"
    legacy_match = re.match(legacy_pattern, base_name)
    if legacy_match:
        staining, case_id, slice_id, tumor_indicator = legacy_match.groups()
        return {
            "case_id": case_id,
            "slice_id": slice_id,
            "staining": staining,
            "tumor": tumor_indicator == "1",
        }

    # Annotated MMCI pattern breakdown:
    # ^SNB_: Starts with 'SNB_'
    # ([A-Z]+): Group 1 (Staining - one or more capital letters)
    # (?:_TEST)?: Optional '_TEST' suffix in the stain prefix
    # _CASE-(\d{4})_: Group 2 (Year, e.g. 2024)
    # (\d+): Group 3 (Case ID)
    # -([A-Z0-9-]+?): Group 4 (Slice ID)
    # (?:-(0|1))?: Optional Group 5 (Tumor Indicator, defaults to positive when missing)
    # \.mrxs$: Matches the literal '.mrxs' extension at the end
    annotated_pattern = (
        r"^SNB_([A-Z]+)(?:_TEST)?_CASE-(\d{4})_(\d+)-([A-Z0-9-]+?)(?:-(0|1))?\.mrxs$"
    )
    annotated_match = re.match(annotated_pattern, base_name)
    if annotated_match:
        staining, year, case_id, slice_id, tumor_indicator = annotated_match.groups()
        return {
            "case_id": f"{year}-{case_id}",
            "slice_id": slice_id,
            "staining": staining,
            "tumor": True,
        }

    if not base_name.startswith("SNB_"):
        return {
            "case_id": "TMA_CASE",
            "slice_id": Path(filename).stem,
            "staining": "DAB",
            "tumor": True,
        }

    raise ValueError(f"Filename does not match expected MMCI pattern: {filename}")


def parse_tma_filename(filename: str) -> dict[str, Any]:
    base_name = Path(filename).stem

    return {
        "case_id": "TMA_CASE",
        "slice_id": base_name,
        "staining": "DAB",
        "tumor": True,
    }


def parse_fnb_filename(filename: str) -> dict[str, Any]:
    # Regex pattern breakdown:
    # ^FNB-?: Starts with 'FNB' optionally followed by a hyphen.
    # P?: Optional 'P' prefix.
    # (\d+): Group 1 (Patient ID Number - one or more digits).
    # -(\d+): Group 2 (Year)
    # -(\d+): Group 3 (Slide ID 1)
    # -(\d+): Group 4 (Slide ID 2)
    # -([A-Z]+): Group 5 (Staining - one or more capital letters)
    # -(0|1): Group 6 (Tumor Indicator - '0' or '1')
    # \.czi$: Matches the literal '.czi' extension at the end
    pattern = r"^FNB-?P?(\d+)-(\d+)-(\d+)-(\d+)-([A-Z]+)-(0|1)\.czi$"

    match = re.match(pattern, Path(filename).name)

    if match:
        # Extract the captured groups. Group 1 (case_id) is now only digits.
        case_id, year, slice_1, slice_2, staining, tumor_indicator = match.groups()

        return {
            "case_id": case_id + "-" + year,
            "slice_id": slice_1 + "-" + slice_2,
            "staining": staining,
            "tumor": tumor_indicator == "1",
        }
    else:
        raise ValueError(f"Filename does not match expected FNB pattern: {filename}")


@with_cli_args(["+preprocessing=slide_dataset"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = hydra.utils.instantiate(config.dataset.slides).to_pandas()

    if config.dataset.institute == "mmci":
        parser_func = parse_mmci_filename
    elif config.dataset.institute == "fnb":
        parser_func = parse_fnb_filename
    else:
        raise ValueError(
            f"Unknown dataset kind specified: {config.dataset.institute}. Must be 'mmci' or 'fnb'."
        )

    parsed_data = df["slide_path"].apply(parser_func).apply(pd.Series)
    df = pd.concat([df, parsed_data], axis=1)
    df["institute"] = config.dataset.institute

    # Save to a temporary CSV file and log as artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        df.to_csv(f"{tmp_dir}/slides.csv", index=False)
        logger.log_artifacts(tmp_dir)

    slides_dataset = mlflow.data.from_pandas(  # type: ignore [attr-defined]
        df,
        name=config.dataset.name,
    )

    mlflow.log_input(slides_dataset, context="slides")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
