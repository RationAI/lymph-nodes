import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import pandas as pd
from aiohttp import ClientSession, ClientTimeout
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


def parse_mmci_filename(filename: str) -> dict[str, Any]:
    pattern = r"^SNB_([A-Z]+)_CASE_(\d+)_SLIDE_(\d+)-(0|1)\.mrxs$"

    match = re.match(pattern, filename)

    if match:
        # Extract the captured groups
        staining, case_id, slice_id, tumor_indicator = match.groups()

        return {
            "case_id": case_id,
            "slice_id": slice_id,
            "staining": staining,
            "tumor": tumor_indicator == "1",
        }
    else:
        raise ValueError(f"Filename does not match expected MMCI pattern: {filename}")


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

    match = re.match(pattern, filename)

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


def save_dataset(slides: pd.DataFrame, logger: MLFlowLogger) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/slides.csv"
        slides.to_csv(path, index=False)
        logger.log_artifacts(path, "slides")

    df.to_csv(output_path, index=False)


@hydra.main(config_path="../configs", config_name="preprocessing/qc", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = pd.DataFrame(
        {
            "slide_path": [str(slide) for slide in config.data_source],
        }
    )

    if config.institute == "mmci":
        parser_func = parse_mmci_filename
    elif config.institute == "fnb":
        parser_func = parse_fnb_filename
    else:
        # Handle cases where config.kind is not recognized
        raise ValueError(
            f"Unknown dataset kind specified: {config.dataset_kind}. Must be 'mmci' or 'fnb'."
        )

    parsed_data = df["slide_path"].apply(parser_func).apply(pd.Series)
    df = pd.concat([df, parsed_data], axis=1)
    df["institute"] = config.institute

    slides_dataset = logger.data.pandas_dataset.from_pandas(
        df,
        name=config.dataset_name,
        context="slides",
    )

    logger.log_input(slides_dataset, context="slides")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
