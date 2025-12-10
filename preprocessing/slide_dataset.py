import re
import tempfile
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


def parse_mmci_filename(filename: str) -> dict[str, Any]:
    pattern = r"^SNB_([A-Z]+)_CASE_(\d+)_SLIDE_([A-Z0-9-]+)-(0|1)\.mrxs$"

    match = re.match(pattern, Path(filename).name)

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


@hydra.main(
    config_path="../configs",
    config_name="preprocessing/slide_dataset",
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


######################
##### RUN PARAMS #####
######################

"""
from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-slide-dataset",
    username="your name",
    cpu=2,
    memory="2Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.slide_dataset +experiment=<experiment_name>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
"""
