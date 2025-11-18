# This script was adapted from the repository:
# https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/prostate-cancer/-/tree/a631d443abf1b618be8ff5c128bf17fecd1ea310/

import shutil
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


QC_MASKS = [
    ("Piqe_focus_score_piqe_median", "blur_per_tile"),
    ("Piqe_piqe_median_activity_mask", "blur_per_pixel"),
]


def create_directory_structure(qc_path: Path, file_prefix: str) -> None:
    prefix_dir = qc_path / file_prefix
    prefix_dir.mkdir(parents=True, exist_ok=True)

    print("Copying files with prefix:", file_prefix)
    for file in qc_path.glob(f"{file_prefix}_*.tiff"):
        slide_name = file.name.replace(f"{file_prefix}_", "")
        destination = prefix_dir / slide_name
        shutil.copy(file, destination)

    for file in qc_path.glob(f"{file_prefix}_*.tiff"):
        file.unlink()

    print("Deleted files with prefix:", file_prefix)


def log_qc_masks_directory(prefix_dir: Path, artifact_path: str) -> None:
    """Log the QC masks directory to MLFlow."""
    print(f"Logging directory: {prefix_dir} into artifact path: {artifact_path}")

    # Can't log the directory directly, since the name of the directory would be used as the artifact name
    for file in prefix_dir.glob("*.tiff"):
        mlflow.log_artifact(str(file), artifact_path=artifact_path)


@hydra.main(
    config_path="../../configs", config_name="preprocessing/qc_masks", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_path)

    for prefix, artifact_name in QC_MASKS:
        create_directory_structure(output_path, prefix)
        artifact_name = f"qc_masks/{artifact_name}"
        log_qc_masks_directory(output_path / prefix, artifact_name)


if __name__ == "__main__":
    main()
