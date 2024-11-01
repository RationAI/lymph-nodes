import mlflow

from preprocessing.annotation_mask import get_metastazis_masks
from preprocessing.data import test_wsis
from preprocessing.tissue_mask import get_tissue_masks


def create_masks() -> None:
    get_tissue_masks(test_wsis())
    get_metastazis_masks(test_wsis())

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB testing data with epytelium - masks") as _:
        mlflow.log_artifacts("data")


if __name__ == "__main__":
    create_masks()
