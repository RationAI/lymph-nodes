import mlflow

from preprocessing.annotation_mask import get_metastazis_masks
from preprocessing.data import positive_training_wsis, training_wsis
from preprocessing.tissue_mask import get_tissue_masks


def create_masks() -> None:
    get_tissue_masks(training_wsis())
    get_metastazis_masks(positive_training_wsis())

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB training data with epytelium - masks") as _:
        mlflow.log_artifacts("data/tissue_masks")
        mlflow.log_artifacts("data/annotation_masks")


if __name__ == "__main__":
    create_masks()
