import mlflow

from preprocessing.annotation_mask import get_metastazis_masks
from preprocessing.data import (
    inference_wsis,
    positive_train_wsis,
    positive_val_wsis,
    test_wsis,
    test_wsis_colorectal,
    test_wsis_lymph_nodes,
    train_wsis,
    val_wsis,
)
from preprocessing.tissue_mask import get_tissue_masks


def create_training_masks() -> None:
    get_tissue_masks([*train_wsis(), *val_wsis(), *test_wsis()])

    get_metastazis_masks(
        [*positive_train_wsis(), *positive_val_wsis(), *test_wsis_colorectal()],
        inverted_anotation=True,
    )

    get_metastazis_masks(list(test_wsis_lymph_nodes()), inverted_anotation=False)

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB testing data with epitelium - masks") as _:
        mlflow.log_artifacts("data")


def create_inference_masks() -> None:
    get_tissue_masks(list(inference_wsis()))

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB inference data 2023 - masks") as _:
        mlflow.log_artifacts("data")
