import mlflow

from preprocessing.annotation_mask import get_metastazis_masks
from preprocessing.data import (
    positive_train_wsis,
    positive_val_wsis,
    test_wsis,
    test_wsis_colorectal,
    test_wsis_lymph_nodes,
    train_wsis,
    val_wsis,
)
from preprocessing.tissue_mask import get_tissue_masks


def create_masks() -> None:
    get_tissue_masks([*train_wsis(), *val_wsis(), *test_wsis()])

    get_metastazis_masks(
        [*positive_train_wsis(), *positive_val_wsis(), *test_wsis_colorectal()],
        inverted_anotation=True,
    )

    get_metastazis_masks(list(test_wsis_lymph_nodes()), inverted_anotation=False)

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB testing data with epitelium - masks") as _:
        mlflow.log_artifacts("data")


if __name__ == "__main__":
    create_masks()
