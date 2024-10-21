from pathlib import Path


PATH_PREFIX = Path("/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry")


def positive_training_wsis() -> list[Path]:
    return [
        *Path(PATH_PREFIX, "Cytokeratin_mask_colorectal_TMAs").rglob("*.mrxs"),
        *Path(PATH_PREFIX, "Cytokeratin_mask_new_breast_TNBC-TMAS").rglob("*.mrxs"),
    ]


def negative_training_wsis() -> list[Path]:
    return [
        *Path(PATH_PREFIX, "dataset1-2023").rglob("*-0.tiff"),
    ]


def training_wsis() -> list[Path]:
    return positive_training_wsis() + negative_training_wsis()
