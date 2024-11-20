import itertools
import re
from collections.abc import Iterable
from pathlib import Path


PATH_PREFIX = Path("/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry")
VAL_CASES_IDS = [3, 34, 52, 80, 114]
TEST_CASES_IDS = [20, 59]


def filter_he_slides(slides: Iterable[Path]) -> Iterable[Path]:
    pattern = r"(?:^|\b|[_-])HE(?:[_-]|\b|$)"
    return filter(lambda slide: not re.search(pattern, slide.stem), slides)


def get_relative_dir_path(path: Path) -> Path:
    return path.relative_to(PATH_PREFIX).parent


def positive_train_wsis() -> Iterable[Path]:
    test_val_slides = list(itertools.chain(val_wsis(), test_wsis()))
    return filter(
        lambda slide: slide not in test_val_slides,
        filter_he_slides(
            itertools.chain(
                Path(PATH_PREFIX, "Cytokeratin_mask_colorectal_TMAs").glob("*.mrxs"),
                Path(PATH_PREFIX, "Cytokeratin_mask_new_breast_TNBC-TMAS/ckae").rglob(
                    "*.mrxs"
                ),
            )
        ),
    )


def negative_train_wsis() -> Iterable[Path]:
    test_val_slides = list(itertools.chain(val_wsis(), test_wsis()))
    return filter(
        lambda slide: slide not in test_val_slides,
        filter_he_slides(Path(PATH_PREFIX, "dataset1-2023").glob("*-0.tiff")),
    )


def negative_val_wsis() -> Iterable[Path]:
    return filter_he_slides(
        itertools.chain(
            *[
                Path(PATH_PREFIX, "dataset1-2023").glob(
                    f"*_{case_id}_SLIDE_[0-9]*-0.tiff"
                )
                for case_id in VAL_CASES_IDS
            ]
        )
    )


def positive_val_wsis() -> Iterable[Path]:
    return [
        Path(
            PATH_PREFIX, "Cytokeratin_mask_new_breast_TNBC-TMAS/ckae/TNBC-BF-4-PNG.mrxs"
        )
    ]


def negative_test_wsis() -> Iterable[Path]:
    return filter_he_slides(
        itertools.chain(
            *[
                Path(PATH_PREFIX, "dataset1-2023").glob(
                    f"*_{case_id}_SLIDE_[0-9]*-0.tiff"
                )
                for case_id in TEST_CASES_IDS
            ]
        )
    )


def test_wsis_lymph_nodes() -> Iterable[Path]:
    return filter_he_slides(Path(PATH_PREFIX, "Annotated_IHC_for_test").glob("*.mrxs"))


def test_wsis_colorectal() -> Iterable[Path]:
    return filter_he_slides(
        Path(PATH_PREFIX, "Cytokeratin_mask_final_scans").glob("*.mrxs")
    )


def train_wsis() -> Iterable[Path]:
    return itertools.chain(positive_train_wsis(), negative_train_wsis())


def val_wsis() -> Iterable[Path]:
    return itertools.chain(positive_val_wsis(), negative_val_wsis())


def test_wsis() -> Iterable[Path]:
    return itertools.chain(
        negative_test_wsis(), test_wsis_lymph_nodes(), test_wsis_colorectal()
    )
