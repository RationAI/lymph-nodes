import itertools
import re
from collections.abc import Iterable
from pathlib import Path


PATH_PREFIX = Path("/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry")


def filter_he_slides(slides: Iterable[Path]) -> Iterable[Path]:
    pattern = r"(?:^|\b|[_-])HE(?:[_-]|\b|$)"
    return filter(lambda slide: not re.search(pattern, slide.stem), slides)


def get_relative_dir_path(path: Path) -> Path:
    return path.relative_to(PATH_PREFIX).parent


def positive_training_wsis() -> Iterable[Path]:
    return filter_he_slides(
        itertools.chain(
            Path(PATH_PREFIX, "Cytokeratin_mask_colorectal_TMAs").rglob("*.mrxs"),
            Path(PATH_PREFIX, "Cytokeratin_mask_new_breast_TNBC-TMAS").rglob("*.mrxs"),
        )
    )


def negative_training_wsis() -> Iterable[Path]:
    return filter_he_slides(Path(PATH_PREFIX, "dataset1-2023").rglob("*-0.tiff"))


def training_wsis() -> Iterable[Path]:
    return itertools.chain(positive_training_wsis(), negative_training_wsis())


def test_wsis_lymph_nodes() -> Iterable[Path]:
    return filter_he_slides(Path(PATH_PREFIX, "Annotated_IHC_for_test").rglob("*.mrxs"))


def test_wsis_colorectal() -> Iterable[Path]:
    return filter_he_slides(
        Path(PATH_PREFIX, "Cytokeratin_mask_final_scans").rglob("*.mrxs")
    )


def test_wsis() -> Iterable[Path]:
    return itertools.chain(test_wsis_lymph_nodes(), test_wsis_colorectal())
