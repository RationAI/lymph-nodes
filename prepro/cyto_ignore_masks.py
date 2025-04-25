import json
from collections.abc import Iterable
from pathlib import Path

import mlflow
import pyvips
import ray
from openslide import OpenSlide
from PIL.ImageDraw import _Ink
from rationai.masks import (
    closest_level,
    process_items,
    slide_resolution,
    write_big_tiff,
)
from rationai.masks.annotations import PolygonMask

from prepro.utils import get_relative_dir_path


class JSONIgnoreMask(PolygonMask[dict]):
    def __init__(
        self,
        annotation_mpp: tuple[float, float],
        path: str | Path,
        mask_size: tuple[int, int],
        mask_mpp_x: float,
        mask_mpp_y: float,
        mode: str = "P",
    ) -> None:
        self.annotation_mpp = annotation_mpp
        super().__init__(
            mask_size=mask_size,
            mask_mpp_x=mask_mpp_x,
            mask_mpp_y=mask_mpp_y,
            mode=mode,
        )
        with open(path) as f:
            self.root = json.load(f)

    @property
    def regions(self) -> Iterable[tuple[dict, _Ink]]:
        regions = self.root["objects"]
        return zip(regions, [255] * len(regions), strict=False)

    def get_region_coordinates(self, region: dict) -> Iterable[tuple[float, float]]:
        for vertex in region["points"]:
            yield float(vertex["x"]), float(vertex["y"])

    @property
    def annotation_mpp_x(self) -> float:
        return self.annotation_mpp[0]

    @property
    def annotation_mpp_y(self) -> float:
        return self.annotation_mpp[1]


def cyto_ignore_mask(slide_path: Path, desired_mpp: float, dest_dir: Path) -> None:
    json_annotation_file = Path(
        slide_path.parent, f"{slide_path.stem}-2025_04_25-all.json"
    )

    annot_class = JSONIgnoreMask
    annotation_file = json_annotation_file

    if annot_class is None:
        with open("data/missing_annotations.txt", "a") as f:
            f.write(f"{slide_path}\n")
        return

    with OpenSlide(slide_path) as slide:
        level = closest_level(slide, mpp=desired_mpp)
        annotation_mpp = slide_resolution(
            slide, level=0
        )  # mppx for annotation is not provided
        mpp_x, mpp_y = slide_resolution(slide, level=level)

        annotator = annot_class(
            annotation_mpp=annotation_mpp,
            path=annotation_file,
            mask_size=slide.level_dimensions[level],
            mask_mpp_x=mpp_x,
            mask_mpp_y=mpp_y,
        )

    mask = annotator()

    mask_path = Path(dest_dir, f"{slide_path.stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(
        pyvips.Image.new_from_array(mask),  # type: ignore pyvips.Image
        path=mask_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )


def generate_cyto_ignore_masks(
    slide_paths: Iterable[Path], mpp: float, reference_path: str, dest: str
) -> None:
    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path(
            dest, get_relative_dir_path(slide_path, Path(reference_path))
        )  # keep last level
        cyto_ignore_mask(slide_path, mpp, dest_dir)

    process_items(slide_paths, process_item=process_slide)

    mlflow.log_artifacts(dest, artifact_path="cyto_ignore_masks")
