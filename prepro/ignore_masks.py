import os
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET

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
from rationai.masks.annotations import XMLPolygonMask

from prepro.utils import get_relative_dir_path, mpp_to_ppmm


class IgnoreMask(XMLPolygonMask):
    def __init__(
        self,
        annotation_mpp: tuple[float, float],
        path: str | Path,
        mask_size: tuple[int, int],
        mask_mpp_x: float,
        mask_mpp_y: float,
        mode: str = "L",
    ) -> None:
        self.annotation_mpp = annotation_mpp
        super().__init__(
            path=path,
            mask_size=mask_size,
            mask_mpp_x=mask_mpp_x,
            mask_mpp_y=mask_mpp_y,
            mode=mode,
        )

    @property
    def regions(self) -> Iterable[tuple[ET.Element, _Ink]]:
        regions = self.root.findall("Annotations/Annotation")
        return zip(regions, [255] * len(regions), strict=False)

    def get_region_coordinates(
        self, region: ET.Element
    ) -> Iterable[tuple[float, float]]:
        for vertex in region.findall("Coordinates/Coordinate"):
            yield float(vertex.get("X")), float(vertex.get("Y"))

    @property
    def annotation_mpp_x(self) -> float:
        return self.annotation_mpp[0]

    @property
    def annotation_mpp_y(self) -> float:
        return self.annotation_mpp[1]


def ignrore_mask(slide_path: Path, desired_mpp: float, dest_dir: Path) -> None:
    annotation_file = Path(slide_path.parent, f"{slide_path.stem}.xml")

    if not os.path.exists(annotation_file):
        with open("data/missing_annotations.txt", "a") as f:
            f.write(f"{slide_path}\n")
        return

    with OpenSlide(slide_path) as slide:
        level = closest_level(slide, mpp=desired_mpp)
        annotation_mpp = slide_resolution(
            slide, level=0
        )  # mppx for annotation is not provided
        mask_mpp_x, mask_mpp_y = slide_resolution(slide, level=level)
        annotator = IgnoreMask(
            annotation_mpp=annotation_mpp,
            path=annotation_file,
            mask_size=slide.level_dimensions[level],
            mask_mpp_x=mask_mpp_x,
            mask_mpp_y=mask_mpp_y,
        )

    mask = annotator()

    xres, yres = mpp_to_ppmm((mask_mpp_x, mask_mpp_y))

    mask_path = Path(dest_dir, f"{slide_path.stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(
        pyvips.Image.new_from_array(mask),  # type: ignore pyvips.Image
        path=mask_path,
        xres=xres,
        yres=yres,
    )


def generate_ignore_masks(
    slide_paths: Iterable[Path], mpp: float, reference_path: str, dest: str
) -> None:
    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path(
            dest, get_relative_dir_path(slide_path, Path(reference_path))
        )  # keep last level
        ignrore_mask(slide_path, mpp, dest_dir)

    process_items(slide_paths, process_item=process_slide)

    mlflow.log_artifacts(dest, artifact_path="ignore_masks")
