from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET

import pyvips
import ray
from openslide import OpenSlide
from PIL.ImageDraw import _Ink
from rationai.masks import process_items, write_big_tiff
from rationai.masks.annotations import XMLPolygonMask

from preprocessing.utils import get_level_by_mpp, get_mpp, mpp_to_ppmm


class MetastazisMask(XMLPolygonMask):
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
        return self.mask_mpp_x  # mppx for annotation is not provided

    @property
    def annotation_mpp_y(self) -> float:
        return self.mask_mpp_y  # mppx for annotation is not provided


def metastazis_mask(slide_path: Path, tissue_mask_mpp: float, dest_dir: Path) -> None:
    annotation_file = Path(slide_path.parent, f"{slide_path.stem}.xml")

    with OpenSlide(slide_path) as slide:
        level = get_level_by_mpp(slide, mpp=tissue_mask_mpp)
        mask_mpp_x, mask_mpp_y = get_mpp(slide, level=level)
        annotator = MetastazisMask(
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
        pyvips.Image.new_from_array(mask),
        path=mask_path,
        xres=xres,
        yres=yres,
    )


def get_metastazis_masks(slide_paths: list[Path]) -> None:
    metastazis_mask_mpp = 2

    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path("data/tissue_masks", slide_path.parent.stem)  # keep last level
        metastazis_mask(slide_path, metastazis_mask_mpp, dest_dir)

    process_items(slide_paths, process_item=process_slide)
