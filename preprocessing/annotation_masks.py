import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import hydra
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.openslide import OpenSlide


if TYPE_CHECKING:
    from PIL import Image
    from PIL.ImageDraw import _Ink  # type: ignore[attr-defined]


class JSONPolygonMask:
    """Rasterizes polygon regions from a Fabric.js-style JSON annotation file.

    Expects ``{"objects": [...]}`` where each object is either a rectangle
    (``type: "rect"`` with ``left``/``top``/``width``/``height``) or a free-form
    polygon (``points: [{"x": ..., "y": ...}, ...]``). Every region is drawn with
    the same label (255) — a single binary mask, not a multi-class one.

    Structurally mirrors ``rationai.masks.annotations.PolygonMask`` (same
    constructor shape, ``__call__``, and the methods that class makes abstract) but
    doesn't subclass it: that class currently fails to import in this environment —
    ``from PIL.ImageDraw import _Ink`` at module level, but the Pillow version
    pinned in uv.lock doesn't export ``_Ink`` at runtime. Worth raising upstream;
    switching this back to a real ``PolygonMask`` subclass is a one-line change
    once that's fixed.
    """

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
        self.mask_size = mask_size
        self.mask_mpp_x = mask_mpp_x
        self.mask_mpp_y = mask_mpp_y
        self.mode = mode
        with Path(path).open() as f:
            self.root = json.load(f)

    @property
    def annotation_mpp_x(self) -> float:
        return self.annotation_mpp[0]

    @property
    def annotation_mpp_y(self) -> float:
        return self.annotation_mpp[1]

    @property
    def regions(self) -> Iterable[tuple[dict[str, Any], "_Ink"]]:
        regions = self.root["objects"]
        return zip(regions, [255] * len(regions), strict=False)

    def get_region_coordinates(self, region: dict[str, Any]) -> Iterable[tuple[float, float]]:
        if region["type"] == "rect":
            left, top = float(region["left"]), float(region["top"])
            width, height = float(region["width"]), float(region["height"])
            yield from [
                (left, top),
                (left, top + height),
                (left + width, top + height),
                (left + width, top),
            ]
        else:
            for vertex in region["points"]:
                yield float(vertex["x"]), float(vertex["y"])

    def __call__(self) -> "Image.Image":
        from PIL import Image, ImageDraw

        scale_x = self.annotation_mpp_x / self.mask_mpp_x
        scale_y = self.annotation_mpp_y / self.mask_mpp_y

        mask = Image.new(self.mode, size=self.mask_size)
        canvas = ImageDraw.Draw(mask)

        for region, label in self.regions:
            polygon = [
                (x * scale_x, y * scale_y)
                for x, y in self.get_region_coordinates(region)
            ]
            canvas.polygon(xy=polygon, outline=label, fill=label)

        return mask


@ray.remote(memory=2 * 1024**3)
def process_slide(
    slide_path: str, annotation_dir: Path, mpp: float, output_path: Path
) -> None:
    annotation_path = annotation_dir / f"{Path(slide_path).stem}.json"
    if not annotation_path.exists():
        # No annotation for this slide — same "nothing to draw" contract
        # OverlayCoverage relies on downstream: a missing mask file means null
        # coverage for that slide's tiles, not an error.
        return

    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        mpp_x, mpp_y = slide.slide_resolution(level)
        # Annotation coordinates are in full-resolution (level 0) pixel space.
        annotation_mpp = slide.slide_resolution(0)
        mask_size = slide.level_dimensions[level]

    mask = JSONPolygonMask(
        annotation_mpp=tuple(annotation_mpp),
        path=annotation_path,
        mask_size=mask_size,
        mask_mpp_x=mpp_x,
        mask_mpp_y=mpp_y,
    )()

    mask_path = output_path / Path(slide_path).with_suffix(".tiff").name
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    write_big_tiff(
        pyvips.Image.new_from_array(mask),
        path=mask_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )


@with_cli_args(["+preprocessing=annotation_masks"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    """General JSON-polygon-annotation -> raster mask generator.

    Reusable across annotation purposes (e.g. custom ignore regions, or any other
    hand-drawn polygon mask) — ``annotation_dir`` and ``artifact_path`` are both
    config-driven, not hardcoded to one specific mask type.
    """
    slides = hydra.utils.instantiate(config.dataset.slides)

    with TemporaryDirectory() as output_dir:
        process_items(
            slides,
            process_item=process_slide,
            fn_kwargs={
                "annotation_dir": Path(config.project_root) / Path(config.annotation_dir),
                "mpp": config.mpp,
                "output_path": Path(output_dir),
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(local_dir=output_dir, artifact_path=config.artifact_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
