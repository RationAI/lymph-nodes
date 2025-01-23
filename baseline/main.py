from collections.abc import Iterable
from pathlib import Path

import ray
from rationai.masks import (
    process_items,
    write_big_tiff,
)

from baseline.color_separation import color_separation
from baseline.utils import save_thumbnail


def baseline(slides: Iterable[Path], reference_path: str, dest: str) -> None:
    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path(dest, slide_path.relative_to(reference_path).parent)
        mask = color_separation(slide_path)

        mask_path = Path(dest_dir, f"{Path(slide_path).stem}.tiff")
        mask_path.parent.mkdir(exist_ok=True, parents=True)

        write_big_tiff(mask, path=mask_path, mpp_x=1, mpp_y=1)
        save_thumbnail(mask_path, "thumbnail.tiff")

    process_items(slides, process_slide)


baseline([Path("/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry")])
