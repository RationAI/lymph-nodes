from pathlib import Path

import mlflow
import pyvips
import ray
from openslide import OpenSlide
from rationai.masks import (
    closest_level,
    process_items,
    slide_resolution,
    write_big_tiff,
)
from skimage.color import rgba2rgb

from cytokeratin_mask.cytokeratin_mask import cytokeratin_mask


REFERENCE_PATH = "/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry/"
DEST_DIR = "./data/cytokeratin_mask"
MPP = 4


def main() -> None:
    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000")
    mlflow.set_experiment("Lymph Nodes")
    mlflow.start_run(run_name="Cytokeratin Mask")

    wsis = [
        *Path(REFERENCE_PATH, "Cytokeratin_mask_colorectal_TMAs").glob("DAB-CK-*.mrxs"),
        *Path(REFERENCE_PATH, "Cytokeratin_mask_final_scans").glob("FIN-CK-*.mrxs"),
        *Path(REFERENCE_PATH, "Cytokeratin_mask_new_breast_TNBC-TMAS/ckae").glob(
            "*.mrxs"
        ),
    ]

    @ray.remote
    def process_item(wsi_path: Path) -> None:
        with OpenSlide(wsi_path) as slide:
            level = closest_level(slide, MPP)
            mpp_x, mpp_y = slide_resolution(slide, level)

        image = pyvips.Image.new_from_file(str(wsi_path), level=level)
        image = rgba2rgb(image)

        mask = cytokeratin_mask(image, mask_min_area=7, holes_min_area=5)
        mask = pyvips.Image.new_from_array(mask)

        mask_path = Path(
            DEST_DIR,
            wsi_path.relative_to(REFERENCE_PATH).parent,
            f"{wsi_path.stem}.tiff",
        )
        mask_path.parent.mkdir(exist_ok=True, parents=True)

        write_big_tiff(pyvips.new_from_array(mask), mask_path, mpp_x=mpp_x, mpp_y=mpp_y)

    process_items(wsis, process_item, max_concurrent=4)

    mlflow.log_artifacts(DEST_DIR, artifact_path="cytokeratin_masks")
    mlflow.end_run()


if __name__ == "__main__":
    main()
