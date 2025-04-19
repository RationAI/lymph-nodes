from pathlib import Path

import mlflow
import numpy as np
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
from tqdm import tqdm

from cytokeratin_mask.cytokeratin_mask import cytokeratin_mask
from cytokeratin_mask.tissue_slicer import tissue_slicer


TISSUE_MASKS_URI = (
    "mlflow-artifacts:/68/76ab7eda9a3d4a7fb3ff7e7baf904343/artifacts/tissue_masks"
)
REFERENCE_PATH = "/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry/"
DEST_DIR = "./data/cytokeratin_mask"
MPP = 0.5
MASK_LEVEL = 3


def main() -> None:
    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000")
    mlflow.set_experiment("Lymph Nodes")
    mlflow.start_run(run_name="Cytokeratin Mask")

    # Download tissue masks
    mlflow.artifacts.download_artifacts(
        artifact_uri=TISSUE_MASKS_URI, dst_path="./data"
    )

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

        with OpenSlide(wsi_path) as slide:
            mask_mpp_x, mask_mpp_y = slide_resolution(slide, level=MASK_LEVEL)

        factror_x, factor_y = mask_mpp_x / mpp_x, mask_mpp_y / mpp_y

        image = pyvips.Image.new_from_file(str(wsi_path), level=level)
        image = rgba2rgb(image)

        tissue_mask_path = Path(
            "data/tissue_masks",
            wsi_path.relative_to(REFERENCE_PATH).parent,
            f"{wsi_path.stem}.tiff",
        )
        tissue_mask = pyvips.Image.new_from_file(tissue_mask_path, page=MASK_LEVEL)
        slices = tissue_slicer(tissue_mask)

        mask = np.memmap(
            str(wsi_path.name) + ".nmp",
            dtype=np.uint8,
            mode="w+",
            shape=(image.height, image.width),
        )

        for x, y, w, h, _area in tqdm(slices):
            # Adjust
            x = round(x * factror_x)
            y = round(y * factor_y)
            w = round(w * factror_x)
            h = round(h * factor_y)

            tma_mask = cytokeratin_mask(
                image.crop(x, y, w, h), mask_min_area=7, holes_min_area=5
            )
            mask[y, x, w + h, x + w] = tma_mask

        mask.flush()

        mask = pyvips.Image.new_from_array(mask)

        mask_path = Path(
            DEST_DIR,
            wsi_path.relative_to(REFERENCE_PATH).parent,
            f"{wsi_path.stem}.tiff",
        )
        mask_path.parent.mkdir(exist_ok=True, parents=True)

        write_big_tiff(mask, mask_path, mpp_x=mpp_x, mpp_y=mpp_y)

    process_items(wsis, process_item, max_concurrent=2)

    mlflow.log_artifacts(DEST_DIR, artifact_path="cytokeratin_masks")
    mlflow.end_run()


if __name__ == "__main__":
    main()
