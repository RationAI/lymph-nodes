# Copyright (c) The RationAI team

import logging
import math
import os
import pathlib
from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyvips
from PIL import Image
from tqdm import tqdm

from preprocessing.base_wsi import extract_tile


__version__ = "1.0"

log = logging.getLogger("tyler-filter")
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(levelname).1s][%(process)d][%(filename)s][%(funcName)-25.25s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
)


class BaseFilter(ABC):
    """BaseFilter.

    BaseFilter is an abstract class defining the interface
    for tile filters and other preprocessing that depend on tiles.
    """

    def __init__(self, tiles_uri: str | None):
        self.tiles_uri = tiles_uri

        if tiles_uri is not None:
            data_uri = self._validate_data_uris(tiles_uri)
            self.metadata, self.tiles = self._load_mlflow_artifacts_tiles(data_uri)

        self.output_dir = pathlib.Path(os.getcwd())
        self._log_script()

    # Mlflow Methods

    @staticmethod
    def _load_mlflow_artifacts_tiles(
        uri: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The method downloads 'slide.parquet' and 'tile.parquet' from an MLflow artifact.

        Args:
            uri (str): MLflow URI.

        Raises:
            FileNotFoundError: Cannot fetch data from MLflow URI.
            FileNotFoundError: Cannot load data from file.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Returns 'slide.parquet' and 'tile.parquet' in a pandas DataFrame structure.
        """
        metadata = None
        tiles = None
        try:
            log.info(f"Downloading tiles from mlflow uri {uri}")
            fp = pathlib.Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            log.info(f"Data saved to {fp}")
        except mlflow.MlflowException as e:
            raise FileNotFoundError(f"Cannot fetch data from {uri}.") from e

        try:
            metadata = pd.read_parquet(fp / "slides.parquet")
            tiles = pd.read_parquet(fp / "tiles.parquet")
        except OSError as e:
            raise FileNotFoundError(f"Cannot load data from {fp}.") from e

        return metadata, tiles

    @staticmethod
    def _validate_data_uris(uri: str) -> str:
        """Validates an MLflow URI.

        Args:
            uri (str): MLflow URI.

        Raises:
            TypeError: Data URI must be a string.
            ValueError: Data URI is not in a valid format.

        Returns:
            str: MLflow URI.
        """
        if not isinstance(uri, str):
            raise TypeError(f"Data URI must be a string, got {type(uri)}")
        if not uri.startswith("mlflow-artifacts:/"):
            raise ValueError(
                f"Data URI must start with 'mlflow-artifacts:/', got {uri}"
            )
        return uri

    @staticmethod
    def _load_mlflow_artifacts_data(
        uri: str,
    ) -> Path:
        """Downloads general data from an MLflow artifact.

        Args:
            uri (str): MLflow URI.

        Raises:
            FileNotFoundError: Cannot fetch data from MLflow URI.
            FileNotFoundError: Cannot load data from file.

        Returns:
            pathlib.PosixPath: Posix path to the data.
        """
        fp = None
        try:
            log.info(f"Downloading data from mlflow uri {uri}")
            # return mask_fp
            fp = pathlib.Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            log.info(f"Data saved to {fp}")
        except mlflow.MlflowException as e:
            raise FileNotFoundError(f"Cannot fetch data from {uri}.") from e

        return fp

    @staticmethod
    def _save_as_tif(input_im, output_path) -> None:
        """Save image in tiff format.

        Args:
            input_im (NDArray) : Image to be saved
            output_path (Path): Path where image will be saved
        """
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        vips_im = pyvips.Image.new_from_memory(
            np.array(Image.fromarray(input_im).convert("RGBA")),
            input_im.shape[1],
            input_im.shape[0],
            4,
            format="uchar",
        )
        vips_im.tiffsave(
            str(output_path),
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
        )

    @staticmethod
    def _log_script():
        """Logs the entire script to MLflow as an artifact."""
        mlflow.log_artifact("/home/jovyan/tylerfilter/tylerfilter", "preprocessing")

    @abstractmethod
    def log_params(self) -> None:
        """Logs inserted parameters into MLflow as parameters."""
        ...

    @abstractmethod
    def filter(self) -> None:
        """The method contains the main code that works with tiles."""
        ...


class MaskFilter(BaseFilter):
    """MaskFilter.

    MaskFilter assigns labels to tiles. The class has two modes:

    - In negative mode, all tiles receive the label 0.

    - In positive mode, the Ground Truth Mask Coverage Ratio (GTMCR) is calculated
      for each tile based on the Ground Truth Mask (GTM). Tiles are then filtered and labeled
      as 1 based on specified GTMCR threshold values.
    """

    def __init__(
        self,
        positive: bool = True,
        ratio_min: float = 0.1,
        ratio_max: float = 0.9,
        masks_uri: str | None = None,
        tiles_uri: str | None = None,
    ) -> None:
        super().__init__(tiles_uri)
        self.masks_uri = masks_uri
        self.positive = positive

        # Parsing masks URI
        if masks_uri is not None:
            data_uri = self._validate_data_uris(masks_uri)
            self.mask_fp = self._load_mlflow_artifacts_data(data_uri)

            self.out_mask_dir = pathlib.Path(os.getcwd()) / "mask"
            self.out_bg_dir = pathlib.Path(os.getcwd()) / "bg"

        if 1 < ratio_min < 0 or 1 < ratio_max < 0:
            raise ValueError("Ratio has to be a float between 0 and 1.")

        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

        if positive:
            self.metadata["is_metastasis"] = 1
        else:
            self.metadata["is_metastasis"] = 0

    def log_params(self) -> None:
        mlflow.log_param("positive", self.positive)
        mlflow.log_param("ratio_min", self.ratio_min)
        mlflow.log_param("ratio_max", self.ratio_min)
        mlflow.log_param("data_uri", self.masks_uri)
        mlflow.log_param("tiles_uri", self.tiles_uri)

    def filter(self):
        log.info("Starting to filtering tiles...")

        if not self.positive:
            self.__negative_mode()
            return

        self.__positive_mode()

    def __negative_mode(self) -> None:
        """In negative mode, all tiles are assigned class_id 0."""
        log.info("Negative mode.")

        self.tiles["class_id"] = 0

        log.info("Saving dataset")
        self.tiles.to_parquet(self.output_dir / "tiles.parquet")
        self.metadata.to_parquet(self.output_dir / "slides.parquet")

        log.info("Saving artifacts")
        mlflow.log_artifact(
            str(self.output_dir / "slides.parquet"), artifact_path="dataset"
        )
        mlflow.log_artifact(
            str(self.output_dir / "tiles.parquet"), artifact_path="dataset"
        )

    def __positive_mode(self) -> None:
        """In positive mode, the GTMCR is calculated.

        Tiles are filtered based on self.ratio_max and self.ratio_min thresholds.
        Unfiltered tiles are assigned class_id 1 and their
        annotation_coverage is determined based on the GTMCR.

        The method also calculates a histogram of tiles based on GTMCR.

        Processed tiles are saved into new tiles.parquet and slides.parquet files.
        """
        log.info("Positive mode.")

        filtered_tiles = []
        filter_histogram = [0 for i in range(11)]

        for _, tile in tqdm(self.tiles.iterrows()):
            # Get slide and extract the corresponding tile
            mask_path = pathlib.Path(self.mask_fp / (tile["slide_name"] + ".tif"))
            slide = self.metadata[(self.metadata["slide_name"] == tile["slide_name"])]
            slide = slide.iloc[0]

            mask = extract_tile(
                mask_path,
                tile["coord_x"],
                tile["coord_y"],
                slide["tile_size"],
                slide["sample_level"],
            )
            _, counts = np.unique(mask[:, :, 0], return_counts=True)

            # Calculate GTMCR
            tile_pixels = slide["tile_size"] * slide["tile_size"]
            gtmcr = counts[1:].sum() / tile_pixels
            if self.ratio_max >= gtmcr >= self.ratio_min:
                filter_histogram[math.floor(gtmcr * 10)] += 1
                tile["class_id"] = 1
                tile["annot_coverage"] = gtmcr
                filtered_tiles.append(tile)

        new_tiles_df = pd.DataFrame(
            data=filtered_tiles, columns=self.tiles.columns
        ).reset_index(drop=True)

        slide_names = new_tiles_df["slide_name"].unique()
        new_metadata_df = self.metadata[self.metadata["slide_name"].isin(slide_names)]

        log.info(f"TylerFilter filtered {len(self.tiles) - len(new_tiles_df)} tyles.")
        log.info(f"Tiles coverage distribution: {filter_histogram}.")

        log.info("Saving dataset")
        new_tiles_df.to_parquet(self.output_dir / "tiles.parquet")
        log.info(
            f"Saved {len(new_tiles_df)} tiles to {self.output_dir / 'tiles.parquet'!s}"
        )
        new_metadata_df.to_parquet(self.output_dir / "slides.parquet")
        log.info(f"Saved metadata to {self.output_dir / 'slides.parquet'!s}")
        log.info("Saving artifacts")

        # Log result files into MlFlow as artifacts
        mlflow.log_artifact(
            str(self.output_dir / "slides.parquet"), artifact_path="dataset"
        )
        mlflow.log_artifact(
            str(self.output_dir / "tiles.parquet"), artifact_path="dataset"
        )


class SlideFiler(BaseFilter):
    """SlideFiler.

    SlideFiler removes tiles and slides based on slide name.
    """

    def __init__(self, tiles_uri: str | None, slides: list[str]):
        super().__init__(tiles_uri)

        self.slides = slides

    def log_params(self) -> None:
        mlflow.log_param("slides", self.slides)

    def filter(self):
        log.info("Starting to filtering tiles...")

        tiles_filtered_df = self.tiles[
            ~self.tiles["slide_name"].isin(self.slides)
        ].reset_index(drop=True)

        slide_names = tiles_filtered_df["slide_name"].unique()
        metadata_filtered_df = self.metadata[
            self.metadata["slide_name"].isin(slide_names)
        ]

        log.info(
            f"TylerFilter filtered {len(self.tiles) - len(tiles_filtered_df)} tyles."
        )

        log.info("Saving dataset")
        tiles_filtered_df.to_parquet(self.output_dir / "tiles.parquet")
        log.info(
            f"Saved {len(tiles_filtered_df)} tiles to {self.output_dir / 'tiles.parquet'!s}"
        )
        metadata_filtered_df.to_parquet(self.output_dir / "slides.parquet")
        log.info(f"Saved metadata to {self.output_dir / 'slides.parquet'!s}")
        log.info("Saving artifacts")

        mlflow.log_artifact(
            str(self.output_dir / "slides.parquet"), artifact_path="dataset"
        )
        mlflow.log_artifact(
            str(self.output_dir / "tiles.parquet"), artifact_path="dataset"
        )
