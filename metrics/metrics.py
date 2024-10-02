# Copyright (c) The RationAI team

import logging
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

from metrics.base_wsi import extract_tile


__version__ = "1.0"

log = logging.getLogger("tyler-filter")
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(levelname).1s][%(process)d][%(filename)s][%(funcName)-25.25s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
)


class Base(ABC):
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
    def calculate_metrics(self) -> None:
        """The method contains the main code that works with tiles."""
        ...


class Metrics(Base):
    """Metrics.

    Metrics calculates metrics from the result heatmap mask of the dataset generated from a specific model.
    The result is the sensitivity, specificity, precision, and f1 score for the entire dataset for every threshold number.
    """

    def __init__(self, tiles_uri: str | None, mask_uri: str | None):
        super().__init__(tiles_uri)

        self.mask_uri = mask_uri
        if mask_uri is not None:
            try:
                data_uri = self._validate_data_uris(mask_uri)
                self.mask_fp = self._load_mlflow_artifacts_data(data_uri)
            except ValueError:
                self.mask_fp = pathlib.Path(mask_uri)

    def log_params(self) -> None:
        mlflow.log_param("tiles_uri", self.tiles_uri)
        mlflow.log_param("mask_uri", self.mask_uri)

    def calculate_metrics(self):
        log.info("Starting to calculate tiles...")

        # Used thresholds
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Calculating values of the confusion matrix
        fp, fn, tp, tn = self.__confusion_matrics(thresholds)

        # Calculating sensitivity, specificity, precision, and F1 score.
        log.info("Calculated TP, TN, FP, FN.")
        for i, threshold in enumerate(thresholds):
            sensitivity = tp[i] / (tp[i] + fn[i])
            precision = tp[i] / (tp[i] + fp[i])
            specificity = tn[i] / (tn[i] + fp[i])
            f1_score = 2 * sensitivity * precision / (sensitivity + precision)
            log.info(
                f"For threshold {threshold}: TP:{tp[i]}, TN:{tn[i]}, FP:{fp[i]}, FN:{fn[i]}"
            )
            log.info(
                f"For threshold {threshold} metrics: sensitivity:{sensitivity}, precision:{precision}, specificity:{specificity}, f1 score:{f1_score}"
            )

        log.info("Calculating metrics done.")

    def __confusion_matrics(
        self, thresholds: list[int]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        len_threshold = len(thresholds)
        """Calculates values of the confusion matrix based on heatmap masks and tiles.

            Arguments:
                thresholds {List[int]}: The list of threshold values used.

            Returns:
                Tuple[List[int], List[int], List[int], List[int]]: A tuple of four lists representing false positives, false negatives, true positives, and true negatives, respectively.
        """

        # Each list corresponds to a segment of the resulting confusion matrix
        # Each position in the lists corresponds to a threshold value
        fp = [0 for _ in range(len_threshold)]
        fn = [0 for _ in range(len_threshold)]
        tp = [0 for _ in range(len_threshold)]
        tn = [0 for _ in range(len_threshold)]

        for _, tile in tqdm(self.tiles.reset_index().iterrows()):
            slide_data = self.metadata[
                (self.metadata["slide_name"] == tile["slide_name"])
            ]
            slide_data = slide_data.iloc[0]

            mask_path = pathlib.Path(self.mask_fp / (tile["slide_name"] + ".tiff"))
            mask = extract_tile(
                mask_path,
                tile["coord_x"],
                tile["coord_y"],
                slide_data["center_size"],
                slide_data["sample_level"],
            )

            # Get the color opacity of each tile from the heatmap mask.
            pixel_value = mask[0][0][0]
            intensity = pixel_value / 255

            for i, threshold in enumerate(thresholds):
                if tile["class_id"] == 1 and intensity > threshold:
                    tp[i] = tp[i] + 1
                elif tile["class_id"] == 1 and intensity <= threshold:
                    fn[i] = fn[i] + 1
                elif tile["class_id"] == 0 and intensity <= threshold:
                    tn[i] = tn[i] + 1
                else:
                    fp[i] = fp[i] + 1

        return fp, fn, tp, tn
