from collections.abc import Iterator
from dataclasses import dataclass
from random import randint
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from prepro.annotation_masks import generate_annotation_masks
from prepro.color_separation import generate_color_separation_masks
from prepro.custom_ignore_masks import generate_custom_ignore_masks
from prepro.data_source import ChainedDataSources, DataSource
from prepro.ignore_masks import generate_ignore_masks
from prepro.tiling import Args, tile_dataset
from prepro.tissue_masks import generate_tissue_masks


@dataclass
class Dataset:
    datasource: DataSource
    source_kind: str
    slide_metastazis: bool


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="./configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.metadata.mlflow_uri)
    mlflow.set_experiment(config.metadata.experiment_name)

    active_run = mlflow.start_run(run_name=config.metadata.run_name)

    if config.metadata.cytokeratin_masks_path:
        print("Donwloading cytokeratin masks")
        mlflow.artifacts.download_artifacts(
            artifact_uri=config.metadata.cytokeratin_masks_path, dst_path="./data"
        )

    # mlflow.artifacts.download_artifacts(
    #     artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/tissue_masks",
    #     dst_path="./data",
    # )

    # mlflow.artifacts.download_artifacts(
    #     artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/ignore_masks",
    #     dst_path="./data",
    # )

    # mlflow.artifacts.download_artifacts(
    #     artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/annotation_masks",
    #     dst_path="./data",
    # )

    # mlflow.artifacts.download_artifacts(
    #     artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/custom_ignore_masks",
    #     dst_path="./data",
    # )

    # mlflow.artifacts.download_artifacts(
    #     artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/color_separation_masks",
    #     dst_path="./data",
    # )

    print("Prepare datasources")

    datasets = {
        "fnbrno": Dataset(
            datasource=DataSource(
                "/mnt/data/Projects/Data/FNBrno/lymph_nodes/maternity_hospital_dataset",
                glob_pattern=[
                    "*B-4008-24-6-2-AE-1.czi",
                    "*B209-23-9-6-CK19-1.czi",
                    "*B977-24-2-2-AE-1.czi",
                    "*B977-24-3-2-AE-1.czi",
                    "*_B977-24-6-2-AE-1.czi",
                    "*B977-24-7-2-AE-1.czi",
                    "*B3310-24-3-6-AE-1.czi",
                    "*B1858-23-16-2-AE-1.czi",
                    "*B1030-24-1-2-AE-0.czi",
                    "*B3563-24-3-2-AE-0.czi",
                ],
            ),
            source_kind="lymph_node",
            slide_metastazis=None,
        )
    }

    # DataSource

    def map_datasets(datasets: dict[str, Any]) -> Iterator[DataSource]:
        if isinstance(datasets, Dataset):
            yield datasets.datasource
        else:
            for key in datasets:
                yield from map_datasets(datasets[key])

    # Masks

    # Tissue masks
    print("Generating tissue masks")
    generate_tissue_masks(
        slide_paths=ChainedDataSources(list(map_datasets(datasets))),
        mpp=2,
        reference_path=config.metadata.relative_path_prefix,
        dest=config.metadata.tissue_mask_dest,
    )

    # Annotation masks
    # print("Generating annotation masks")
    # generate_annotation_masks(
    #     slide_paths=ChainedDataSources(
    #         [
    #             datasets["lymhps-2023"]["positive-test"].datasource,
    #             datasets["positive-lymph-nodes"].datasource,
    #         ]
    #     ),
    #     mpp=2,
    #     reference_path=config.metadata.relative_path_prefix,
    #     dest=config.metadata.annotation_mask_dest,
    # )

    # # Ignore masks
    # print("Generating ignore masks")
    # generate_ignore_masks(
    #     slide_paths=ChainedDataSources(
    #         [
    #             datasets["tmas"]["test"].datasource,
    #             datasets["tmas"]["train"].datasource,
    #             datasets["tmas"]["val"].datasource,
    #         ]
    #     ),
    #     mpp=2,
    #     reference_path=config.metadata.relative_path_prefix,
    #     dest=config.metadata.ignore_mask_dest,
    # )

    # # CustomIgnore masks
    # print("Generating custom ignore masks")
    # generate_custom_ignore_masks(
    #     slide_paths=ChainedDataSources(
    #         [
    #             datasets["tmas"]["test"].datasource,
    #             datasets["tmas"]["train"].datasource,
    #             datasets["tmas"]["val"].datasource,
    #             datasets["lymhps-2023"]["negative-test"].datasource,
    #             datasets["lymhps-2023"]["positive-test"].datasource,
    #         ]
    #     ),
    #     mpp=2,
    #     reference_path=config.metadata.relative_path_prefix,
    #     dest=config.metadata.ignore_mask_dest,
    # )

    # # Color separation masks
    # print("Generating color separation masks")
    # generate_color_separation_masks(
    #     slide_paths=ChainedDataSources(list(map_datasets(datasets))),
    #     mpp=1,
    #     reference_path=config.metadata.relative_path_prefix,
    #     dest=config.metadata.color_separation_mask_dest,
    # )

    # Tiling
    print("Tiling")

    def tile_config(
        dataset: DataSource, source_kind: str, slide_metastazis: bool
    ) -> Args:
        return {
            "slides": dataset,
            "source_kind": source_kind,
            "slide_metastazis": slide_metastazis,
            "desired_mpp": config.metadata.tiling.mpp,
            "tissue_threshold": config.metadata.tiling.tissue_threshold,
            "tile_crop": config.metadata.tiling.tile_crop,
            "min_roi_area": config.metadata.tiling.min_roi_area,
            "stride": config.metadata.tiling.stride,
            "tissue_masks_dir": config.metadata.tissue_mask_dest,
            "cytokeratin_masks_dir": config.metadata.cytokeratin_mask_dest,
            "color_separation_mask_dir": config.metadata.color_separation_mask_dest,
            "ignore_mask_dir": config.metadata.ignore_mask_dest,
            "custom_ignore_mask_dir": config.metadata.custom_ignore_mask_dest,
            "annotation_masks_dir": config.metadata.annotation_mask_dest,
            "relative_path_prefix": config.metadata.relative_path_prefix,
        }

    def process_dataset(dataset: dict[str, Any], prefix: str = "") -> None:
        for key in dataset:
            name: str = f"{prefix}/{key}" if prefix else key

            if isinstance(dataset[key], Dataset):
                data = dataset[key]
                tile_dataset(
                    [
                        tile_config(
                            data.datasource, data.source_kind, data.slide_metastazis
                        )
                    ],
                    dataset_name=name,
                )

            else:
                process_dataset(dataset[key], name)

    process_dataset(datasets)
    mlflow.end_run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
