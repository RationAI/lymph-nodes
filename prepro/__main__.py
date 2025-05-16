from collections.abc import Iterator
from dataclasses import dataclass
from random import randint
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

# from prepro.annotation_masks import generate_annotation_masks
# from prepro.color_separation import generate_color_separation_masks
# from prepro.cyto_ignore_masks import generate_cyto_ignore_masks
from prepro.data_source import ChainedDataSources, DataSource

# from prepro.ignore_masks import generate_ignore_masks
from prepro.tiling import Args, tile_dataset


# from prepro.tissue_masks import generate_tissue_masks


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

    print("Donwloading cytokeratin masks")
    mlflow.artifacts.download_artifacts(
        artifact_uri=config.metadata.cytokeratin_masks_path, dst_path="./data"
    )

    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/tissue_masks",
        dst_path="./data",
    )

    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/ignore_masks",
        dst_path="./data",
    )

    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/annotation_masks",
        dst_path="./data",
    )

    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/cyto_ignore_masks",
        dst_path="./data",
    )

    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/color_separation_masks",
        dst_path="./data",
    )

    print("Prepare datasources")

    datasets = {
        "lymhps-2023": {
            "positive-train": Dataset(
                datasource=DataSource(
                    "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
                    glob_pattern=[
                        "*_1_SLIDE_3-1.mrxs",
                        "*_5_SLIDE_2-1.mrxs",
                        "*_7_SLIDE_2-1.mrxs",
                        "*_8_SLIDE_1-1.mrxs",
                        "*_105_SLIDE_1-1.mrxs",
                    ],
                ),
                source_kind="lymph_node",
                slide_metastazis=True,
            ),
            "positive-val": Dataset(
                datasource=DataSource(
                    "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
                    glob_pattern=[
                        "*_2_SLIDE_1-1.mrxs",
                        "*_3_SLIDE_2-1.mrxs",
                    ],
                ),
                source_kind="lymph_node",
                slide_metastazis=True,
            ),
            "positive-infer": Dataset(
                datasource=DataSource(
                    "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
                    glob_pattern="*-1.mrxs",
                    exclue_pattern=[
                        "*_1_SLIDE_3-1.mrxs",
                        "*_2_SLIDE_1-1.mrxs",
                        "*_3_SLIDE_2-1.mrxs",
                        "*_5_SLIDE_2-1.mrxs",
                        "*_7_SLIDE_2-1.mrxs",
                        "*_8_SLIDE_1-1.mrxs",
                        "*_105_SLIDE_1-1.mrxs",
                    ],
                ),
                source_kind="lymph_node",
                slide_metastazis=True,
            ),
            "negative-test": Dataset(
                datasource=DataSource(
                    "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
                    glob_pattern=[
                        "*_1_SLIDE_[0-9]*-0.mrxs",
                        "*_5_SLIDE_[0-9]*-0.mrxs",
                        "*_7_SLIDE_[0-9]*-0.mrxs",
                        "*_8_SLIDE_[0-9]*-0.mrxs",
                        "*_105_SLIDE_[0-9]*-0.mrxs",
                    ],
                ),
                source_kind="lymph_node",
                slide_metastazis=False,
            ),
            "negative-train": Dataset(
                datasource=DataSource(
                    "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
                    glob_pattern="*-0.mrxs",
                    exclue_pattern=[
                        "*_1_SLIDE_[0-9]*-0.mrxs",  # test
                        "*_2_SLIDE_[0-9]*-0.mrxs",  # val
                        "*_3_SLIDE_[0-9]*-0.mrxs",  # val
                        "*_5_SLIDE_[0-9]*-0.mrxs",  # test
                        "*_7_SLIDE_[0-9]*-0.mrxs",  # test
                        "*_8_SLIDE_[0-9]*-0.mrxs",  # test
                        "*_52_SLIDE_[0-9]*-0.mrxs",  # val
                        "*_105_SLIDE_[0-9]*-0.mrxs",  # test
                        # Faulty files
                        "SNB_IHC_CASE_81_SLIDE_1-0.mrxs",
                        "SNB_IHC_CASE_82_SLIDE_1-0.mrxs",
                    ],
                ),
                source_kind="lymph_node",
                slide_metastazis=False,
            ),
            "negative-val": Dataset(
                datasource=DataSource(
                    "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
                    glob_pattern=[
                        "*_2_SLIDE_[0-9]*-0.mrxs",
                        "*_3_SLIDE_[0-9]*-0.mrxs",
                        "*_52_SLIDE_[0-9]*-0.mrxs",
                    ],
                ),
                source_kind="lymph_node",
                slide_metastazis=False,
            ),
        },
        "positive-lymph-nodes": Dataset(
            datasource=DataSource(
                "/mnt/data/Projects/lymph_nodes/annotated_ihc_test",
                glob_pattern=["*.mrxs"],
            ),
            source_kind="lymph_node",
            slide_metastazis=True,
        ),
        # "lymphs-2024": {
        #     "positive": Dataset(
        #         datasource=DataSource(
        #             "/mnt/data/Projects/lymph_nodes/dataset2-ihc-2024/positive",
        #             glob_pattern="*-1.mrxs",
        #         ),
        #         source_kind="lymph_node",
        #         slide_metastazis=True,
        #     ),
        #     "negative": Dataset(
        #         datasource=DataSource(
        #             "/mnt/data/Projects/lymph_nodes/dataset2-ihc-2024/negative",
        #             glob_pattern="*-0.mrxs",
        #         ),
        #         source_kind="lymph_node",
        #         slide_metastazis=False,
        #     ),
        # },
        "tmas": {
            "test": Dataset(
                datasource=DataSource(
                    "/mnt/data/Projects/lymph_nodes/Cytokeratin_mask_final_scans",
                    glob_pattern=["FIN-CK-*.mrxs"],
                ),
                source_kind="tma",
                slide_metastazis=True,
            ),
            "train": Dataset(
                datasource=ChainedDataSources(
                    [
                        DataSource(
                            "/mnt/data/Projects/lymph_nodes/Cytokeratin_mask_new_breast_TNBC-TMAS/ckae",
                            glob_pattern="*.mrxs",
                            exclue_pattern=["TNBC-BF-4-*.mrxs"],
                        ),
                        DataSource(
                            "/mnt/data/Projects/lymph_nodes/Cytokeratin_mask_colorectal_TMAs",
                            glob_pattern="DAB-*.mrxs",
                            exclue_pattern=["DAB-CK-KOS04.mrxs"],
                        ),
                    ]
                ),
                source_kind="tma",
                slide_metastazis=True,
            ),
            "val": Dataset(
                datasource=ChainedDataSources(
                    [
                        DataSource(
                            "/mnt/data/Projects/lymph_nodes/Cytokeratin_mask_new_breast_TNBC-TMAS/ckae",
                            glob_pattern="TNBC-BF-4-*mrxs",
                        ),
                        DataSource(
                            "/mnt/data/Projects/lymph_nodes/Cytokeratin_mask_colorectal_TMAs",
                            glob_pattern="DAB-CK-KOS04.mrxs",
                        ),
                    ]
                ),
                source_kind="tma",
                slide_metastazis=True,
            ),
        },
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
    # print("Generating tissue masks")
    # generate_tissue_masks(
    #     slide_paths=ChainedDataSources(list(map_datasets(datasets))),
    #     mpp=2,
    #     reference_path=config.metadata.relative_path_prefix,
    #     dest=config.metadata.tissue_mask_dest,
    # )

    # Annotation masks
    # print("Generating annotation masks")
    # generate_annotation_masks(
    #     slide_paths=ChainedDataSources(
    #         [
    #             datasets["lymhps-2023"]["positive-train"].datasource,
    #             datasets["lymhps-2023"]["positive-val"].datasource,
    #             datasets["positive-lymph-nodes"].datasource,
    #         ]
    #     ),
    #     mpp=2,
    #     reference_path=config.metadata.relative_path_prefix,
    #     dest=config.metadata.annotation_mask_dest,
    # )

    # Ignore masks
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

    # CytoIgnore masks
    # print("Generating cyto ignore masks")
    # generate_cyto_ignore_masks(
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

    # Color separation masks
    # print("Generating color separation masks")
    # generate_color_separation_masks(
    #     slide_paths=ChainedDataSources(list(map_datasets(datasets))),
    #     mpp=2,
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
            "cyto_ignore_mask_dir": config.metadata.cyto_ignore_mask_dest,
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
