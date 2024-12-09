from random import randint

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from prepro.annotation_masks import generate_annotation_masks
from prepro.data_source import ChainedDataSources, DataSource
from prepro.tiling import tile_dataset
from prepro.tissue_masks import generate_tissue_masks
from prepro.ignore_masks import generate_ignore_masks


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="./configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.metadata.mlflow_uri)
    mlflow.set_experiment(config.metadata.experiment_name)

    active_run = mlflow.start_run(run_name=config.metadata.run_name)

    mlflow.artifacts.download_artifacts(
        artifact_uri=config.metadata.cytokeratin_masks_path, dst_path="./data"
    )

    # DataSources
    ## Infer
    infer_negative_lymph_nodes = DataSource(
        "/mnt/data/Projects/lymph_nodes/dataset2-ihc-2024/negative",
        glob_pattern="*-0.mrxs",
    )
    infer_positive_lymph_nodes = ChainedDataSources(
        [
            DataSource(
                "/mnt/data/Projects/lymph_nodes/dataset2-ihc-2024/positive",
                glob_pattern="*-1.mrxs",
            ),
            DataSource(
                "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
                glob_pattern="*-1.tiff",
            ),
        ]
    )

    ## Test
    test_negative_lymph_nodes = DataSource(
        "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
        glob_pattern=["*_20_SLIDE_[0-9]*-0.tiff", "*_59_SLIDE_[0-9]*-0.tiff"],
    )
    test_positive_lymph_nodes = DataSource(
        "/mnt/data/Projects/lymph_nodes/annotated_ihc_test",
        glob_pattern=["*.mrxs"],
    )
    test_tmas = DataSource(
        "/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry/Cytokeratin_mask_final_scans",
        glob_pattern=["FIN-CK-*.mrxs"],
    )

    ## Train
    train_negative_lymph_nodes = DataSource(
        "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
        glob_pattern="*-0.tiff",
        exclue_pattern=[
            "*_3_SLIDE_[0-9]*-0.tiff",
            "*_34_SLIDE_[0-9]*-0.tiff",
            "*_52_SLIDE_[0-9]*-0.tiff",
            "*_80_SLIDE_[0-9]*-0.tiff",
            "*_144_SLIDE_[0-9]*-0.tiff",
            "*_20_SLIDE_[0-9]*-0.tiff",
            "*_59_SLIDE_[0-9]*-0.tiff",
        ],
    )

    train_tmas = ChainedDataSources(
        [
            DataSource(
                "/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry/Cytokeratin_mask_new_breast_TNBC-TMAS/ckae",
                glob_pattern="*.mrxs",
                exclue_pattern=["TNBC-BF-4-*.mrxs"],
            ),
            DataSource(
                "/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry/Cytokeratin_mask_colorectal_TMAs",
                glob_pattern="DAB-*.mrxs",
            ),
        ]
    )

    ## Val
    val_negative_lymph_nodes = DataSource(
        "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023",
        glob_pattern=[
            "*_3_SLIDE_[0-9]*-0.tiff",
            "*_34_SLIDE_[0-9]*-0.tiff",
            "*_52_SLIDE_[0-9]*-0.tiff",
            "*_80_SLIDE_[0-9]*-0.tiff",
            "*_144_SLIDE_[0-9]*-0.tiff",
        ],
    )
    val_tmas = DataSource(
        "/mnt/data/Projects/Lymph_nodes/MMCI/Immunohistochemistry/Cytokeratin_mask_new_breast_TNBC-TMAS/ckae",
        glob_pattern="TNBC-BF-4-*mrxs",
    )

    # Tissue masks
    generate_tissue_masks(
        slide_paths=ChainedDataSources(
            [
                infer_negative_lymph_nodes,
                infer_positive_lymph_nodes,
                test_negative_lymph_nodes,
                test_positive_lymph_nodes,
                test_tmas,
                train_negative_lymph_nodes,
                train_tmas,
                val_negative_lymph_nodes,
                val_tmas,
            ]
        ),
        mpp=2,
        reference_path=config.metadata.relative_path_prefix,
        dest=config.metadata.tissue_mask_dest,
    )

    # Annotation masks
    generate_annotation_masks(
        slide_paths=ChainedDataSources([test_positive_lymph_nodes]),
        mpp=2,
        reference_path=config.metadata.relative_path_prefix,
        dest=config.metadata.annotation_mask_dest,
    )

    # Ignore masks
    generate_ignore_masks(
        slide_paths=ChainedDataSources([test_tmas, train_tmas, val_tmas]),
        mpp=2,
        reference_path=config.metadata.relative_path_prefix,
        dest=config.metadata.ignore_mask_dest,
    )

    # Tiling
    ## Infer
    tile_dataset(
        [
            {
                "slides": infer_negative_lymph_nodes,
                "source_kind": "lymph_node",
                "slide_metastazis": False,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
            {
                "slides": infer_positive_lymph_nodes,
                "source_kind": "lymph_node",
                "slide_metastazis": True,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
        ],
        dataset_name="Inference",
    )

    ## Test
    tile_dataset(
        [
            {
                "slides": test_negative_lymph_nodes,
                "source_kind": "lymph_node",
                "slide_metastazis": False,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
            {
                "slides": test_positive_lymph_nodes,
                "source_kind": "lymph_node",
                "slide_metastazis": True,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
            {
                "slides": test_tmas,
                "source_kind": "tma",
                "slide_metastazis": True,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
        ],
        dataset_name="Test",
    )

    ## Train
    tile_dataset(
        [
            {
                "slides": train_negative_lymph_nodes,
                "source_kind": "lymph_node",
                "slide_metastazis": False,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
            {
                "slides": train_tmas,
                "source_kind": "tma",
                "slide_metastazis": True,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
        ],
        dataset_name="Training",
    )

    ## Val
    tile_dataset(
        [
            {
                "slides": val_negative_lymph_nodes,
                "source_kind": "lymph_node",
                "slide_metastazis": False,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
            {
                "slides": val_tmas,
                "source_kind": "tma",
                "slide_metastazis": True,
                "desired_mpp": config.metadata.tiling.mpp,
                "tissue_threshold": config.metadata.tiling.tissue_threshold,
                "tile_extent": config.metadata.tiling.tile_extent,
                "stride": config.metadata.tiling.stride,
                "tissue_masks_dir": config.metadata.tissue_mask_dest,
                "cytokeratin_masks_dir": config.metadata.cytokeratin_masks_dest,
                "ignore_mask_dir": config.metadata.ignore_mask_dest,
                "annotation_masks_dir": config.metadata.annotation_mask_dest,
                "realative_path_prefix": config.metadata.relative_path_prefix,
            },
        ],
        dataset_name="Validation",
    )

    active_run.end()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
