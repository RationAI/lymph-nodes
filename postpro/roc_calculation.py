import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pyvips
import ray
from numpy.typing import NDArray
from openslide import OpenSlide
from rationai.masks import (
    closest_level,
    process_items,
    slide_resolution,
)
from sklearn.metrics import auc


def extract_hist(image: pyvips.Image) -> NDArray:
    hist = image.hist_find().numpy()

    if len(hist.shape) == 0:
        empty = np.zeros(256)
        empty[0] = hist
        return empty

    return hist[0]


def save_roc(path: str | Path, tpr: NDArray, fpr: NDArray) -> None:
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binned ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def vec_from_hist(hist: NDArray) -> NDArray:
    s = hist.sum()
    return np.roll(s - hist, 1)


def process_prediction(
    pred_path: str | Path, run_id: str, prefix: str
) -> tuple[NDArray, NDArray]:
    print("Processing slide: ", Path(pred_path).stem)

    rel_path = Path(pred_path).relative_to(f"./data/{run_id}/{prefix}").parent

    gt_path = Path(
        f"./data/gt/{prefix}",
        rel_path,
        f"{Path(pred_path).stem}.tiff",
    )

    if os.path.exists(gt_path):
        with OpenSlide(gt_path) as slide:
            mpp = slide_resolution(slide, 0)

        with OpenSlide(pred_path) as slide:
            level = closest_level(slide, mpp[0])

        pred = pyvips.Image.new_from_file(pred_path, page=level + 3)
        gt = pyvips.Image.new_from_file(pred_path, page=3) > 0

        gt_hist = extract_hist(gt)
        n, p = gt_hist[0], gt_hist[-1]

        tps = vec_from_hist(extract_hist(pred * gt))
        fps = vec_from_hist(extract_hist(pred * (~gt)))

        tpr = tps / p if p > 0 else np.zeros(256)
        fpr = fps / n if n > 0 else np.zeros(256)

    else:
        pred = pyvips.Image.new_from_file(pred_path, page=3)
        n = pred.width * pred.height
        fps = vec_from_hist(extract_hist(pred))

        tpr = np.zeros(256)
        fpr = fps / n

    print(tpr.shape, fpr.shape, flush=True)
    np.savetxt(
        Path(f"./data/{run_id}/roc/{prefix}", rel_path, f"{Path(pred_path).stem}.txt"),
        np.array([tpr, fpr]),
        fmt="%.5f",
    )
    return tpr, fpr


def process_sections(run_id: str, prefix: str) -> None:
    total_tprs = np.zeros(256)
    total_fprs = np.zeros(256)

    for section in os.listdir(f"./data/{run_id}/{prefix}"):
        print("Processing section: ", section)

        tprs = np.zeros(256)
        fprs = np.zeros(256)

        paths = Path(f"./data/{run_id}/{prefix}", section).rglob("*.tiff")

        for path in paths:
            tpr, fpr = process_prediction(path, run_id, prefix)
            tprs += tpr
            fprs += fpr

        np.savetxt(
            Path(f"./data/{run_id}/roc/{prefix}", f"{section}.txt"),
            np.array([tprs, fprs]),
            fmt="%.5f",
        )

        save_roc(Path(f"./data/{run_id}/roc/{prefix}", f"{section}.png"), tprs, fprs)

        total_tprs += tprs
        total_fprs += fprs

    np.savetxt(
        f"./data/{run_id}/roc/{prefix}/total_roc.txt",
        np.array([total_tprs, total_fprs]),
        fmt="%.5f",
    )

    save_roc(f"./data/{run_id}/roc/{prefix}/total_roc.png", total_tprs, total_fprs)


@ray.remote
def process_seg_run(run_id: str) -> None:
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"mlflow-artifacts:/68/{run_id}/artifacts/segmentation_masks",
        dst_path=f"./data/{run_id}",
    )

    process_sections(run_id, "segmentation_masks")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(f"./data/{run_id}/roc", artifact_path="roc")


@ray.remote
def process_cls_run(run_id: str) -> None:
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"mlflow-artifacts:/68/{run_id}/artifacts/classifcation_heatmaps",
        dst_path=f"./data/{run_id}",
    )

    process_sections(run_id, "classifcation_heatmaps")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(f"./data/{run_id}/roc", artifact_path="roc")


def main(seg_run_ids: list[str], cls_run_ids: list[str]) -> None:
    mlflow.set_experiment("Lymph Nodes")

    # Lymph node annotation
    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/annotation_masks",
        dst_path="./data/gt",
    )

    result = subprocess.run(
        ["mv", "./data/gt/annotation_masks", "./data/gt/segmentation_masks"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result)

    # Cyto masks
    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/8ad173ea482d4db999bee7686d9dc2d5/artifacts/cytokeratin_masks",
        dst_path="./data/gt",
    )

    result = subprocess.run(
        [
            "cp",
            "-r",
            "./data/gt/cytokeratin_masks/.",
            "./data/gt/segmentation_masks",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result)
    result = subprocess.run(
        ["rm", "-rf", "./data/gt/cytokeratin_masks"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result)

    # Classification GT masks
    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/98594591ef8940eb864e4b60136e58fc/artifacts/cls_gt_masks",
        dst_path="./data/gt",
    )

    result = subprocess.run(
        ["mv", "./data/gt/cls_gt_masks", "./data/gt/classifcation_heatmaps"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result)

    process_items(seg_run_ids, process_item=process_seg_run)
    process_items(cls_run_ids, process_item=process_cls_run)


if __name__ == "__main__":
    main([], ["a2d577f56700457fabacfdda7c4f6b87"])  # pylint: disable=no-value-for-parameter


# 964a7353a2cd42a19db85cbcea45207b
# 1d7a0316f4ef43039009667906095e30
# ead47eb59ad8420c8eab094ec43b6333


# db0b05671f824fe083ca8d884e68ce61
# 8dceedbb49844ccc9f5b4a2f30578d2e
# 5d55e75a0fc446a095cfa8282fac467c
# 6b5be7163a7e42a88a8fe7d4773af6f0
# 6bd8a2b178ca40e4af7df32213c52378
# a2d577f56700457fabacfdda7c4f6b87
