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


def save_hist(path: str | Path, y: NDArray, fps: NDArray, label: str) -> None:
    x = np.linspace(1, 100, 100)
    plt.plot(x, y, label=label)
    plt.plot(x, fps, label="FPS")

    # Set y-axis to logarithmic scale
    plt.yscale("log")

    # Labels and formatting
    plt.xlabel("thresholds")
    plt.title(f"{label} vs FPS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_roc(path: str | Path, tps: NDArray, fps: NDArray) -> None:
    if not tps.sum() or not fps.sum():
        # There is nothing to plot
        return

    tpr = tps[::-1] / tps.sum()
    fpr = fps[::-1] / fps.sum()

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def vec_from_hist(hist: NDArray) -> NDArray:
    return np.cumsum(hist[::-1])


def process_prediction(
    pred_path: str | Path, run_id: str, prefix: str
) -> tuple[NDArray, NDArray, NDArray]:
    rel_path = Path(pred_path).relative_to(f"./data/{run_id}/{prefix}").parent

    tissue_mask_path = Path(
        "./data/gt/tissue_masks",
        rel_path,
        f"{Path(pred_path).stem}.tiff",
    )

    custom_ignore_mask_path = Path(
        "./data/gt/custom_ignore_masks",
        rel_path,
        f"{Path(pred_path).stem}.tiff",
    )

    ignore_mask_path = Path(
        "./data/gt/ignore_masks",
        rel_path,
        f"{Path(pred_path).stem}.tiff",
    )

    gt_path = Path(
        f"./data/gt/{prefix}",
        rel_path,
        f"{Path(pred_path).stem}.tiff",
    )

    tissue_mask = pyvips.Image.new_from_file(tissue_mask_path, page=0)  # MPP=2

    if os.path.exists(ignore_mask_path):
        ignore_mask = pyvips.Image.new_from_file(ignore_mask_path, page=0)  # MPP=2
        tissue_mask = tissue_mask & (~ignore_mask)

    if os.path.exists(custom_ignore_mask_path):
        ignore_mask = pyvips.Image.new_from_file(
            custom_ignore_mask_path, page=0
        )  # MPP=2
        tissue_mask = tissue_mask & (~ignore_mask)

    tissue_mask = tissue_mask.resize(2, kernel="nearest") > 0  # MPP=2 => MPP=
    pred = pyvips.Image.new_from_file(pred_path, page=1)  # MPP=1

    if not os.path.exists(gt_path):
        fps = vec_from_hist(extract_hist(pred & tissue_mask))
        tps = np.zeros(256)
        fns = np.zeros(256)

    else:
        with OpenSlide(gt_path) as slide:
            mpp = slide_resolution(slide, 0)[0]

            if mpp < 1.5:
                level = closest_level(slide, 1)
                scale = 1
            else:
                level = 0
                scale = round(mpp)

        gt = pyvips.Image.new_from_file(pred_path, page=level)
        if scale != 1:
            gt = gt.resize(mpp, kernel="nearest")

        gt = gt > 0
        tissue_mask = gt | tissue_mask

        p = extract_hist(gt)[-1]

        tps = vec_from_hist(extract_hist(pred & gt))
        fps = vec_from_hist(extract_hist(pred & ((~gt) & tissue_mask)))
        fns = p - tps

    roc_path = Path(
        f"./data/{run_id}/roc/{prefix}", rel_path, f"{Path(pred_path).stem}.txt"
    )
    roc_path.parent.mkdir(exist_ok=True, parents=True)

    np.savetxt(
        roc_path,
        np.array([tps, fps, fns]),
        fmt="%.5f",
    )
    return tps, fps, fns


def process_sections(run_id: str, prefix: str) -> None:
    total_tps = np.zeros(256)
    total_fps = np.zeros(256)
    total_fns = np.zeros(256)

    for section in os.listdir(f"./data/{run_id}/{prefix}"):
        section_tps = np.zeros(256)
        section_fps = np.zeros(256)
        section_fns = np.zeros(256)

        paths = Path(f"./data/{run_id}/{prefix}", section).rglob("*.tiff")

        for path in paths:
            tps, fps, fns = process_prediction(path, run_id, prefix)
            section_tps += tps
            section_fps += fps
            section_fns += fns

        np.savetxt(
            Path(f"./data/{run_id}/roc/{prefix}", f"{section}.txt"),
            np.array([section_tps, section_fps, section_fns]),
            fmt="%.5f",
        )

        save_roc(
            Path(f"./data/{run_id}/roc/{prefix}", f"{section}-roc.png"),
            section_tps,
            section_fps,
        )
        save_hist(
            Path(f"./data/{run_id}/roc/{prefix}", f"{section}-hist-tp.png"),
            section_tps,
            section_fps,
            "TPS",
        )
        save_hist(
            Path(f"./data/{run_id}/roc/{prefix}", f"{section}-hist-fn.png"),
            section_fns,
            section_fps,
            "FNS",
        )

        total_tps += section_tps
        total_fps += section_fps
        total_fns += section_fns

    np.savetxt(
        f"./data/{run_id}/roc/{prefix}/total.txt",
        np.array([total_tps, total_fps, total_fns]),
        fmt="%.5f",
    )

    save_roc(f"./data/{run_id}/roc/{prefix}/total_roc.png", total_tps, total_fps)
    save_hist(
        f"./data/{run_id}/roc/{prefix}/total_hist-tp.png", total_tps, total_fps, "TPS"
    )
    save_hist(
        f"./data/{run_id}/roc/{prefix}/total_hist-fn.png", total_fns, total_fps, "FNS"
    )


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

    # Cusom ignore masks
    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/4b2f46aba7ec4ed7a54437181fca5718/artifacts/custom_ignore_masks",
        dst_path="./data/gt",
    )

    # ignore masks
    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/ignore_masks",
        dst_path="./data/gt",
    )

    # ignore masks
    mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/68/10bfc155a303465882aada4928487822/artifacts/tissue_masks",
        dst_path="./data/gt",
    )

    process_items(seg_run_ids, process_item=process_seg_run)
    process_items(cls_run_ids, process_item=process_cls_run)


if __name__ == "__main__":
    main([], ["a2d577f56700457fabacfdda7c4f6b87"])  # pylint: disable=no-value-for-parameter


# 964a7353a2cd42a19db85cbcea45207b
# 1d7a0316f4ef43039009667906095e30
# ead47eb59ad8420c8eab094ec43b6333
# 8411fbb4697e4a55b1dc7d1b69a42aff
# 710bd45e63234f89b6e2c80830417c1e
# 3608dfa17ece4f8ea6dfac8f9bb1fe9f
# 1ba95fe6ab2b4215a9faba764ba4d78a
# 64637a53dbe44a47bc36844c98659b61


# db0b05671f824fe083ca8d884e68ce61
# 8dceedbb49844ccc9f5b4a2f30578d2e
# 5d55e75a0fc446a095cfa8282fac467c
# 6b5be7163a7e42a88a8fe7d4773af6f0
# 6bd8a2b178ca40e4af7df32213c52378
# a2d577f56700457fabacfdda7c4f6b87
# 65a81f3aab9b4ac4813e900153ef308f
# 4fa33159b0db42d4a14eb4a15a0e5010
