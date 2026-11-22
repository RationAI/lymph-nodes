"""How much of a pasted cell does each cut keep (2026-11-21, HANDOFF.md step 5e a)?

The halo-on ablation (fit 7f76bbc1) found real single tumour cells about as often as
the old blending, and twice as often as the clean one (444f1214). Without the halo,
CellPasting blends in only the cut's cell: each pixel is weighted by its total optical
density between the cut's 30th and 70th percentile (augmentations._cell_alpha), and only
the pixels weighted above 0.5 are labelled. Total density is mostly haematoxylin, so
the clean cut may drop a cell's paler cytokeratin-stained cytoplasm (what a faint ITC
shows) and shrink its label. The same donor cells, cut as the scenario's training cuts
them (radii, donors, counterstain matched to a recipient tile), are compared for three
cuts: ``halo`` (the old cut, the whole disk), ``clean`` (halo false) and ``stain``
(halo false, keep_stain):

- label: the labelled pixels (weight above 0.5), relative to the halo's disk, and in
  cells of the target grid (area, and the largest cell's coverage at a random offset);
- stained kept: the share of the disk's DAB-stained pixels (above _STAINED) labelled,
  split into the nucleus (haematoxylin above the disk's 70th percentile) and the rest
  (the stained cytoplasm);
- DAB kept: the disk's DAB density as blended in (weighted by the cut's weights),
  relative to the halo's;
- unstained kept: the share of the disk's unstained pixels still labelled (the halo).

Fading (cell_fade_p) comes after the cut and doesn't change its weights. Run where the
data is, as a job (prints the tables, logs them to MLflow, experiment of the config):

    uv run --group run python scripts/analysis/cut_coverage.py --submit [--cells 2000]

In the job (or anywhere the data is), from the repository root (the project isn't an
installed package: ``lymph_nodes`` imports from the root):

    PYTHONPATH=. uv run python scripts/analysis/cut_coverage.py [--cells 2000] [HYDRA OVERRIDES...]
"""

from __future__ import annotations

import argparse
import copy
import shlex
import subprocess
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OVERRIDES = [
    "+data=segmentation/mmci_virchow2",
    "+experiment=training/segmentation_cross_tissue_es_dfp_table_control",
    "fold=0",
]
CUTS = {
    "halo": {"halo": True, "keep_stain": False},
    "clean": {"halo": False, "keep_stain": False},
    "stain": {"halo": False, "keep_stain": True},
}


def submit(args: argparse.Namespace) -> None:
    from kube_jobs import storage, submit_job
    from kube_jobs.constants import DEFAULT_DOCKER_IMAGE

    command = shlex.join(
        [
            "uv", "run", "python", "scripts/analysis/cut_coverage.py",
            "--cells", str(args.cells), "--seed", str(args.seed),
            *args.overrides,
        ]
    )  # fmt: skip
    submit_job(
        job_name=args.name,
        image=DEFAULT_DOCKER_IMAGE,
        # Reads ~2 tiles per cell from the slides; the dataset's index as a fit's.
        cpu=4,
        memory="16Gi",
        gpu=None,
        public=False,
        script=[
            "git clone -b feature/prop/train --single-branch https://github.com/RationAI/lymph-nodes.git workdir",
            "cd workdir",
            "uv sync --frozen",
            "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
            # lymph_nodes imports from the repository root (not an installed package).
            "export PYTHONPATH=$PWD",
            command,
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


def placed_coverage(
    label: np.ndarray, cell_px: float, rng: np.random.Generator
) -> float:
    """The largest target-grid cell's coverage by ``label`` at a random offset."""
    from lymph_nodes.data.augmentations import pixels_to_cells

    step = round(cell_px)
    cells = int(np.ceil(label.shape[0] / step)) + 1
    canvas = np.zeros((cells * step, cells * step), dtype=bool)
    y, x = (int(v) for v in rng.integers(0, step, size=2))
    canvas[y : y + label.shape[0], x : x + label.shape[1]] = label
    return float(pixels_to_cells(canvas, cells).max())


def measure(
    cutters: dict[str, object],
    window: np.ndarray,
    alpha: np.ndarray,
    recipient: np.ndarray,
    recipient_tissue: np.ndarray,
    cell_px: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    from lymph_nodes.data.augmentations import _STAINED, DAB, H

    disk = alpha > 0.5
    row: dict[str, float] = {"disk_px": float(disk.sum())}
    weights = {}
    blended = None
    for name, cutter in cutters.items():
        # harmonise is the same for every cut (same inputs), so their windows agree.
        blended, weights[name] = cutter._blend(  # type: ignore[attr-defined]
            window.copy(), alpha.copy(), recipient, recipient_tissue
        )
    assert blended is not None
    dab = np.clip(blended[..., DAB], 0, None)
    stained = disk & (blended[..., DAB] > _STAINED)
    nucleus = disk & (blended[..., H] > np.percentile(blended[..., H][disk], 70))
    cytoplasm = stained & ~nucleus
    unstained = disk & ~stained
    row["stained_share"] = stained.sum() / max(disk.sum(), 1)
    for name, weight in weights.items():
        label = weight > 0.5
        row[f"{name}/label_vs_disk"] = label.sum() / max(disk.sum(), 1)
        row[f"{name}/label_cells"] = label.sum() / cell_px**2
        row[f"{name}/max_cell_coverage"] = placed_coverage(label, cell_px, rng)
        for part, mask in (
            ("stained_kept", stained),
            ("nucleus_kept", nucleus),
            ("cytoplasm_stained_kept", cytoplasm),
            ("unstained_kept", unstained),
        ):
            row[f"{name}/{part}"] = (
                (label & mask).sum() / mask.sum() if mask.sum() else np.nan
            )
        row[f"{name}/dab_kept"] = float((dab * weight).sum()) / max(
            float((dab * weights["halo"]).sum()), 1e-9
        )
    return row


def run(args: argparse.Namespace) -> None:
    import mlflow
    import pandas as pd
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from lymph_nodes.data.augmentations import (
        H,
        _scale_counterstain,
        cellular_mask,
        rgb_to_hed,
        tissue_mask,
    )
    from lymph_nodes.resolvers import register_resolvers

    # The configs use custom resolvers (other_fold_uris, ...); __main__ registers them.
    register_resolvers()
    with initialize_config_dir(str(REPO / "configs"), version_base=None):
        config = compose("lymph_nodes", overrides=args.overrides)
    dataset = instantiate(config.data.train)
    pasting = dataset.cell_pasting
    assert pasting is not None, "the scenario pastes no cells"
    cutters = {}
    for name, settings in CUTS.items():
        cutter = copy.copy(pasting)
        for key, value in settings.items():
            setattr(cutter, key, value)
        cutters[name] = cutter
    index = dataset._get_batch_index()
    recipients = np.flatnonzero(dataset._recipients)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | str]] = []
    while len(rows) < args.cells:
        donor_position = dataset._draw_donor(rng)
        slide = int(index.slide_of[donor_position])
        donor_rgb = dataset._read_position(donor_position)
        mpp = float(dataset._mpp[slide])
        cut = pasting._cut_cell(
            rgb_to_hed(donor_rgb), tissue_mask(donor_rgb), mpp, rng, cluster=False
        )
        if cut is None:
            continue
        window, alpha = cut
        recipient_rgb = dataset._read_position(int(rng.choice(recipients)))
        recipient = rgb_to_hed(recipient_rgb)
        recipient_tissue = tissue_mask(recipient_rgb) & cellular_mask(
            recipient, pasting.placement_density
        )
        if recipient_tissue.any():  # the counterstain matched as CellPasting does
            reference = np.percentile(recipient[..., H][recipient_tissue], 90)
            level = np.percentile(window[..., H][alpha > 0.5], 90)
            factor = float(np.clip(reference / max(level, 1e-6), 0.6, 1.5))
            if pasting.match == "od":
                window = _scale_counterstain(window, factor)
            else:
                window[..., H] *= factor
        cell_px = recipient_rgb.shape[0] / dataset.grid
        cell = measure(
            cutters, window, alpha, recipient, recipient_tissue, cell_px, rng
        )
        radius_um = float(np.sqrt(cell["disk_px"] / np.pi)) * mpp
        rows.append({**cell, "radius_um": radius_um, "donor": str(index.names[slide])})
    table = pd.DataFrame(rows)
    summary = (
        table.drop(columns=["donor"])
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["mean", "25%", "50%", "75%"]]
    )
    pd.set_option("display.width", 160)
    print(f"{len(table)} cells, {table['donor'].nunique()} donor slides")
    print(summary.round(3).to_string())
    mlflow.set_experiment(config.metadata.experiment_name)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout.strip()
    with mlflow.start_run(run_name="cut coverage"):
        mlflow.log_params(
            {
                "scenario": "cut coverage (halo / clean / stain cuts)",
                "overrides": " ".join(args.overrides),
                "cells": len(table),
                "seed": args.seed,
                "commit": commit,
            }
        )
        mlflow.log_metrics(
            {
                f"cut/{name}/{stat}".replace("%", "pct"): float(value)
                for name, values in summary.iterrows()
                for stat, value in values.items()
                if np.isfinite(value)
            }
        )
        mlflow.log_text(table.to_csv(index=False), "cut_coverage/cells.csv")
        mlflow.log_text(summary.to_string(), "cut_coverage/summary.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("overrides", nargs="*", default=DEFAULT_OVERRIDES)
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--submit", action="store_true", help="run it as a job")
    parser.add_argument(
        "--name",
        default="lymph-nodes-cut-coverage",
        help="the job's name (a new one for a rerun: a job's name stays taken until deleted)",
    )
    args = parser.parse_args()
    args.overrides = args.overrides or DEFAULT_OVERRIDES
    if args.submit:
        submit(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
