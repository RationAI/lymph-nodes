"""Submit training jobs: one per foundation model x cross-validation fold.

Each job fits on the scenario's training folds, then validates its best checkpoint on
the lymph node evaluation set (data.eval), logged as its own MLflow run. The test set
(data.test) is the final evaluation, run only on request for a chosen checkpoint
(MLflow stores every fit run's checkpoints, see log_model in the MLFlowLogger).

Used by the scenario scripts (cross_tissue.py, tma_epithelium.py):
    python scripts/training/cross_tissue.py                    # all models, all folds
    python scripts/training/cross_tissue.py uni2 virchow2 --folds 0 1
    python scripts/training/cross_tissue.py uni2 --folds 0 --test mlflow-artifacts:/4/<run>/artifacts/<name>/<name>.ckpt
"""

import argparse

from kube_jobs import storage, submit_job


MODELS = ("gigapath", "uni2", "virchow2")  # configs/data/tiled/mmci_<model>.yaml
FOLDS = (0, 1, 2, 3, 4)


def submit_scenario(scenario: str) -> None:
    """Submit the jobs of ``configs/experiment/training/<scenario>.yaml``."""
    parser = argparse.ArgumentParser(description=f"Submit {scenario} training jobs.")
    parser.add_argument("models", nargs="*", choices=MODELS, help="default: all")
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=FOLDS)
    parser.add_argument(
        "--test",
        metavar="CHECKPOINT",
        help="only test this checkpoint (an MLflow URI or path); needs exactly one model and fold",
    )
    args = parser.parse_args()
    models = args.models or MODELS

    if args.test is not None and (len(models) != 1 or len(args.folds) != 1):
        parser.error("--test needs exactly one model and one --folds value: the checkpoint's")

    for model in models:
        for fold in args.folds:
            overrides = f"+data=tiled/mmci_{model} +experiment=training/{scenario} fold={fold}"
            if args.test is None:
                commands = [
                    f"uv run -m lymph_nodes mode=fit {overrides}",
                    # save_top_k=1 leaves exactly one checkpoint.
                    'CKPT=$(find . -name "*.ckpt" | head -1); [ -n "$CKPT" ] || { echo "no checkpoint"; exit 1; }',
                    f"""uv run -m lymph_nodes mode=validate {overrides} "checkpoint='$CKPT'" """,
                ]
            else:
                commands = [f"""uv run -m lymph_nodes mode=test {overrides} "checkpoint='{args.test}'" """]

            submit_job(
                job_name=f"lymph-nodes-{scenario.replace('_', '-')}-{model}-fold-{fold}"
                + ("-test" if args.test is not None else ""),
                username=...,
                image="cerit.io/rationai/base:2.0.6",
                cpu=8,  # data.num_workers=6 loader workers + the trainer
                memory="16Gi",
                gpu="mig-1g.10gb",  # the trained head is a small MLP
                shm="16Gi",
                public=False,
                script=[
                    "git clone https://github.com/RationAI/lymph-nodes.git workdir",
                    "cd workdir",
                    "uv sync --frozen",
                    "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
                    *commands,
                ],
                storage=[storage.secure.DATA, storage.secure.PROJECTS],
            )
