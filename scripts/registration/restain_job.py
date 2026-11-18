"""Run restain_score.py as a CPU job on kubas-cluster (the slides are sensitive).

The job clones this branch (push it first), scores the pairs in parallel (one process per
CPU) and logs pairs.csv, patches.csv and the QC sheets to a run of MLflow experiment 4.

    uv run --group run python scripts/registration/restain_job.py --pilot   # 10 pairs
    uv run --group run python scripts/registration/restain_job.py           # all 187

Sizing: synthetic test (4 x 3 mm, 12 patches) 5 s and < 1 GB per pair; real slides to be
measured on the pilot (job_usage.py) and recorded here.
"""

import argparse

from kube_jobs import storage, submit_job


BRANCH = "feature/he-registration"
# Coarse same-section calls (all negative), visually identical outlines, the positives that
# look restained, a mid-Dice positive, and two mirrored pairs (cannot be the same glass).
PILOT = [
    "SNB_IHC_CASE_19_SLIDE_1-0",
    "SNB_IHC_CASE_80_SLIDE_1-0",
    "SNB_IHC_CASE_123_SLIDE_2-0",
    "SNB_IHC_CASE_70_SLIDE_7-0",
    "SNB_IHC_CASE_134_SLIDE_1-1",
    "SNB_IHC_CASE_134_SLIDE_2-1",
    "SNB_IHC_CASE_2_SLIDE_1-1",
    "SNB_IHC_CASE_129_SLIDE_1-1",
    "SNB_IHC_CASE_1_SLIDE_3-1",
    "SNB_IHC_CASE_135_SLIDE_1-1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pilot", action="store_true", help="only the PILOT IHC slides"
    )
    parser.add_argument("--cpu", type=int, default=None, help="default: 4 pilot, 8 all")
    parser.add_argument("--memory", default=None, help="default: 8Gi pilot, 16Gi all")
    args = parser.parse_args()
    cpu = args.cpu or (4 if args.pilot else 8)
    memory = args.memory or ("8Gi" if args.pilot else "16Gi")
    only = (" --only " + " ".join(PILOT)) if args.pilot else ""

    submit_job(
        job_name="ln-he-restain-score" + ("-pilot" if args.pilot else ""),
        image="cerit.io/rationai/base:2.0.6",
        cpu=cpu,
        memory=memory,
        public=False,
        script=[
            f"git clone -b {BRANCH} --single-branch https://github.com/RationAI/lymph-nodes.git workdir",
            "cd workdir",
            "uv sync --frozen",
            "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
            # One process per CPU: no nested BLAS/FFT threads.
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
            "uv run python scripts/registration/restain_score.py"
            " --pairs scripts/registration/pairs.csv --out /tmp/restain"
            f" --workers {cpu} --mlflow{only}",
        ],
        storage=[storage.secure.DATA],
    )


if __name__ == "__main__":
    main()
