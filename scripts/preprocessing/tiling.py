from kube_jobs import storage, submit_job


submit_job(
    job_name="tiling-lymph-nodes",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=32,
    memory="64Gi",
    gpu="H100",
    public=False,
    script=[
        "git clone -b feature/prop/preprocessing --single-branch https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "expport RAY_ENABLE_UV_RUN_RUNTIME_ENV=0",
        "uv run -m preprocessing.tiling +experiment=tile_mmci +data=tiling/mmci memory_per_gpu_worker=16_000_000_000 batch_size=4096 embedding_model=GigaPathEncoder",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
