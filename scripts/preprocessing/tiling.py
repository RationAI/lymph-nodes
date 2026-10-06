from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-tiling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=6,
    memory="40Gi",
    gpu="A40",
    shm="40Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0",
        "export HF_TOKEN=...",
        "uv run -m preprocessing.tiling +experiment=tiling/... +data=tiling/... memory_per_gpu_worker=null batch_size=4096 embedding_model=<UNI2Encoder | GigaPathEncoder | Virchow2Encoder>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
