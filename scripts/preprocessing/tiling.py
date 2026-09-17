from kube_jobs import storage, submit_job


submit_job(
    job_name="tiling-lymph-nodes",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=10,
    memory="60Gi",
    gpu="A40",
    shm="60Gi",
    public=False,
    script=[
        "git clone -b feature/prop/preprocessing --single-branch https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0",
        "export HF_TOKEN=...",
        "export NO_PROXY=.cloud.trusted.e-infra.cz,.cluster.local,.rationai-mlflow", 
        "uv run -m preprocessing.tiling +experiment=tile_mmci +data=tiling/mmci memory_per_gpu_worker=null batch_size=4096 embedding_model=<UNI2Encoder | GigaPathEncoder | Virchow2Encoder> +tiling_blocks=[drop_columns, compute_embeddings, save_embeddings] +tiling_blocks.drop_columns.cols=[level, tile, path, mpp_x, mpp_y, tile_extent_x, tile_extent_y] +tiling_blocks.compute_embeddings.embedding_model=<UNI2Encoder | GigaPathEncoder | Virchow2Encoder>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
