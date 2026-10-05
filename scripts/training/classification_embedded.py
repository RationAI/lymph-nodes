from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-cls-head",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="16Gi",
    gpu="",
    shm="16Gi",
    public=False,
    script=[
        "git clone -b feature/prop/train --single-branch https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "export NO_PROXY=.cloud.trusted.e-infra.cz,.cluster.local,.rationai-mlflow", 
        "uv run lymph_nodes mode=fit +data=tiled/mmci_embed_gigapath fold=0",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
