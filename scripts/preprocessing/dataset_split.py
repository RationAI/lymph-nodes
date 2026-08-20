from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-dataset-split",
    username="pekarj",
    cpu=1,
    memory="2Gi",
    gpu=None,
    public=False,
    script=[
        "git clone -b feature/prop/preprocessing --single-branch https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv run -m preprocessing.data_exploration.dataset_split +data=split/mmci_ihc_2023",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
