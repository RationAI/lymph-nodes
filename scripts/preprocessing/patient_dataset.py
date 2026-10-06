from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-patient-dataset",
    username=...,
    cpu=4,
    memory="4Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv run -m preprocessing.data_exploration.patient_dataset +data=raw/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
