from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-slide-dataset",
    username="your name",
    cpu=2,
    memory="2Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.slide_dataset +experiment=<experiment_name>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
