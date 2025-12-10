from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-tissue-masks",
    username="your name",
    cpu=12,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.tissue_masks +experiment=<experiment_name>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
