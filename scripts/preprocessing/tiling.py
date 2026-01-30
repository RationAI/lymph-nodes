from kube_jobs import storage, submit_job


submit_job(
    job_name="tiling-hdab-lymph-nodes",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="12Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git",
        "cd lymph-nodes",
        "uv sync --frozen",
        "uv run -m preprocessing.tiling +experiment=<experiment_name>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
