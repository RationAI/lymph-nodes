from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-tissue-masks",
    username=...,
    cpu=12,
    memory="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.tissue_masks +data=raw/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
