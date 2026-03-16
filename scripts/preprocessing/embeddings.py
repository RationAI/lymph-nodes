from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-embeddings",
    username=...,
    cpu=16,
    memory="32Gi",
    gpu="H100",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "export HF_TOKEN=...",
        "uv sync --frozen",
        "uv run -m preprocessing.embeddings +experiment/embeddings=<data_file>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
