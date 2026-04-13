from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-shard-embeddings",
    username=...,
    cpu=4,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m preprocessing.shard_embeddings "
        "+preprocessing=shard_embeddings "
        "tiles_uri=<tiles_uri_or_path> "
        "slides_uri=<slides_uri_or_path>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
