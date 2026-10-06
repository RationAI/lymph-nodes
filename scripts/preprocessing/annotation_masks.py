from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-annotation-masks",
    username=...,
    cpu=4,
    memory="4Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.annotation_masks +data=annotations/... annotation_dir=... artifact_path=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
