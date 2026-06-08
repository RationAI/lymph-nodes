from kube_jobs import storage, submit_job


submit_job(
    job_name="lymph-nodes-dataset-split",
    username=...,
    cpu=1,
    memory="2Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --single-branch https://github.com/RationAI/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.data_exploration.dataset_split +data=raw/mmci_ihc_2023",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
