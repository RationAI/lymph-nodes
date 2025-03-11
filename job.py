from kube_jobs import Storage, submit_job


submit_job(
    job_name="lymph - convnext",
    username="pekarj",
    cpu=10,
    memory="20Gi",
    gpu="H100",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "git checkout feature/segmentation",
        "pdm sync --skip=post_install",
        "pdm fit task=segmentation kind=mix model/backbone=convnext",
    ],
    storage=Storage(mou=True),
)
