from kube_jobs import Storage, submit_job


submit_job(
    job_name="lymph-nodes-evaluation",
    image="cerit.io/rationai/base:2.0.2-cuda",
    username="pekarj",
    cpu=32,
    memory="50Gi",
    shm="50Gi",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "git checkout feature/segmentation",
        "pdm sync --skip=post_install",
        "pdm run python postpro/evaluation.py",
    ],
    storage=Storage(mou=True),
)
