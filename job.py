from kube_jobs import Storage, submit_job


submit_job(
    job_name="lymph-resnet18-pure",
    username="pekarj",
    cpu=8,
    memory="24Gi",
    shm="24Gi",
    gpu="A40",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "git checkout feature/segmentation",
        "pdm sync --skip=post_install",
        "pdm fit task=classification kind=pure model/backbone=resnet18",
        # - pdm run preprocess
        # - pdm fit task=classification kind=mix model/backbone=resnet18
        # - pdm predict model/backbone=resnet18 'checkpoint="mlflow-artifacts:/68/0cf65fb29e6448d6b3e513ecac1c985c/artifacts/checkpoints/epoch=26-step=6750/checkpoint.ckpt"'
        # - pdm test model/backbone=resnet18 'checkpoint="mlflow-artifacts:/68/0cf65fb29e6448d6b3e513ecac1c985c/artifacts/checkpoints/epoch=26-step=6750/checkpoint.ckpt"'
        # - pdm run postprocess
        # - pdm run visualize-dataset
    ],
    storage=Storage(mou=True),
)
