from kube_jobs import storage, submit_job


submit_job(
    job_name="patch-mlp-heatmap-lymph-nodes",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git",
        "cd lymph-nodes",
        "uv sync --frozen",
        (
            "uv run -m lymph_nodes --config-name predict_patch"
            " foundation=<foundation>"  # e.g. prov-gigapath — must match training
            " level=<level>"  # e.g. level1
            " checkpoint=\"'mlflow-artifacts:/68/b9aa8bd2a0c746a582c452c5dc666c12/artifacts/checkpoints/epoch=1-step=74688/checkpoint.ckpt'\""
            " embeddings_uri=mlflow-artifacts:/68/149c1f462e4a4fff9dde6b7696fe89e5/artifacts/embeddings"
            " 'data.predict.include_slides=[FIN-CK-HR2-19-ERA-DAB]'"
        ),
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
