from kube_jobs import storage, submit_job


# ── Fill in your URIs before submitting ──────────────────────────────────────
EMBEDDINGS_URI = "mlflow-artifacts:/68/<run_id>/artifacts/embeddings"
MODEL_URI = "mlflow-artifacts:/68/<run_id>/artifacts/checkpoints/epoch=1-step=74688"


submit_job(
    job_name="lymph-nodes-inference",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",
    gpu=1,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f'uv run -m postprocessing.run_inference --embeddings-uri "{EMBEDDINGS_URI}" --model-uri "{MODEL_URI}"',
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
