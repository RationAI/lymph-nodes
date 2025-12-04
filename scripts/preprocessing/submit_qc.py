from kube_jobs import storage, submit_job

submit_job(
    job_name="🎭 QC Masks H&DAB Lymph Nodes",
    username="tomas-homola",
    cpu=4,
    memory="6Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes.git",
        "cd lymph-nodes",
        "git checkout feature/test-qc",
        "uv sync",
        "uv run python preprocessing/quality-control/qc.py slides_df_uri=/mnt/projects/lymph_nodes/slide_path.csv data_name=FNBrno_selected +lymph_nodes_path=/mnt/projects/lymph_nodes"
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
    image="cerit.io/rationai/base:2.0.6",
)
