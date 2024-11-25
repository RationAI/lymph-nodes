import mlflow
import mlflow.artifacts
import pandas as pd


def load_parquet(uri: str, name: str) -> pd.DataFrame:
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path="./data")

    return pd.read_parquet(f"./data/{name}")


def download_inference_data(
    slides_uri: str, predictions_uri: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mlflow.artifacts.download_artifacts(
        artifact_uri=slides_uri, dst_path="data/inference"
    )
    mlflow.artifacts.download_artifacts(
        artifact_uri=predictions_uri, dst_path="data/inference"
    )

    slides = pd.read_parquet("data/inference/slides.parquet")
    predictions = pd.read_parquet("data/inference/predictions.parquet")

    return slides, predictions
