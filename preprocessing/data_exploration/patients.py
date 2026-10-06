import pandas as pd


PATIENT_COLUMNS = [
    "case_id",
    "n_slides",
    "n_positive_slides",
    "n_negative_slides",
    "n_tma_control_slides",
    "n_confounding_slides",
]


def build_patients_df(slides_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a slides table into one row per case (patient)."""
    if slides_df.empty:
        return pd.DataFrame(columns=PATIENT_COLUMNS)
    return (
        slides_df.groupby("case_id")
        .agg(
            n_slides=("slide_path", "count"),
            n_positive_slides=("tumor", "sum"),
            n_negative_slides=("tumor", lambda x: (~x).sum()),
            n_tma_control_slides=("has_tma_control", "sum"),
            n_confounding_slides=("has_confounding_structures", "sum"),
        )
        .reset_index()
    )
