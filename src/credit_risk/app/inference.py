from __future__ import annotations

import pandas as pd

from src.credit_risk.features.engineering import build_features


def prepare_input_data(input_data: dict) -> pd.DataFrame:
    """Convert user input dictionary to model-ready dataFrame."""
    df = pd.DataFrame([input_data])
    df = build_features(df)
    return df
