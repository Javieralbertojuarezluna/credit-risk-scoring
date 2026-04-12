from pathlib import Path

from src.credit_risk.data.loader import load_raw_data
from src.credit_risk.data.preprocessing import preprocess_credit_data
from src.credit_risk.features.engineering import build_features


def test_build_features_adds_expected_columns():
    project_root = Path(__file__).resolve().parent.parent
    paths_config = project_root / "configs" / "paths.yaml"
    model_config = project_root / "configs" / "model.yaml"

    df = load_raw_data(paths_config)
    df = preprocess_credit_data(df, model_config)
    df = build_features(df)

    expected_cols = {
        "debt_income_ratio",
        "credit_maturity",
        "employment_years",
        "age_group",
    }

    assert expected_cols.issubset(df.columns)
