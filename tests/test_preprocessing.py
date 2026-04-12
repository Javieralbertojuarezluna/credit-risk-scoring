from pathlib import Path

from src.credit_risk.data.loader import load_raw_data
from src.credit_risk.data.preprocessing import preprocess_credit_data


def test_preprocess_credit_data_creates_target():
    project_root = Path(__file__).resolve().parent.parent
    paths_config = project_root / "configs" / "paths.yaml"
    model_config = project_root / "configs" / "model.yaml"

    df = load_raw_data(paths_config)
    df = preprocess_credit_data(df, model_config)

    assert "target" in df.columns
    assert df["target"].isin([0, 1]).all()
    assert (df["customer_age"] >= 18).all()
