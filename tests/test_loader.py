from pathlib import Path

from src.credit_risk.data.loader import load_raw_data


def test_load_raw_data_returns_dataframe():
    project_root = Path(__file__).resolve().parent.parent
    paths_config = project_root / "configs" / "paths.yaml"

    df = load_raw_data(paths_config)

    assert df is not None
    assert not df.empty
