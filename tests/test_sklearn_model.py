from pathlib import Path

from sklearn.model_selection import train_test_split

from src.credit_risk.data.loader import load_raw_data
from src.credit_risk.data.preprocessing import preprocess_credit_data
from src.credit_risk.features.engineering import build_features
from src.credit_risk.models.sklearn_model import SklearnCreditModel


def test_sklearn_model_trains_and_predicts():
    project_root = Path(__file__).resolve().parent.parent
    paths_config = project_root / "configs" / "paths.yaml"
    model_config = project_root / "configs" / "model.yaml"

    df = load_raw_data(paths_config)
    df = preprocess_credit_data(df, model_config)
    df = build_features(df)

    X = df.drop(columns=["Current_loan_status", "target"])
    y = df["target"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = SklearnCreditModel(
        model_config_path=model_config,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    model.train(X_train, y_train)

    preds = model.predict(X_test)

    assert len(preds) == len(X_test)
