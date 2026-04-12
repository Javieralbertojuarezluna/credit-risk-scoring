from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.credit_risk.models.base import BaseModel


def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class SklearnCreditModel(BaseModel):
    """Logistic regression credit risk model wrapped in a project class."""

    def __init__(
        self,
        model_config_path: str | Path,
        num_cols: list[str],
        cat_cols: list[str],
    ) -> None:
        # Guardar como string para evitar problemas de serialización entre Windows y Linux
        self.model_config_path = str(model_config_path)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline."""
        config = load_yaml_config(self.model_config_path)

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.num_cols),
                ("cat", categorical_pipeline, self.cat_cols),
            ]
        )

        classifier = LogisticRegression(
            max_iter=config["sklearn"]["logistic_regression"]["max_iter"],
            solver=config["sklearn"]["logistic_regression"]["solver"],
            class_weight=config["sklearn"]["logistic_regression"]["class_weight"],
        )

        return Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("classifier", classifier),
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the sklearn pipeline."""
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame):
        """Generate class predictions."""
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Generate probability predictions."""
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet.")
        return self.pipeline.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Save trained pipeline to disk."""
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> SklearnCreditModel:
        """Load trained pipeline from disk."""
        path = Path(path)

        with path.open("rb") as file:
            model = joblib.load(file)

        return model
