from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn


def set_tracking_uri(tracking_dir: str | Path = "mlruns") -> None:
    """Set a consistent local MLflow tracking URI."""
    tracking_path = Path(tracking_dir).resolve()
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_path.as_uri())


def set_mlflow_experiment(experiment_name: str) -> None:
    """Set the active MLflow experiment."""
    mlflow.set_experiment(experiment_name)


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict[str, float]) -> None:
    """Log metrics to MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_sklearn_model(model: Any, model_name: str = "sklearn_model") -> None:
    """Log a scikit-learn model to MLflow."""
    mlflow.sklearn.log_model(sk_model=model, name=model_name)


def log_torch_model_artifact(
    model_path: str | Path,
    artifact_path: str = "torch_model",
) -> None:
    """Log a saved model file as an MLflow artifact."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    mlflow.log_artifact(str(model_path), artifact_path=artifact_path)


def start_run(run_name: str | None = None):
    """Start an MLflow run."""
    return mlflow.start_run(run_name=run_name)
