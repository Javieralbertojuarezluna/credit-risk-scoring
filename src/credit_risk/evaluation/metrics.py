from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def get_classification_report(y_true, y_pred) -> dict:
    """Return classification report as dictionary."""
    return classification_report(y_true, y_pred, output_dict=True)


def get_roc_auc(y_true, y_pred_proba) -> float:
    """Compute ROC-AUC score."""
    return roc_auc_score(y_true, y_pred_proba)


def get_confusion_matrix(y_true, y_pred) -> pd.DataFrame:
    """Return confusion matrix as DataFrame."""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Pred_0", "Pred_1"],
    )


def summarize_metrics(y_true, y_pred, y_pred_proba) -> dict:
    """Return key metrics summary."""
    report = classification_report(y_true, y_pred, output_dict=True)

    # Detecta automáticamente la clase positiva
    possible_keys = ["1", 1, "1.0"]

    positive_key = None
    for key in possible_keys:
        if key in report:
            positive_key = key
            break

    if positive_key is None:
        raise ValueError("No se encontró la clase positiva en el classification_report")

    return {
        "precision_default": report[positive_key]["precision"],
        "recall_default": report[positive_key]["recall"],
        "f1_default": report[positive_key]["f1-score"],
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
