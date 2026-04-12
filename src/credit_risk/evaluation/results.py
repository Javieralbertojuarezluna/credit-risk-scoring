from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelMetrics:
    model_name: str
    precision_default: float
    recall_default: float
    f1_default: float
    roc_auc: float
    notes: str = ""
