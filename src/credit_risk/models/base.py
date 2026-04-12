from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all project models."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Generate class predictions."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Generate probability predictions"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the trained model to disk"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a trained model from disk"""
        raise NotImplementedError
