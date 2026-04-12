from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CreditRiskTorchDataset(Dataset):
    """Custom PyTorch dataset for credit risk classification."""

    def __init__(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> None:
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
