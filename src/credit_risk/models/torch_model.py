from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CreditRiskMLP(nn.Module):
    """MLP model for credit risk classification."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits"""
        return self.network(x)


def get_torch_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_torch_model(
    model: nn.Module,
    train_loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0001,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list[float]:
    """Train a PyTorch model with early stopping and return training losses."""
    if device is None:
        device = get_torch_device()

    model.to(device)

    train_losses: list[float] = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch).squeeze(1)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        improved = avg_loss < best_loss - min_delta

        if improved:
            best_loss = avg_loss
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"- Loss: {avg_loss:.4f} "
                f"- Best Loss: {best_loss:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    return train_losses


def predict_torch_model(
    model: nn.Module,
    data_loader: Any,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate labels, probabilities and predictions from a trained model."""
    if device is None:
        device = get_torch_device()

    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch).squeeze(1)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    y_pred = (probs >= threshold).float()

    return y_true, probs, y_pred


def save_torch_model(model: nn.Module, path: str | Path) -> None:
    """Save PyTorch model weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_model(
    model: nn.Module,
    path: str | Path,
    device: torch.device | None = None,
) -> nn.Module:
    """Load PyTorch model weights into an existing model instance."""
    if device is None:
        device = get_torch_device()

    path = Path(path)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model
