from __future__ import annotations

from pathlib import Path

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
        return self.network(x)


def train_torch_model(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    num_epochs: int = 20,
) -> list[float]:
    """Train a PyTorch model and return training losses."""
    train_losses: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    return train_losses


def predict_torch_model(
    model: nn.Module, data_loader
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate logits, probabilities and predictions from a trained model."""
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch).squeeze(1)
            all_logits.append(outputs)
            all_labels.append(y_batch)

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    y_pred = (probs >= 0.5).float()

    return y_true, probs, y_pred


def save_torch_model(model: nn.Module, path: str | Path) -> None:
    """Save PyTorch model weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_model(model: nn.Module, path: str | Path) -> nn.Module:
    """Load PyTorch model weights into an existing model instance."""
    path = Path(path)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
