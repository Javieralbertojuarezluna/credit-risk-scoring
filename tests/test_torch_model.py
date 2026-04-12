import torch

from src.credit_risk.models.torch_model import CreditRiskMLP


def test_torch_model_forward_shape():
    input_dim = 10
    model = CreditRiskMLP(input_dim=input_dim)

    X_dummy = torch.randn(32, input_dim)
    output = model(X_dummy)

    assert output.shape == (32, 1)
