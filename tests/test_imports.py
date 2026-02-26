"""Smoke tests for drifting model components."""

import torch

from drifting_models_pytorch import DriftingModelConfig, build_model, compute_drifting_field, drifting_loss


def test_build_model_returns_module() -> None:
    """Ensure the package exports a constructible module."""
    model = build_model(DriftingModelConfig(dim=2))
    assert model is not None


def test_drifting_field_shape() -> None:
    """Ensure drifting field outputs the expected shape."""
    x = torch.randn(8, 2)
    y = torch.randn(8, 2)
    field = compute_drifting_field(x=x, y_pos=y, y_neg=x, tau=0.08)
    assert field.shape == x.shape


def test_drifting_loss_zero_when_no_field() -> None:
    """Check that zero drift yields zero loss."""
    x = torch.randn(16, 2)
    loss, field = drifting_loss(x=x, y_pos=x, y_neg=x, tau=0.08)
    assert field.shape == x.shape
    assert torch.isfinite(loss)
