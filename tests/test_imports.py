"""Smoke tests for repository scaffold."""

from drifting_models_pytorch import DriftingModelConfig, build_model


def test_build_model_returns_module() -> None:
    """Ensure the scaffold exports a constructible module."""
    model = build_model(DriftingModelConfig(dim=2))
    assert model is not None
