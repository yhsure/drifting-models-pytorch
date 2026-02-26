"""Core module placeholder for drifting models in PyTorch."""

from dataclasses import dataclass

import torch


@dataclass
class DriftingModelConfig:
    """Configuration for drifting model experiments."""

    dim: int
    hidden_dim: int = 256
    depth: int = 4


class _DriftingModel(torch.nn.Module):
    """Placeholder model implementation."""

    def __init__(self, config: DriftingModelConfig) -> None:
        """Initialize the placeholder module.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass placeholder.

        Args:
            noise: Input latent noise tensor.

        Returns:
            Output tensor.
        """
        raise NotImplementedError("Model implementation has not been added yet.")


def build_model(config: DriftingModelConfig) -> torch.nn.Module:
    """Build a drifting model from config.

    Args:
        config: Model configuration.

    Returns:
        A torch module placeholder for future implementation.
    """
    return _DriftingModel(config)
