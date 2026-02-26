"""Drifting Models implementation in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class DriftingModelConfig:
    """Configuration for generator backbones."""

    dim: int
    hidden_dim: int = 256
    depth: int = 4
    image_size: int | None = None
    channels: int = 1
    latent_dim: int = 128
    model_type: str = "mlp"


class MLPGenerator(nn.Module):
    """Simple MLP generator for vector-valued outputs."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> None:
        """Initialize an MLP generator.

        Args:
            input_dim: Input latent dimension.
            output_dim: Target output feature dimension.
            hidden_dim: Width of hidden layers.
            depth: Number of hidden layers.
        """
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, noise: Tensor) -> Tensor:
        """Map latent noise to data-space outputs.

        Args:
            noise: Latent tensor with shape [batch, input_dim].

        Returns:
            Predicted samples with shape [batch, output_dim].
        """
        return self.net(noise)


class ConvGenerator(nn.Module):
    """Small convolutional generator for image outputs."""

    def __init__(self, latent_dim: int, channels: int, image_size: int, hidden_dim: int) -> None:
        """Initialize a transposed-convolution generator.

        Args:
            latent_dim: Input latent dimension.
            channels: Output image channels.
            image_size: Output side length. Supported values are 32 and 64.
            hidden_dim: Base hidden channels.
        """
        super().__init__()
        if image_size not in {32, 64}:
            raise ValueError("ConvGenerator only supports image_size in {32, 64}")
        mult = 4 if image_size == 32 else 8
        self.image_size = image_size
        self.project = nn.Linear(latent_dim, hidden_dim * mult * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * mult, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: Tensor) -> Tensor:
        """Map latent noise to an image batch.

        Args:
            noise: Latent tensor with shape [batch, latent_dim].

        Returns:
            Image tensor with shape [batch, channels, image_size, image_size].
        """
        projected = self.project(noise)
        mult = 4 if self.image_size == 32 else 8
        x = projected.view(noise.shape[0], -1, 4, 4)
        if x.shape[1] % mult != 0:
            raise RuntimeError("Invalid projected feature dimensions for ConvGenerator")
        return self.net(x)


class TimeConditionedMLP(nn.Module):
    """Time-conditioned MLP used for Rectified Flow."""

    def __init__(self, dim: int, hidden_dim: int, depth: int) -> None:
        """Initialize the velocity network.

        Args:
            dim: Data dimension.
            hidden_dim: Hidden width.
            depth: Number of hidden layers.
        """
        super().__init__()
        self.net = MLPGenerator(input_dim=dim + 1, output_dim=dim, hidden_dim=hidden_dim, depth=depth)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Predict velocity at interpolation point.

        Args:
            x_t: Interpolated data with shape [batch, dim].
            t: Time values in [0, 1] with shape [batch, 1].

        Returns:
            Velocity prediction with shape [batch, dim].
        """
        return self.net(torch.cat((x_t, t), dim=-1))


def build_model(config: DriftingModelConfig) -> nn.Module:
    """Build a generator from a model configuration.

    Args:
        config: Model configuration.

    Returns:
        A generator module compatible with the drifting objective.
    """
    if config.model_type == "mlp":
        return MLPGenerator(
            input_dim=config.latent_dim,
            output_dim=config.dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
    if config.model_type == "conv":
        if config.image_size is None:
            raise ValueError("image_size must be set for model_type='conv'")
        return ConvGenerator(
            latent_dim=config.latent_dim,
            channels=config.channels,
            image_size=config.image_size,
            hidden_dim=config.hidden_dim,
        )
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def pairwise_l2_distance(x: Tensor, y: Tensor) -> Tensor:
    """Compute pairwise L2 distances.

    Args:
        x: Tensor with shape [n_x, d].
        y: Tensor with shape [n_y, d].

    Returns:
        Pairwise distances with shape [n_x, n_y].
    """
    return torch.cdist(x, y, p=2)


def normalized_affinity(logits: Tensor, dual_softmax: bool = True) -> Tensor:
    """Compute normalized affinities using softmax.

    Args:
        logits: Pairwise logits with shape [n_x, n_y].
        dual_softmax: If true, combine row- and column-softmax with geometric mean.

    Returns:
        Affinity matrix with shape [n_x, n_y].
    """
    row = logits.softmax(dim=-1)
    if not dual_softmax:
        return row
    col = logits.softmax(dim=-2)
    return torch.sqrt(row * col + 1e-12)


def compute_drifting_field(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    tau: float,
    dual_softmax: bool = True,
    exclude_self_from_negatives: bool = True,
) -> Tensor:
    """Compute mean-shift drifting field V_{p,q}(x).

    Args:
        x: Generated samples [n_x, d].
        y_pos: Positive/data samples [n_pos, d].
        y_neg: Negative/generated samples [n_neg, d].
        tau: Kernel temperature.
        dual_softmax: Whether to apply extra x-axis normalization from appendix pseudocode.
        exclude_self_from_negatives: Mask diagonal when y_neg is x from the same batch.

    Returns:
        Drifting vectors with shape [n_x, d].
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    dist_pos = pairwise_l2_distance(x, y_pos)
    dist_neg = pairwise_l2_distance(x, y_neg)
    if exclude_self_from_negatives and x.shape[0] == y_neg.shape[0]:
        diag = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(diag, 1e6)

    logits_pos = -dist_pos / tau
    logits_neg = -dist_neg / tau
    logits = torch.cat((logits_pos, logits_neg), dim=1)
    affinity = normalized_affinity(logits, dual_softmax=dual_softmax)
    n_pos = y_pos.shape[0]
    a_pos = affinity[:, :n_pos]
    a_neg = affinity[:, n_pos:]
    w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)
    drift_pos = w_pos @ y_pos
    drift_neg = w_neg @ y_neg
    return drift_pos - drift_neg


def drifting_loss(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor | None = None,
    tau: float = 0.05,
    dual_softmax: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute drifting loss with stop-gradient target.

    Args:
        x: Generated samples [n_x, d] or images [n_x, c, h, w].
        y_pos: Data samples with same non-batch shape as x.
        y_neg: Negative samples. If omitted, uses x.
        tau: Kernel temperature.
        dual_softmax: Whether to apply appendix-style dual normalization.

    Returns:
        Tuple of (loss, drifting field) where drift matches flattened x shape.
    """
    x_flat = x.flatten(1)
    y_pos_flat = y_pos.flatten(1)
    y_neg_flat = x_flat if y_neg is None else y_neg.flatten(1)
    drift = compute_drifting_field(
        x=x_flat,
        y_pos=y_pos_flat,
        y_neg=y_neg_flat,
        tau=tau,
        dual_softmax=dual_softmax,
        exclude_self_from_negatives=y_neg is None,
    )
    target = (x_flat + drift).detach()
    loss = F.mse_loss(x_flat, target)
    return loss, drift


def rectified_flow_loss(model: nn.Module, data: Tensor, noise: Tensor) -> Tensor:
    """Compute the standard Rectified Flow objective.

    Args:
        model: Time-conditioned velocity model with signature model(x_t, t).
        data: Data samples [batch, dim].
        noise: Noise samples [batch, dim].

    Returns:
        Mean squared velocity-matching loss.
    """
    t = torch.rand(data.shape[0], 1, device=data.device, dtype=data.dtype)
    x_t = (1.0 - t) * noise + t * data
    target = data - noise
    pred = model(x_t, t)
    return F.mse_loss(pred, target)
