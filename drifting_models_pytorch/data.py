"""Data utilities for drifting model experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _standardize_points(points: Tensor) -> Tensor:
    """Standardize point cloud to zero mean and unit variance.

    Args:
        points: Input point tensor with shape [n, d].

    Returns:
        Standardized points with shape [n, d].
    """
    centered = points - points.mean(dim=0, keepdim=True)
    return centered / centered.std(dim=0, keepdim=True).clamp_min(1e-6)


def sample_checkerboard(
    batch_size: int,
    device: torch.device,
    normalize: bool = False,
    noise: float = 0,
) -> Tensor:
    """Sample a canonical 2D checkerboard distribution.

    This matches the commonly used checkerboard variant where points are sampled
    from parity-matched cells and then scaled to roughly ``[-1, 1]``.

    Args:
        batch_size: Number of samples.
        device: Target device.
        normalize: Whether to standardize coordinates.
        noise: Optional isotropic Gaussian noise magnitude.

    Returns:
        A tensor of shape [batch_size, 2].
    """
    parity = torch.randint(0, 2, (batch_size,), device=device)
    i = torch.randint(0, 2, (batch_size,), device=device) * 2 + parity
    j = torch.randint(0, 2, (batch_size,), device=device) * 2 + parity
    u = torch.rand(batch_size, device=device)
    v = torch.rand(batch_size, device=device)
    points = torch.stack((i.to(torch.float32) + u, j.to(torch.float32) + v), dim=1) - 2.0
    points = points * 0.5
    if noise > 0.0:
        points = points + noise * torch.randn_like(points)
    if normalize:
        return _standardize_points(points)
    return points


def sample_swissroll(
    batch_size: int,
    device: torch.device,
    normalize: bool = False,
    noise: float = 0.03,
) -> Tensor:
    """Sample a canonical 2D swiss-roll style distribution.

    Args:
        batch_size: Number of samples.
        device: Target device.
        normalize: Whether to standardize coordinates.
        noise: Optional isotropic Gaussian noise magnitude.

    Returns:
        A tensor of shape [batch_size, 2].
    """
    t = 0.5 * math.pi + 4.0 * math.pi * torch.rand(batch_size, device=device)
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    points = torch.stack((x, y), dim=-1)
    points = points / points.abs().max().clamp_min(1e-8)
    if noise > 0.0:
        points = points + noise * torch.randn_like(points)
    if normalize:
        return _standardize_points(points)
    return points


def get_toy_batch(dataset: str, batch_size: int, device: torch.device, noise: float = 0.0) -> Tensor:
    """Sample a toy dataset batch.

    Args:
        dataset: Name of toy dataset, one of {"checkerboard", "swissroll"}.
        batch_size: Number of samples.
        device: Target device.
        noise: Isotropic Gaussian noise magnitude added to data samples.

    Returns:
        A tensor of shape [batch_size, 2].
    """
    if dataset == "checkerboard":
        return sample_checkerboard(batch_size=batch_size, device=device, noise=noise)
    if dataset == "swissroll":
        return sample_swissroll(batch_size=batch_size, device=device, noise=noise)
    raise ValueError(f"Unsupported toy dataset: {dataset}")


def build_cifar10_loader(batch_size: int, root: str = "./data", train: bool = True) -> DataLoader:
    """Create a CIFAR-10 dataloader.

    Args:
        batch_size: Batch size.
        root: Dataset root directory.
        train: Whether to use train split.

    Returns:
        A dataloader yielding normalized images in [-1, 1].
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=2, pin_memory=True)
