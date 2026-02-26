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


def sample_checkerboard(batch_size: int, device: torch.device, normalize: bool = True) -> Tensor:
    """Sample a canonical 2D checkerboard distribution.

    The support is an alternating 4x4 grid over ``[-2, 2) x [-2, 2)``
    with unit-width cells. This avoids degenerate strips and matches the
    usual checkerboard benchmark used in generative modeling.

    Args:
        batch_size: Number of samples.
        device: Target device.
        normalize: Whether to standardize coordinates.

    Returns:
        A tensor of shape [batch_size, 2].
    """
    cell_indices = torch.tensor(
        [
            [0, 0],
            [0, 2],
            [1, 1],
            [1, 3],
            [2, 0],
            [2, 2],
            [3, 1],
            [3, 3],
        ],
        device=device,
        dtype=torch.long,
    )
    picked = cell_indices[torch.randint(0, cell_indices.shape[0], (batch_size,), device=device)]
    offsets = torch.rand(batch_size, 2, device=device)
    points = -2.0 + picked.to(dtype=torch.float32) + offsets
    if normalize:
        return _standardize_points(points)
    return points


def sample_swissroll(batch_size: int, device: torch.device, normalize: bool = True) -> Tensor:
    """Sample a canonical 2D swiss-roll style distribution.

    Args:
        batch_size: Number of samples.
        device: Target device.
        normalize: Whether to standardize coordinates.

    Returns:
        A tensor of shape [batch_size, 2].
    """
    t = 1.5 * math.pi * (1.0 + 2.0 * torch.rand(batch_size, device=device))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    points = torch.stack((x, y), dim=-1)
    if normalize:
        return _standardize_points(points)
    return points


def get_toy_batch(dataset: str, batch_size: int, device: torch.device) -> Tensor:
    """Sample a toy dataset batch.

    Args:
        dataset: Name of toy dataset, one of {"checkerboard", "swissroll"}.
        batch_size: Number of samples.
        device: Target device.

    Returns:
        A tensor of shape [batch_size, 2].
    """
    if dataset == "checkerboard":
        return sample_checkerboard(batch_size=batch_size, device=device)
    if dataset == "swissroll":
        return sample_swissroll(batch_size=batch_size, device=device)
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
