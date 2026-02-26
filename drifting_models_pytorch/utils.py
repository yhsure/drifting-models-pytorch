"""Shared utilities for experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Target path.

    Returns:
        The path object.
    """
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to json.

    Args:
        data: Serializable content.
        path: Output file path.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def mmd_rbf(x: Tensor, y: Tensor, sigma: float = 1.0) -> Tensor:
    """Compute RBF-kernel MMD^2 between two batches.

    Args:
        x: Batch of samples [n, d].
        y: Batch of samples [m, d].
        sigma: Kernel bandwidth.

    Returns:
        Scalar MMD^2 estimate.
    """
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    k_xx = torch.exp(-xx / (2.0 * sigma**2))
    k_yy = torch.exp(-yy / (2.0 * sigma**2))
    k_xy = torch.exp(-xy / (2.0 * sigma**2))
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
