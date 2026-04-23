"""Compatibility shim for resize.forward API."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward(
    img: torch.Tensor,
) -> torch.Tensor:
    batch_size, channels, _, _ = img.shape
    theta = torch.eye(2, 3, device=img.device, dtype=img.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    grid = F.affine_grid(theta, [batch_size, channels, 299, 299], align_corners=False)
    x = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return (x - 128.0) / 128.0
