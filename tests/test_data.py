"""Tests for toy data samplers."""

from __future__ import annotations

import torch

from drifting_models_pytorch.data import sample_checkerboard, sample_swissroll


def test_checkerboard_sampler_hits_alternating_cells() -> None:
    """Checkerboard samples should lie in alternating unit cells."""
    points = sample_checkerboard(batch_size=2048, device=torch.device("cpu"), normalize=False, noise=0.0)
    shifted = points * 2.0 + 2.0
    cells = torch.floor(shifted).to(dtype=torch.long)
    valid_x = (cells[:, 0] >= 0) & (cells[:, 0] <= 3)
    valid_y = (cells[:, 1] >= 0) & (cells[:, 1] <= 3)
    parity = (cells[:, 0] + cells[:, 1]) % 2
    assert bool((valid_x & valid_y).all())
    assert bool((parity == 0).all())


def test_swissroll_sampler_has_spread() -> None:
    """Swissroll samples should have non-trivial variance on both axes."""
    points = sample_swissroll(batch_size=2048, device=torch.device("cpu"), normalize=False)
    std = points.std(dim=0)
    assert float(std[0]) > 0.2
    assert float(std[1]) > 0.2
