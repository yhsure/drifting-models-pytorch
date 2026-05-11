from __future__ import annotations

import numpy as np
import torch

from memory_bank import ComponentArrayMemoryBank


def test_component_memory_bank_prefers_active_component_samples():
    np.random.seed(0)
    bank = ComponentArrayMemoryBank(num_classes=2, max_size=8)
    samples = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    labels = torch.tensor([0, 0, 0, 1])
    components = torch.tensor([2, 3, 2, 4])
    bank.add(samples, labels, components)

    out = bank.sample_likelihood(
        labels=np.array([0]),
        active_components=np.array([[2]]),
        active_weights=np.array([[1.0]]),
        n_samples=20,
        uniform_fraction=0.0,
    )

    assert out.shape == (1, 20, 1)
    assert set(out.reshape(-1).tolist()).issubset({10.0, 30.0})


def test_component_memory_bank_falls_back_to_class_when_component_missing():
    np.random.seed(1)
    bank = ComponentArrayMemoryBank(num_classes=2, max_size=8)
    samples = torch.tensor([[10.0], [20.0], [30.0]])
    labels = torch.tensor([0, 0, 0])
    components = torch.tensor([2, 3, 2])
    bank.add(samples, labels, components)

    out = bank.sample_likelihood(
        labels=np.array([0]),
        active_components=np.array([[99]]),
        active_weights=np.array([[1.0]]),
        n_samples=20,
        uniform_fraction=0.0,
    )

    assert out.shape == (1, 20, 1)
    assert set(out.reshape(-1).tolist()).issubset({10.0, 20.0, 30.0})
