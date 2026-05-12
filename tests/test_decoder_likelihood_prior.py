from __future__ import annotations

import torch

from decoder_likelihood_prior import (
    DecoderPriorSampler,
    _class_component_counts,
    _class_offsets,
    _fit_kmeans,
    _latent_components_for_assignments,
    _latent_moments_for_assignments,
)


def test_class_component_counts_cover_all_components():
    counts = _class_component_counts(8192, 1000)
    assert int(counts.sum()) == 8192
    assert int(counts.min()) == 8
    assert int(counts.max()) == 9
    offsets = _class_offsets(counts)
    assert int(offsets[0]) == 0
    assert int(offsets[-1] + counts[-1]) == 8192


def test_kmeans_and_latent_moments_shapes():
    x = torch.randn(12, 5)
    centers, assign = _fit_kmeans(x, 3, iters=2, seed=1, device=torch.device("cpu"))
    assert centers.shape == (3, 5)
    assert assign.shape == (12,)
    z = torch.randn(12, 4, 4, 2)
    means, scales, weights = _latent_moments_for_assignments(
        z,
        assign,
        3,
        min_scale=0.0,
        max_scale=2.0,
        device=torch.device("cpu"),
    )
    assert means.shape == (3, 4, 4, 2)
    assert scales.shape == (3, 4, 4, 2)
    assert weights.shape == (3,)
    assert int(weights.sum()) == 12


def test_medoid_components_use_nearest_descriptor_member():
    z = torch.tensor([[[[0.0]]], [[[4.0]]], [[[100.0]]]])
    desc = torch.tensor([[0.0], [4.0], [100.0]])
    desc_centers = torch.tensor([[5.0]])
    assign = torch.zeros(3, dtype=torch.long)
    centers, scales, weights = _latent_components_for_assignments(
        z,
        desc,
        desc_centers,
        assign,
        1,
        center_mode="medoid",
        min_scale=0.0,
        max_scale=0.0,
        device=torch.device("cpu"),
    )
    assert centers.shape == (1, 1, 1, 1)
    assert float(centers[0, 0, 0, 0]) == 4.0
    assert scales.shape == centers.shape
    assert int(weights.item()) == 3


def test_decoder_prior_sampler_indices(tmp_path):
    prior = {
        "format": "decoder_likelihood_prior.v1",
        "config": {"component_count": 4, "num_classes": 2},
        "centers": torch.zeros(4, 32, 32, 4),
        "scales": torch.zeros(4, 32, 32, 4),
        "component_weights": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "class_offsets": torch.tensor([0, 2]),
        "class_counts": torch.tensor([2, 2]),
    }
    path = tmp_path / "prior.pt"
    torch.save(prior, path)
    sampler = DecoderPriorSampler(path, device=torch.device("cpu"))
    idx = sampler._sample_indices(torch.tensor([0, 0, 1, 1]))
    assert idx.shape == (4,)
    assert idx[:2].max() < 2
    assert idx[2:].min() >= 2
