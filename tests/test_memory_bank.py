import numpy as np
import torch

from memory_bank import (
    ArrayMemoryBank,
    force_leverage_proposal_with_stats,
    rank_mixture_proposal,
    rank_mixture_proposal_with_stats,
    rank_stratified_proposal_with_stats,
)


def test_rank_mixture_proposal_importance_weights_recover_uniform_average():
    values = np.arange(10, dtype=np.float64)
    proposal = rank_mixture_proposal(
        values,
        local_alpha=0.8,
        rank_temperature=2.0,
        top_k=5,
    )
    weights = (1.0 / values.shape[0]) / proposal

    assert np.isclose(proposal.sum(), 1.0)
    assert proposal[0] > proposal[-1]
    assert weights[-1] > weights[0]
    assert np.isclose(np.sum(proposal * weights * values), values.mean())


def test_rank_mixture_proposal_can_limit_importance_variance():
    values = np.arange(128, dtype=np.float64)
    proposal, stats = rank_mixture_proposal_with_stats(
        values,
        local_alpha=0.85,
        rank_temperature=12.0,
        top_k=48,
        min_ess_frac=0.35,
        max_importance_weight=4.0,
    )
    weights = (1.0 / values.shape[0]) / proposal

    assert stats["effective_alpha"] < 0.85
    assert stats["expected_ess_frac"] >= 0.35 - 1e-6
    assert stats["proposal_weight_max"] <= 4.0 + 1e-6
    assert np.isclose(weights.max(), stats["proposal_weight_max"])
    assert np.isclose(np.sum(proposal * weights * values), values.mean())


def test_importance_sampler_returns_local_samples_with_likelihood_weights():
    np.random.seed(0)
    bank = ArrayMemoryBank(num_classes=1, max_size=10)
    samples = torch.arange(10, dtype=torch.float32).view(10, 1)
    labels = torch.zeros((10,), dtype=torch.long)
    bank.add(samples, labels)

    out, weights, info = bank.sample_importance(
        torch.zeros((1,), dtype=torch.long),
        n_samples=256,
        anchors=torch.zeros((1, 1), dtype=torch.float32),
        local_alpha=0.8,
        rank_temperature=2.0,
        top_k=5,
    )

    assert out.shape == (1, 256, 1)
    assert weights.shape == (1, 256)
    assert float(out.float().mean()) < 4.5
    assert info["pos_sampler/ess_frac"] > 0.0
    assert info["pos_sampler/weight_max"] > 1.0
    assert "pos_sampler/effective_alpha" in info
    assert "pos_sampler/expected_ess_frac" in info


def test_memory_bank_state_dict_roundtrip_preserves_samples_and_features():
    bank = ArrayMemoryBank(num_classes=2, max_size=4)
    samples = torch.arange(6, dtype=torch.float32).view(6, 1)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    features = torch.arange(6, dtype=torch.float32).view(6, 1) + 10
    bank.add(samples, labels, features=features)

    restored = ArrayMemoryBank(num_classes=1, max_size=1)
    restored.load_state_dict(bank.state_dict())

    assert restored.num_classes == bank.num_classes
    assert restored.max_size == bank.max_size
    assert np.array_equal(restored.ptr, bank.ptr)
    assert np.array_equal(restored.count, bank.count)
    assert np.array_equal(restored.bank, bank.bank)
    assert np.array_equal(restored.bank_feat, bank.bank_feat)


def test_interclass_sampler_excludes_anchor_class_and_weights_candidate_proposal():
    np.random.seed(0)
    bank = ArrayMemoryBank(num_classes=3, max_size=4)
    samples = torch.tensor([[0.0], [1.0], [10.0], [11.0], [20.0], [21.0]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    features = samples.clone()
    bank.add(samples, labels, features=features)

    out, weights, info = bank.sample_interclass_importance(
        torch.tensor([0]),
        n_samples=64,
        anchor_features=torch.tensor([[10.0]]),
        candidate_pool_size=4,
        local_alpha=0.9,
        rank_temperature=1.0,
    )

    assert out.shape == (1, 64, 1)
    assert weights.shape == (1, 64)
    assert not set(out.reshape(-1).tolist()).intersection({0.0, 1.0})
    assert float(out.float().mean()) < 18.0
    assert info["neg_sampler/weight_max"] > 1.0
    assert info["neg_sampler/candidate_pool_size"] == 4.0


def test_importance_sampler_reports_adaptive_alpha():
    np.random.seed(0)
    bank = ArrayMemoryBank(num_classes=1, max_size=128)
    samples = torch.arange(128, dtype=torch.float32).view(128, 1)
    labels = torch.zeros((128,), dtype=torch.long)
    bank.add(samples, labels)

    _, _, info = bank.sample_importance(
        torch.zeros((1,), dtype=torch.long),
        n_samples=512,
        anchors=torch.zeros((1, 1), dtype=torch.float32),
        local_alpha=0.85,
        rank_temperature=12.0,
        top_k=48,
        min_ess_frac=0.35,
        max_importance_weight=4.0,
    )

    assert info["pos_sampler/effective_alpha"] < info["pos_sampler/local_alpha"]
    assert info["pos_sampler/expected_ess_frac"] >= 0.35 - 1e-6
    assert info["pos_sampler/proposal_weight_max"] <= 4.0 + 1e-6


def test_force_leverage_proposal_recovers_uniform_average_under_constraints():
    values = np.arange(64, dtype=np.float64)
    distances = (values / values.max()) ** 2
    proposal, stats = force_leverage_proposal_with_stats(
        distances,
        local_alpha=1.0,
        radii=[0.2, 0.05, 0.02],
        score_floor=0.02,
        min_ess_frac=0.55,
        max_importance_weight=2.5,
    )
    weights = (1.0 / values.shape[0]) / proposal

    assert np.isclose(proposal.sum(), 1.0)
    assert stats["expected_ess_frac"] >= 0.55 - 1e-6
    assert stats["proposal_weight_max"] <= 2.5 + 1e-6
    assert stats["near_mass"] > 0.125
    assert np.isclose(np.sum(proposal * weights * values), values.mean())


def test_force_leverage_sampler_returns_likelihood_weights():
    np.random.seed(0)
    bank = ArrayMemoryBank(num_classes=1, max_size=64)
    samples = torch.arange(64, dtype=torch.float32).view(64, 1)
    labels = torch.zeros((64,), dtype=torch.long)
    features = torch.linspace(0, 1, 64, dtype=torch.float32).view(64, 1)
    bank.add(samples, labels, features=features)

    out, weights, info = bank.sample_importance_force_leverage(
        torch.zeros((1,), dtype=torch.long),
        n_samples=256,
        anchor_features=torch.tensor([[0.0]]),
        local_alpha=1.0,
        radii=[0.2, 0.05, 0.02],
        score_floor=0.02,
        min_ess_frac=0.55,
        max_importance_weight=2.5,
    )

    assert out.shape == (1, 256, 1)
    assert weights.shape == (1, 256)
    assert info["pos_sampler/effective_alpha"] <= info["pos_sampler/local_alpha"]
    assert info["pos_sampler/expected_ess_frac"] >= 0.55 - 1e-6
    assert info["pos_sampler/proposal_weight_max"] <= 2.5 + 1e-6
    assert info["pos_sampler/near_mass"] > 0.125


def test_importance_mix_uses_feature_neighbors_with_uniform_target_weights():
    np.random.seed(0)
    bank = ArrayMemoryBank(num_classes=1, max_size=6)
    samples = torch.arange(6, dtype=torch.float32).view(6, 1)
    labels = torch.zeros((6,), dtype=torch.long)
    features = torch.tensor([[20.0], [0.0], [1.0], [2.0], [10.0], [11.0]])
    bank.add(samples, labels, features=features)

    out, weights, info = bank.sample_importance_mix(
        torch.zeros((1,), dtype=torch.long),
        n_samples=4,
        anchor_features=torch.tensor([[0.0]]),
        local_k=2,
        global_r=2,
    )
    proposal = np.full((6,), (2 / 4) / 6, dtype=np.float64)
    proposal[[1, 2]] += (2 / 4) / 2
    expected_weights = (1.0 / 6) / proposal

    values = set(out.reshape(-1).tolist())
    assert {1.0, 2.0}.issubset(values)
    for sample, weight in zip(out.reshape(-1).numpy(), weights.reshape(-1).numpy()):
        assert np.isclose(weight, expected_weights[int(sample)])
    assert info["pos_sampler/local_frac"] == 0.5
    assert info["pos_sampler/weight_max"] > 1.0


def test_importance_mix_weight_power_zero_recovers_unit_weights():
    np.random.seed(0)
    bank = ArrayMemoryBank(num_classes=1, max_size=6)
    samples = torch.arange(6, dtype=torch.float32).view(6, 1)
    labels = torch.zeros((6,), dtype=torch.long)
    features = torch.arange(6, dtype=torch.float32).view(6, 1)
    bank.add(samples, labels, features=features)

    _, weights, info = bank.sample_importance_mix(
        torch.zeros((1,), dtype=torch.long),
        n_samples=4,
        anchor_features=torch.tensor([[0.0]]),
        local_k=2,
        global_r=2,
        weight_power=0.0,
    )

    assert torch.allclose(weights, torch.ones_like(weights))
    assert info["pos_sampler/weight_power"] == 0.0


def test_rank_stratified_proposal_recovers_uniform_average():
    values = np.arange(12, dtype=np.float64)
    proposal, stats = rank_stratified_proposal_with_stats(
        values,
        n_samples=12,
        local_alpha=1.0,
        strata_edges=[0.25, 0.5, 1.0],
        allocation=[0.6, 0.3, 0.1],
        max_importance_weight=4.0,
    )
    weights = (1.0 / values.shape[0]) / proposal

    assert np.isclose(proposal.sum(), 1.0)
    assert proposal[0] > proposal[-1]
    assert weights[-1] > weights[0]
    assert stats["proposal_weight_max"] <= 4.0 + 1e-6
    assert np.isclose(np.sum(proposal * weights * values), values.mean())


def test_stratified_sampler_returns_exact_stratum_weights():
    np.random.seed(0)
    bank = ArrayMemoryBank(num_classes=1, max_size=12)
    samples = torch.arange(12, dtype=torch.float32).view(12, 1)
    labels = torch.zeros((12,), dtype=torch.long)
    features = torch.arange(12, dtype=torch.float32).view(12, 1)
    bank.add(samples, labels, features=features)

    out, weights, info = bank.sample_importance_stratified(
        torch.zeros((1,), dtype=torch.long),
        n_samples=64,
        anchor_features=torch.tensor([[0.0]]),
        local_alpha=1.0,
        strata_edges=[0.25, 0.5, 1.0],
        allocation=[0.6, 0.3, 0.1],
    )
    proposal, _ = rank_stratified_proposal_with_stats(
        np.arange(12, dtype=np.float64) ** 2,
        n_samples=64,
        local_alpha=1.0,
        strata_edges=[0.25, 0.5, 1.0],
        allocation=[0.6, 0.3, 0.1],
    )
    expected_weights = (1.0 / 12) / proposal

    assert out.shape == (1, 64, 1)
    assert weights.shape == (1, 64)
    assert info["pos_sampler/stratum_alloc_0"] > info["pos_sampler/stratum_alloc_2"]
    for sample, weight in zip(out.reshape(-1).numpy(), weights.reshape(-1).numpy()):
        assert np.isclose(weight, expected_weights[int(sample)])
