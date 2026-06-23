from __future__ import annotations

import numpy as np
import torch


def rank_mixture_proposal(
    distances: np.ndarray,
    *,
    local_alpha: float,
    rank_temperature: float,
    top_k: int = 0,
) -> np.ndarray:
    """Local-plus-uniform proposal over one class bank."""

    proposal, _ = rank_mixture_proposal_with_stats(
        distances,
        local_alpha=local_alpha,
        rank_temperature=rank_temperature,
        top_k=top_k,
    )
    return proposal


def _uniform_importance_stats(proposal: np.ndarray) -> tuple[float, float]:
    proposal = np.asarray(proposal, dtype=np.float64)
    n = int(proposal.shape[0])
    clipped = np.clip(proposal, 1e-30, None)
    ess_frac = float((n * n) / np.sum(1.0 / clipped))
    max_weight = float((1.0 / n) / clipped.min())
    return ess_frac, max_weight


def rank_mixture_proposal_with_stats(
    distances: np.ndarray,
    *,
    local_alpha: float,
    rank_temperature: float,
    top_k: int = 0,
    min_ess_frac: float | None = None,
    max_importance_weight: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Local-plus-uniform proposal plus exact uniform-target diagnostics."""

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError(f"Expected 1D distances, got shape={distances.shape}")
    n = int(distances.shape[0])
    if n <= 0:
        raise ValueError("Cannot build a proposal for an empty class bank.")

    alpha = float(np.clip(local_alpha, 0.0, 1.0))
    uniform = np.full((n,), 1.0 / n, dtype=np.float64)
    if alpha <= 0.0:
        ess_frac, proposal_weight_max = _uniform_importance_stats(uniform)
        return uniform, {
            "effective_alpha": 0.0,
            "expected_ess_frac": ess_frac,
            "proposal_weight_max": proposal_weight_max,
        }

    k = n if int(top_k) <= 0 else min(n, int(top_k))
    order = np.argsort(distances, kind="stable")[:k]
    temperature = max(float(rank_temperature), 1e-6)
    logits = -np.arange(k, dtype=np.float64) / temperature
    logits = logits - logits.max()
    local = np.zeros((n,), dtype=np.float64)
    local[order] = np.exp(logits)
    local_sum = float(local.sum())
    if local_sum <= 0.0 or not np.isfinite(local_sum):
        local = uniform
    else:
        local = local / local_sum

    def build_proposal(alpha_value: float) -> np.ndarray:
        proposal_value = (1.0 - alpha_value) * uniform + alpha_value * local
        return proposal_value / proposal_value.sum()

    min_ess = 0.0 if min_ess_frac is None else float(min_ess_frac)
    max_weight = 0.0 if max_importance_weight is None else float(max_importance_weight)

    def is_ok(proposal_value: np.ndarray) -> bool:
        ess_frac, proposal_weight_max = _uniform_importance_stats(proposal_value)
        if min_ess > 0.0 and ess_frac < min_ess:
            return False
        if max_weight > 0.0 and proposal_weight_max > max_weight:
            return False
        return True

    proposal = build_proposal(alpha)
    if not is_ok(proposal):
        lo, hi = 0.0, alpha
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            candidate = build_proposal(mid)
            if is_ok(candidate):
                lo = mid
            else:
                hi = mid
        alpha = lo
        proposal = build_proposal(alpha)

    ess_frac, proposal_weight_max = _uniform_importance_stats(proposal)
    return proposal, {
        "effective_alpha": float(alpha),
        "expected_ess_frac": ess_frac,
        "proposal_weight_max": proposal_weight_max,
    }


def _rank_band_stats(proposal: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    n = int(proposal.shape[0])
    if n <= 0:
        return {}
    order = np.argsort(distances, kind="stable")
    bands = {
        "near_mass": (0.0, 0.125),
        "mid_mass": (0.125, 0.5),
        "far_mass": (0.5, 1.0),
    }
    stats = {}
    for name, (lo, hi) in bands.items():
        start = int(round(lo * n))
        end = n if hi >= 1.0 else int(round(hi * n))
        start = max(0, min(n, start))
        end = max(start, min(n, end))
        stats[name] = float(proposal[order[start:end]].sum()) if end > start else 0.0
    return stats


def force_leverage_proposal_with_stats(
    distances: np.ndarray,
    *,
    local_alpha: float,
    radii=(0.2, 0.05, 0.02),
    score_floor: float = 0.05,
    distance_power: float = 1.0,
    kernel_power: float = 1.0,
    distance_scale: str = "mean",
    min_ess_frac: float | None = None,
    max_importance_weight: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Continuous drift-force proxy proposal with uniform-target diagnostics.

    The target measure remains uniform over the class bank. The proposal is a
    uniform/continuous-leverage mixture, where leverage approximates the size of
    the drift force a support can exert at the current anchor.
    """

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError(f"Expected 1D distances, got shape={distances.shape}")
    n = int(distances.shape[0])
    if n <= 0:
        raise ValueError("Cannot build a proposal for an empty class bank.")

    uniform = np.full((n,), 1.0 / n, dtype=np.float64)
    alpha = float(np.clip(local_alpha, 0.0, 1.0))
    if alpha <= 0.0:
        ess_frac, proposal_weight_max = _uniform_importance_stats(uniform)
        return uniform, {
            "effective_alpha": 0.0,
            "expected_ess_frac": ess_frac,
            "proposal_weight_max": proposal_weight_max,
            "score_entropy": 1.0,
            **_rank_band_stats(uniform, distances),
        }

    dist = np.sqrt(np.clip(distances, 0.0, None))
    scale_mode = str(distance_scale or "mean").strip().lower()
    if scale_mode == "median":
        scale = float(np.median(dist))
    elif scale_mode.startswith("q"):
        try:
            q = float(scale_mode[1:]) / 100.0
        except ValueError:
            q = 0.75
        scale = float(np.quantile(dist, np.clip(q, 0.01, 0.99)))
    else:
        scale = float(np.mean(dist))
    norm_dist = dist / max(scale, 1e-12)

    leverage_sq = np.zeros((n,), dtype=np.float64)
    radii = [float(r) for r in radii]
    for radius in radii:
        radius = max(radius, 1e-6)
        component = np.power(np.clip(norm_dist, 1e-12, None), float(distance_power))
        component *= np.exp(-np.power(norm_dist / radius, float(kernel_power)))
        leverage_sq += component * component
    leverage = np.sqrt(leverage_sq)
    score = np.clip(leverage, 0.0, None) + max(float(score_floor), 0.0)
    score_sum = float(score.sum())
    if score_sum <= 0.0 or not np.isfinite(score_sum):
        local = uniform
    else:
        local = score / score_sum

    def build_proposal(alpha_value: float) -> np.ndarray:
        proposal_value = (1.0 - alpha_value) * uniform + alpha_value * local
        return proposal_value / proposal_value.sum()

    min_ess = 0.0 if min_ess_frac is None else float(min_ess_frac)
    max_weight = 0.0 if max_importance_weight is None else float(max_importance_weight)

    def is_ok(proposal_value: np.ndarray) -> bool:
        ess_frac, proposal_weight_max = _uniform_importance_stats(proposal_value)
        if min_ess > 0.0 and ess_frac < min_ess:
            return False
        if max_weight > 0.0 and proposal_weight_max > max_weight:
            return False
        return True

    proposal = build_proposal(alpha)
    if not is_ok(proposal):
        lo, hi = 0.0, alpha
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            candidate = build_proposal(mid)
            if is_ok(candidate):
                lo = mid
            else:
                hi = mid
        alpha = lo
        proposal = build_proposal(alpha)

    ess_frac, proposal_weight_max = _uniform_importance_stats(proposal)
    score_entropy = float(-(local * np.log(np.clip(local, 1e-30, None))).sum() / max(np.log(n), 1e-12))
    return proposal, {
        "effective_alpha": float(alpha),
        "expected_ess_frac": ess_frac,
        "proposal_weight_max": proposal_weight_max,
        "score_entropy": score_entropy,
        "distance_scale": float(scale),
        "mean_norm_dist": float(norm_dist.mean()),
        **_rank_band_stats(proposal, distances),
    }


def _normalize_strata_edges(strata_edges) -> list[float]:
    if strata_edges is None:
        edges = [0.125, 0.5, 1.0]
    else:
        edges = [float(x) for x in strata_edges]
    edges = sorted({float(np.clip(edge, 1e-6, 1.0)) for edge in edges})
    if not edges or edges[-1] < 1.0:
        edges.append(1.0)
    return edges


def _strata_from_distances(
    distances: np.ndarray,
    strata_edges,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    distances = np.asarray(distances, dtype=np.float64)
    n = int(distances.shape[0])
    order = np.argsort(distances, kind="stable")
    edges = _normalize_strata_edges(strata_edges)
    strata = []
    bounds = []
    start = 0
    for edge in edges:
        end = int(round(edge * n))
        if edge >= 1.0:
            end = n
        end = max(start, min(n, end))
        if end > start:
            strata.append(order[start:end].astype(np.int32))
            bounds.append((start, end))
        start = end
    if start < n:
        strata.append(order[start:n].astype(np.int32))
        bounds.append((start, n))
    return order, strata, bounds


def _counts_from_allocation(allocation: np.ndarray, n_samples: int, min_per_stratum: int) -> np.ndarray:
    allocation = np.asarray(allocation, dtype=np.float64)
    h = int(allocation.shape[0])
    if h <= 0:
        return np.zeros((0,), dtype=np.int32)
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    min_count = max(0, int(min_per_stratum))
    if h * min_count > n_samples:
        min_count = n_samples // h

    counts = np.full((h,), min_count, dtype=np.int32)
    remaining = n_samples - int(counts.sum())
    if remaining <= 0:
        return counts

    alloc = np.clip(allocation, 0.0, None)
    alloc_sum = float(alloc.sum())
    if alloc_sum <= 0.0 or not np.isfinite(alloc_sum):
        alloc = np.full((h,), 1.0 / h, dtype=np.float64)
    else:
        alloc = alloc / alloc_sum

    raw = alloc * remaining
    extra = np.floor(raw).astype(np.int32)
    counts += extra
    leftover = n_samples - int(counts.sum())
    if leftover > 0:
        order = np.argsort(-(raw - extra), kind="stable")
        counts[order[:leftover]] += 1
    return counts


def rank_stratified_proposal_with_stats(
    distances: np.ndarray,
    *,
    n_samples: int,
    local_alpha: float,
    strata_edges=None,
    rank_temperatures=None,
    score_floor: float = 0.05,
    allocation=None,
    min_per_stratum: int = 1,
    min_ess_frac: float | None = None,
    max_importance_weight: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rank-stratified proposal with exact uniform-target diagnostics.

    Strata are contiguous bands in distance-rank space. The target remains
    uniform over the class bank; nonuniform strata only change the proposal q.
    """

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError(f"Expected 1D distances, got shape={distances.shape}")
    n = int(distances.shape[0])
    if n <= 0:
        raise ValueError("Cannot build a proposal for an empty class bank.")

    _, strata, bounds = _strata_from_distances(distances, strata_edges)
    h = len(strata)
    mass = np.asarray([len(idx) / float(n) for idx in strata], dtype=np.float64)
    rank_frac = (np.arange(n, dtype=np.float64) + 0.5) / max(1, n)

    if allocation is not None:
        desired = np.asarray([float(x) for x in allocation], dtype=np.float64)
        if desired.shape[0] != h:
            desired = mass.copy()
        else:
            desired = np.clip(desired, 0.0, None)
    else:
        temps = [0.05, 0.2, 0.5] if rank_temperatures is None else [float(x) for x in rank_temperatures]
        proxy = np.full((n,), max(float(score_floor), 0.0), dtype=np.float64)
        for temp in temps:
            proxy += np.exp(-rank_frac / max(float(temp), 1e-6))
        desired = np.asarray(
            [
                mass_i * float(np.sqrt(np.mean(proxy[start:end] * proxy[start:end])))
                for mass_i, (start, end) in zip(mass, bounds, strict=True)
            ],
            dtype=np.float64,
        )
    desired_sum = float(desired.sum())
    if desired_sum <= 0.0 or not np.isfinite(desired_sum):
        desired = mass.copy()
    else:
        desired = desired / desired_sum

    alpha = float(np.clip(local_alpha, 0.0, 1.0))

    def build_proposal(alpha_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        continuous = (1.0 - alpha_value) * mass + alpha_value * desired
        continuous = continuous / max(float(continuous.sum()), 1e-12)
        counts = _counts_from_allocation(continuous, n_samples=n_samples, min_per_stratum=min_per_stratum)
        actual = counts.astype(np.float64) / max(1, int(counts.sum()))
        proposal_value = np.zeros((n,), dtype=np.float64)
        for alloc_h, idx_h in zip(actual, strata, strict=True):
            if len(idx_h) > 0 and alloc_h > 0.0:
                proposal_value[idx_h] = alloc_h / float(len(idx_h))
        proposal_sum = float(proposal_value.sum())
        if proposal_sum <= 0.0 or not np.isfinite(proposal_sum):
            proposal_value[:] = 1.0 / n
        else:
            proposal_value = proposal_value / proposal_sum
        return proposal_value, actual, counts

    min_ess = 0.0 if min_ess_frac is None else float(min_ess_frac)
    max_weight = 0.0 if max_importance_weight is None else float(max_importance_weight)

    def is_ok(proposal_value: np.ndarray) -> bool:
        ess_frac, proposal_weight_max = _uniform_importance_stats(proposal_value)
        if min_ess > 0.0 and ess_frac < min_ess:
            return False
        if max_weight > 0.0 and proposal_weight_max > max_weight:
            return False
        return True

    proposal, actual, counts = build_proposal(alpha)
    if not is_ok(proposal):
        lo, hi = 0.0, alpha
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            candidate, _, _ = build_proposal(mid)
            if is_ok(candidate):
                lo = mid
            else:
                hi = mid
        alpha = lo
        proposal, actual, counts = build_proposal(alpha)

    ess_frac, proposal_weight_max = _uniform_importance_stats(proposal)
    stats = {
        "effective_alpha": float(alpha),
        "expected_ess_frac": ess_frac,
        "proposal_weight_max": proposal_weight_max,
        "num_strata": float(h),
    }
    for i in range(min(h, 4)):
        stats[f"stratum_alloc_{i}"] = float(actual[i])
        stats[f"stratum_mass_{i}"] = float(mass[i])
        stats[f"stratum_count_{i}"] = float(counts[i])
    return proposal, stats


class ArrayMemoryBank:
    """Class-wise ring buffer for feature/image samples used by generator training."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: np.ndarray | None = None
        self.bank_feat: np.ndarray | None = None
        self.feature_shape: tuple[int, ...] | None = None
        self.feat_dim: int | None = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    def _init_bank(self, sample_shape: tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype)

    def _maybe_init_feat_bank(self, feat_dim: int) -> None:
        feat_dim = int(feat_dim)
        if self.feat_dim is None:
            self.feat_dim = feat_dim
        elif self.feat_dim != feat_dim:
            raise ValueError(f"Feature dim changed from {self.feat_dim} to {feat_dim}.")
        if self.bank_feat is None:
            self.bank_feat = np.zeros((self.num_classes, self.max_size, self.feat_dim), dtype=self.dtype)

    def add(self, samples, labels, features=None) -> None:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        if self.bank is None:
            self._init_bank(samples.shape[1:])

        features_np = None
        if features is not None:
            features_np = np.asarray(features, dtype=self.dtype).reshape(labels.shape[0], -1)
            self._maybe_init_feat_bank(features_np.shape[1])

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            if features_np is not None and self.bank_feat is not None:
                self.bank_feat[lbl, idx] = features_np[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    @staticmethod
    def _as_numpy_optional(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _distance_inputs(self, lbl: int, valid: int, anchor, anchor_feature):
        if anchor_feature is not None and self.bank_feat is not None:
            pool = self.bank_feat[lbl, :valid]
            query = np.asarray(anchor_feature, dtype=np.float32).reshape(1, -1)
            return pool, query
        if anchor is not None and self.bank is not None:
            pool = self.bank[lbl, :valid]
            query = np.asarray(anchor, dtype=np.float32).reshape(1, -1)
            return pool, query
        return None, None

    def state_dict(self) -> dict:
        """Return a torch-saveable snapshot of the bank contents."""

        return {
            "num_classes": int(self.num_classes),
            "max_size": int(self.max_size),
            "dtype": np.dtype(self.dtype).name,
            "bank": self.bank,
            "bank_feat": self.bank_feat,
            "feature_shape": None if self.feature_shape is None else tuple(self.feature_shape),
            "feat_dim": self.feat_dim,
            "ptr": self.ptr,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a bank snapshot produced by :meth:`state_dict`."""

        self.num_classes = int(state["num_classes"])
        self.max_size = int(state["max_size"])
        self.dtype = np.dtype(state.get("dtype", np.dtype(self.dtype).name)).type
        feature_shape = state.get("feature_shape")
        self.feature_shape = None if feature_shape is None else tuple(feature_shape)
        self.feat_dim = None if state.get("feat_dim") is None else int(state["feat_dim"])
        bank = state.get("bank")
        bank_feat = state.get("bank_feat")
        self.bank = None if bank is None else np.asarray(bank, dtype=self.dtype)
        self.bank_feat = None if bank_feat is None else np.asarray(bank_feat, dtype=self.dtype)
        self.ptr = np.asarray(state["ptr"], dtype=np.int32)
        self.count = np.asarray(state["count"], dtype=np.int32)

    def sample(self, labels, n_samples: int):
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
            else:
                sample_indices[i] = np.random.choice(valid, n_samples, replace=(valid < n_samples))

        out = self.bank[labels[:, None], sample_indices]
        return torch.from_numpy(out)

    def sample_interclass_importance(
        self,
        labels,
        n_samples: int,
        *,
        anchors=None,
        anchor_features=None,
        candidate_pool_size: int = 2048,
        local_alpha: float = 0.8,
        rank_temperature: float = 32.0,
        top_k: int = 0,
        min_ess_frac: float | None = None,
        max_importance_weight: float | None = None,
        weight_clip: float | None = None,
        weight_power: float = 1.0,
        normalize_weights: bool = False,
    ):
        """Sample close other-class supports with p/q weights within a uniform candidate pool.

        The first stage draws a uniform candidate pool from all currently valid
        bank entries outside the anchor class. The second stage draws locally
        within that pool and returns exact likelihood weights for the candidate
        pool's uniform target measure.
        """

        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        anchors = self._as_numpy_optional(anchors)
        anchor_features = self._as_numpy_optional(anchor_features)

        bsz = labels.shape[0]
        out = np.zeros((bsz, n_samples, *self.feature_shape), dtype=self.dtype)
        weights = np.ones((bsz, n_samples), dtype=np.float32)
        ess = np.ones((bsz,), dtype=np.float64)
        max_weight = np.ones((bsz,), dtype=np.float64)
        mean_weight = np.ones((bsz,), dtype=np.float64)
        entropy = np.ones((bsz,), dtype=np.float64)
        sampled_rank = np.zeros((bsz,), dtype=np.float64)
        effective_alpha = np.zeros((bsz,), dtype=np.float64)
        expected_ess = np.ones((bsz,), dtype=np.float64)
        proposal_weight_max = np.ones((bsz,), dtype=np.float64)
        observed_pool_size = np.zeros((bsz,), dtype=np.float64)
        valid_class_frac = np.zeros((bsz,), dtype=np.float64)

        valid_all = np.flatnonzero(self.count > 0).astype(np.int32)
        for i in range(bsz):
            excluded = int(labels[i])
            valid_labels = valid_all[valid_all != excluded]
            if valid_labels.size <= 0:
                continue

            counts = self.count[valid_labels].astype(np.int64)
            total_valid = int(counts.sum())
            if total_valid <= 0:
                continue

            requested_pool = int(candidate_pool_size)
            if requested_pool <= 0:
                pool_size = total_valid
            else:
                pool_size = min(total_valid, max(int(n_samples), requested_pool))

            draws = np.random.choice(total_valid, size=pool_size, replace=(total_valid < pool_size))
            cum = np.cumsum(counts)
            class_pos = np.searchsorted(cum, draws, side="right")
            prev = np.concatenate([np.zeros((1,), dtype=np.int64), cum[:-1]])
            candidate_labels = valid_labels[class_pos]
            candidate_slots = (draws - prev[class_pos]).astype(np.int64)
            candidate_samples = self.bank[candidate_labels, candidate_slots]

            anchor = anchors[i] if anchors is not None else None
            anchor_feature = anchor_features[i] if anchor_features is not None else None
            if anchor_feature is not None and self.bank_feat is not None:
                flat_candidates = self.bank_feat[candidate_labels, candidate_slots].astype(np.float32, copy=False)
                dist_anchor = np.asarray(anchor_feature, dtype=np.float32).reshape(1, -1)
            elif anchor is not None:
                flat_candidates = candidate_samples.astype(np.float32, copy=False).reshape(pool_size, -1)
                dist_anchor = np.asarray(anchor, dtype=np.float32).reshape(1, -1)
            else:
                flat_candidates = None
                dist_anchor = None

            if flat_candidates is None or dist_anchor is None or float(local_alpha) <= 0.0:
                proposal = np.full((pool_size,), 1.0 / pool_size, dtype=np.float64)
                rank_lookup = np.arange(pool_size, dtype=np.float64)
                proposal_stats = {
                    "effective_alpha": 0.0,
                    "expected_ess_frac": 1.0,
                    "proposal_weight_max": 1.0,
                }
            else:
                diff = flat_candidates.reshape(pool_size, -1) - dist_anchor.reshape(1, -1)
                distances = np.mean(diff * diff, axis=1)
                proposal, proposal_stats = rank_mixture_proposal_with_stats(
                    distances,
                    local_alpha=local_alpha,
                    rank_temperature=rank_temperature,
                    top_k=top_k,
                    min_ess_frac=min_ess_frac,
                    max_importance_weight=max_importance_weight,
                )
                order = np.argsort(distances, kind="stable")
                rank_lookup = np.empty((pool_size,), dtype=np.float64)
                rank_lookup[order] = np.arange(pool_size, dtype=np.float64)

            idx = np.random.choice(pool_size, n_samples, replace=True, p=proposal)
            row_weights = (1.0 / pool_size) / proposal[idx]
            if weight_clip is not None and float(weight_clip) > 0.0:
                row_weights = np.minimum(row_weights, float(weight_clip))
            power = float(weight_power)
            if abs(power - 1.0) > 1e-12:
                row_weights = np.power(np.clip(row_weights, 1e-30, None), power)
            if normalize_weights:
                row_weights = row_weights / max(float(row_weights.mean()), 1e-8)

            out[i] = candidate_samples[idx]
            weights[i] = row_weights.astype(np.float32)
            weight_sum = float(row_weights.sum())
            weight_sq_sum = float(np.sum(row_weights * row_weights))
            ess[i] = (weight_sum * weight_sum / max(weight_sq_sum, 1e-12)) / max(1, n_samples)
            max_weight[i] = float(row_weights.max())
            mean_weight[i] = float(row_weights.mean())
            entropy_denom = max(float(np.log(pool_size)), 1e-12)
            entropy[i] = float(-(proposal * np.log(np.clip(proposal, 1e-30, None))).sum() / entropy_denom)
            sampled_rank[i] = float(rank_lookup[idx].mean() / max(1, pool_size - 1))
            effective_alpha[i] = proposal_stats["effective_alpha"]
            expected_ess[i] = proposal_stats["expected_ess_frac"]
            proposal_weight_max[i] = proposal_stats["proposal_weight_max"]
            observed_pool_size[i] = float(pool_size)
            valid_class_frac[i] = float(valid_labels.size / max(1, self.num_classes - 1))

        info = {
            "neg_sampler/ess_frac": float(ess.mean()),
            "neg_sampler/weight_max": float(max_weight.mean()),
            "neg_sampler/weight_mean": float(mean_weight.mean()),
            "neg_sampler/proposal_entropy": float(entropy.mean()),
            "neg_sampler/sampled_rank_frac": float(sampled_rank.mean()),
            "neg_sampler/local_alpha": float(local_alpha),
            "neg_sampler/effective_alpha": float(effective_alpha.mean()),
            "neg_sampler/expected_ess_frac": float(expected_ess.mean()),
            "neg_sampler/proposal_weight_max": float(proposal_weight_max.mean()),
            "neg_sampler/weight_power": float(weight_power),
            "neg_sampler/candidate_pool_size": float(observed_pool_size.mean()),
            "neg_sampler/valid_class_frac": float(valid_class_frac.mean()),
        }
        return torch.from_numpy(out), torch.from_numpy(weights), info

    def sample_importance(
        self,
        labels,
        n_samples: int,
        *,
        anchors=None,
        anchor_features=None,
        local_alpha: float = 0.0,
        rank_temperature: float = 16.0,
        top_k: int = 0,
        min_ess_frac: float | None = None,
        max_importance_weight: float | None = None,
        weight_clip: float | None = None,
        weight_power: float = 1.0,
        normalize_weights: bool = False,
    ):
        """Sample positives from a local proposal and return p/q likelihood weights.

        The target distribution p is uniform over valid examples in the class bank.
        The proposal q is a mixture of uniform mass and a rank-local kernel around
        the supplied anchor. Returned weights are p/q for each sampled point.
        """

        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        anchors = self._as_numpy_optional(anchors)
        anchor_features = self._as_numpy_optional(anchor_features)

        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        weights = np.ones((bsz, n_samples), dtype=np.float32)
        ess = np.ones((bsz,), dtype=np.float64)
        max_weight = np.ones((bsz,), dtype=np.float64)
        mean_weight = np.ones((bsz,), dtype=np.float64)
        entropy = np.ones((bsz,), dtype=np.float64)
        sampled_rank = np.zeros((bsz,), dtype=np.float64)
        effective_alpha = np.zeros((bsz,), dtype=np.float64)
        expected_ess = np.ones((bsz,), dtype=np.float64)
        proposal_weight_max = np.ones((bsz,), dtype=np.float64)

        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
                continue

            if (anchors is None and anchor_features is None) or float(local_alpha) <= 0.0:
                proposal = np.full((valid,), 1.0 / valid, dtype=np.float64)
                rank_lookup = np.arange(valid, dtype=np.float64)
            else:
                anchor = anchors[i] if anchors is not None else None
                anchor_feature = anchor_features[i] if anchor_features is not None else None
                dist_pool, dist_anchor = self._distance_inputs(lbl, valid, anchor, anchor_feature)
                if dist_pool is None or dist_anchor is None:
                    proposal = np.full((valid,), 1.0 / valid, dtype=np.float64)
                    rank_lookup = np.arange(valid, dtype=np.float64)
                    idx = np.random.choice(valid, n_samples, replace=True, p=proposal)
                    row_weights = np.ones((idx.shape[0],), dtype=np.float64)
                    sample_indices[i] = idx.astype(np.int32)
                    weights[i] = row_weights.astype(np.float32)
                    continue
                flat_candidates = dist_pool.astype(np.float32, copy=False).reshape(valid, -1)
                diff = flat_candidates - dist_anchor.astype(np.float32, copy=False).reshape(1, -1)
                distances = np.mean(diff * diff, axis=1)
                proposal, proposal_stats = rank_mixture_proposal_with_stats(
                    distances,
                    local_alpha=local_alpha,
                    rank_temperature=rank_temperature,
                    top_k=top_k,
                    min_ess_frac=min_ess_frac,
                    max_importance_weight=max_importance_weight,
                )
                effective_alpha[i] = proposal_stats["effective_alpha"]
                expected_ess[i] = proposal_stats["expected_ess_frac"]
                proposal_weight_max[i] = proposal_stats["proposal_weight_max"]
                order = np.argsort(distances, kind="stable")
                rank_lookup = np.empty((valid,), dtype=np.float64)
                rank_lookup[order] = np.arange(valid, dtype=np.float64)

            idx = np.random.choice(valid, n_samples, replace=True, p=proposal)
            target_prob = 1.0 / valid
            row_weights = target_prob / proposal[idx]
            if weight_clip is not None and float(weight_clip) > 0.0:
                row_weights = np.minimum(row_weights, float(weight_clip))
            power = float(weight_power)
            if abs(power - 1.0) > 1e-12:
                row_weights = np.power(np.clip(row_weights, 1e-30, None), power)
            if normalize_weights:
                row_weights = row_weights / max(float(row_weights.mean()), 1e-8)

            sample_indices[i] = idx.astype(np.int32)
            weights[i] = row_weights.astype(np.float32)
            weight_sum = float(row_weights.sum())
            weight_sq_sum = float(np.sum(row_weights * row_weights))
            ess[i] = (weight_sum * weight_sum / max(weight_sq_sum, 1e-12)) / max(1, n_samples)
            max_weight[i] = float(row_weights.max())
            mean_weight[i] = float(row_weights.mean())
            entropy_denom = max(float(np.log(valid)), 1e-12)
            entropy[i] = float(-(proposal * np.log(np.clip(proposal, 1e-30, None))).sum() / entropy_denom)
            sampled_rank[i] = float(rank_lookup[idx].mean() / max(1, valid - 1))

        out = self.bank[labels[:, None], sample_indices]
        info = {
            "pos_sampler/ess_frac": float(ess.mean()),
            "pos_sampler/weight_max": float(max_weight.mean()),
            "pos_sampler/weight_mean": float(mean_weight.mean()),
            "pos_sampler/proposal_entropy": float(entropy.mean()),
            "pos_sampler/sampled_rank_frac": float(sampled_rank.mean()),
            "pos_sampler/local_alpha": float(local_alpha),
            "pos_sampler/effective_alpha": float(effective_alpha.mean()),
            "pos_sampler/expected_ess_frac": float(expected_ess.mean()),
            "pos_sampler/proposal_weight_max": float(proposal_weight_max.mean()),
            "pos_sampler/weight_power": float(weight_power),
        }
        return torch.from_numpy(out), torch.from_numpy(weights), info

    def sample_importance_force_leverage(
        self,
        labels,
        n_samples: int,
        *,
        anchors=None,
        anchor_features=None,
        local_alpha: float = 0.75,
        radii=(0.2, 0.05, 0.02),
        score_floor: float = 0.05,
        distance_power: float = 1.0,
        kernel_power: float = 1.0,
        distance_scale: str = "mean",
        min_ess_frac: float | None = None,
        max_importance_weight: float | None = None,
        weight_clip: float | None = None,
        weight_power: float = 1.0,
        normalize_weights: bool = False,
    ):
        """Sample positives from a continuous drift-force proxy proposal."""

        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        anchors = self._as_numpy_optional(anchors)
        anchor_features = self._as_numpy_optional(anchor_features)

        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        weights = np.ones((bsz, n_samples), dtype=np.float32)
        ess = np.ones((bsz,), dtype=np.float64)
        max_weight = np.ones((bsz,), dtype=np.float64)
        mean_weight = np.ones((bsz,), dtype=np.float64)
        entropy = np.ones((bsz,), dtype=np.float64)
        sampled_rank = np.zeros((bsz,), dtype=np.float64)
        effective_alpha = np.zeros((bsz,), dtype=np.float64)
        expected_ess = np.ones((bsz,), dtype=np.float64)
        proposal_weight_max = np.ones((bsz,), dtype=np.float64)
        score_entropy = np.ones((bsz,), dtype=np.float64)
        near_mass = np.zeros((bsz,), dtype=np.float64)
        mid_mass = np.zeros((bsz,), dtype=np.float64)
        far_mass = np.zeros((bsz,), dtype=np.float64)
        distance_scale_stats = np.zeros((bsz,), dtype=np.float64)
        mean_norm_dist = np.zeros((bsz,), dtype=np.float64)

        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
                continue

            anchor = anchors[i] if anchors is not None else None
            anchor_feature = anchor_features[i] if anchor_features is not None else None
            dist_pool, dist_anchor = self._distance_inputs(lbl, valid, anchor, anchor_feature)
            if dist_pool is None or dist_anchor is None or float(local_alpha) <= 0.0:
                proposal = np.full((valid,), 1.0 / valid, dtype=np.float64)
                proposal_stats = {
                    "effective_alpha": 0.0,
                    "expected_ess_frac": 1.0,
                    "proposal_weight_max": 1.0,
                    "score_entropy": 1.0,
                    "near_mass": 0.125,
                    "mid_mass": 0.375,
                    "far_mass": 0.5,
                    "distance_scale": 0.0,
                    "mean_norm_dist": 0.0,
                }
                rank_lookup = np.arange(valid, dtype=np.float64)
            else:
                flat_candidates = dist_pool.astype(np.float32, copy=False).reshape(valid, -1)
                diff = flat_candidates - dist_anchor.astype(np.float32, copy=False).reshape(1, -1)
                distances = np.mean(diff * diff, axis=1)
                proposal, proposal_stats = force_leverage_proposal_with_stats(
                    distances,
                    local_alpha=local_alpha,
                    radii=radii,
                    score_floor=score_floor,
                    distance_power=distance_power,
                    kernel_power=kernel_power,
                    distance_scale=distance_scale,
                    min_ess_frac=min_ess_frac,
                    max_importance_weight=max_importance_weight,
                )
                order = np.argsort(distances, kind="stable")
                rank_lookup = np.empty((valid,), dtype=np.float64)
                rank_lookup[order] = np.arange(valid, dtype=np.float64)

            idx = np.random.choice(valid, n_samples, replace=True, p=proposal)
            target_prob = 1.0 / valid
            row_weights = target_prob / proposal[idx]
            if weight_clip is not None and float(weight_clip) > 0.0:
                row_weights = np.minimum(row_weights, float(weight_clip))
            power = float(weight_power)
            if abs(power - 1.0) > 1e-12:
                row_weights = np.power(np.clip(row_weights, 1e-30, None), power)
            if normalize_weights:
                row_weights = row_weights / max(float(row_weights.mean()), 1e-8)

            sample_indices[i] = idx.astype(np.int32)
            weights[i] = row_weights.astype(np.float32)
            weight_sum = float(row_weights.sum())
            weight_sq_sum = float(np.sum(row_weights * row_weights))
            ess[i] = (weight_sum * weight_sum / max(weight_sq_sum, 1e-12)) / max(1, n_samples)
            max_weight[i] = float(row_weights.max())
            mean_weight[i] = float(row_weights.mean())
            entropy_denom = max(float(np.log(valid)), 1e-12)
            entropy[i] = float(-(proposal * np.log(np.clip(proposal, 1e-30, None))).sum() / entropy_denom)
            sampled_rank[i] = float(rank_lookup[idx].mean() / max(1, valid - 1))
            effective_alpha[i] = proposal_stats["effective_alpha"]
            expected_ess[i] = proposal_stats["expected_ess_frac"]
            proposal_weight_max[i] = proposal_stats["proposal_weight_max"]
            score_entropy[i] = proposal_stats.get("score_entropy", 1.0)
            near_mass[i] = proposal_stats.get("near_mass", 0.0)
            mid_mass[i] = proposal_stats.get("mid_mass", 0.0)
            far_mass[i] = proposal_stats.get("far_mass", 0.0)
            distance_scale_stats[i] = proposal_stats.get("distance_scale", 0.0)
            mean_norm_dist[i] = proposal_stats.get("mean_norm_dist", 0.0)

        out = self.bank[labels[:, None], sample_indices]
        info = {
            "pos_sampler/ess_frac": float(ess.mean()),
            "pos_sampler/weight_max": float(max_weight.mean()),
            "pos_sampler/weight_mean": float(mean_weight.mean()),
            "pos_sampler/proposal_entropy": float(entropy.mean()),
            "pos_sampler/sampled_rank_frac": float(sampled_rank.mean()),
            "pos_sampler/local_alpha": float(local_alpha),
            "pos_sampler/effective_alpha": float(effective_alpha.mean()),
            "pos_sampler/expected_ess_frac": float(expected_ess.mean()),
            "pos_sampler/proposal_weight_max": float(proposal_weight_max.mean()),
            "pos_sampler/weight_power": float(weight_power),
            "pos_sampler/score_entropy": float(score_entropy.mean()),
            "pos_sampler/near_mass": float(near_mass.mean()),
            "pos_sampler/mid_mass": float(mid_mass.mean()),
            "pos_sampler/far_mass": float(far_mass.mean()),
            "pos_sampler/distance_scale": float(distance_scale_stats.mean()),
            "pos_sampler/mean_norm_dist": float(mean_norm_dist.mean()),
        }
        return torch.from_numpy(out), torch.from_numpy(weights), info

    def sample_importance_mix(
        self,
        labels,
        n_samples: int,
        *,
        anchors=None,
        anchor_features=None,
        local_k: int = 0,
        global_r: int = 0,
        local_mode: str = "nearest",
        weight_power: float = 1.0,
    ):
        """Sample a deterministic local/global support mix with p/q weights."""

        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        anchors = self._as_numpy_optional(anchors)
        anchor_features = self._as_numpy_optional(anchor_features)

        local_k = max(0, int(local_k))
        global_r = max(0, int(global_r))
        if local_k + global_r <= 0:
            global_r = int(n_samples)
        if local_k + global_r != int(n_samples):
            global_r = max(0, int(n_samples) - local_k)
            local_k = min(local_k, int(n_samples))

        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        weights = np.ones((bsz, n_samples), dtype=np.float32)
        ess = np.ones((bsz,), dtype=np.float64)
        expected_ess = np.ones((bsz,), dtype=np.float64)
        max_weight = np.ones((bsz,), dtype=np.float64)
        mean_weight = np.ones((bsz,), dtype=np.float64)
        entropy = np.ones((bsz,), dtype=np.float64)
        sampled_rank = np.zeros((bsz,), dtype=np.float64)
        local_frac = np.zeros((bsz,), dtype=np.float64)

        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
                continue

            anchor = anchors[i] if anchors is not None else None
            anchor_feature = anchor_features[i] if anchor_features is not None else None
            dist_pool, dist_anchor = self._distance_inputs(lbl, valid, anchor, anchor_feature)

            local_idx = np.empty((0,), dtype=np.int32)
            rank_lookup = np.arange(valid, dtype=np.float64)
            if local_k > 0 and dist_pool is not None and dist_anchor is not None:
                flat_candidates = dist_pool.astype(np.float32, copy=False).reshape(valid, -1)
                diff = flat_candidates - dist_anchor.astype(np.float32, copy=False).reshape(1, -1)
                distances = np.mean(diff * diff, axis=1)
                order = np.argsort(distances, kind="stable")
                rank_lookup = np.empty((valid,), dtype=np.float64)
                rank_lookup[order] = np.arange(valid, dtype=np.float64)
                k_eff = min(local_k, valid)
                if local_mode == "farthest":
                    local_idx = order[-k_eff:].astype(np.int32)
                else:
                    local_idx = order[:k_eff].astype(np.int32)

            remaining = int(n_samples) - int(local_idx.shape[0])
            if remaining > 0:
                global_idx = np.random.choice(valid, remaining, replace=(valid < remaining)).astype(np.int32)
            else:
                global_idx = np.empty((0,), dtype=np.int32)

            idx = np.concatenate([local_idx, global_idx], axis=0)
            if idx.shape[0] < n_samples:
                extra = np.random.choice(valid, int(n_samples) - idx.shape[0], replace=True).astype(np.int32)
                idx = np.concatenate([idx, extra], axis=0)
            elif idx.shape[0] > n_samples:
                idx = idx[:n_samples]
            np.random.shuffle(idx)

            proposal = np.zeros((valid,), dtype=np.float64)
            local_count = int(local_idx.shape[0])
            global_count = int(n_samples) - local_count
            if local_count > 0:
                proposal[local_idx] += (local_count / float(n_samples)) / float(local_count)
            if global_count > 0:
                proposal += (global_count / float(n_samples)) / float(valid)
            if proposal.sum() <= 0.0:
                proposal[:] = 1.0 / valid
            proposal = proposal / proposal.sum()

            target_prob = 1.0 / valid
            row_weights = target_prob / proposal[idx]
            power = float(weight_power)
            if abs(power - 1.0) > 1e-12:
                row_weights = np.power(np.clip(row_weights, 1e-30, None), power)

            sample_indices[i] = idx.astype(np.int32)
            weights[i] = row_weights.astype(np.float32)
            weight_sum = float(row_weights.sum())
            weight_sq_sum = float(np.sum(row_weights * row_weights))
            ess[i] = (weight_sum * weight_sum / max(weight_sq_sum, 1e-12)) / max(1, n_samples)
            expected_ess[i], proposal_weight_max = _uniform_importance_stats(proposal)
            max_weight[i] = float(row_weights.max()) if row_weights.size else proposal_weight_max
            mean_weight[i] = float(row_weights.mean()) if row_weights.size else 1.0
            entropy_denom = max(float(np.log(valid)), 1e-12)
            entropy[i] = float(-(proposal * np.log(np.clip(proposal, 1e-30, None))).sum() / entropy_denom)
            sampled_rank[i] = float(rank_lookup[idx].mean() / max(1, valid - 1))
            local_frac[i] = local_count / max(1, n_samples)

        out = self.bank[labels[:, None], sample_indices]
        info = {
            "pos_sampler/ess_frac": float(ess.mean()),
            "pos_sampler/expected_ess_frac": float(expected_ess.mean()),
            "pos_sampler/weight_max": float(max_weight.mean()),
            "pos_sampler/weight_mean": float(mean_weight.mean()),
            "pos_sampler/proposal_entropy": float(entropy.mean()),
            "pos_sampler/sampled_rank_frac": float(sampled_rank.mean()),
            "pos_sampler/local_frac": float(local_frac.mean()),
            "pos_sampler/local_k": float(local_k),
            "pos_sampler/global_r": float(global_r),
            "pos_sampler/weight_power": float(weight_power),
        }
        return torch.from_numpy(out), torch.from_numpy(weights), info

    def sample_importance_stratified(
        self,
        labels,
        n_samples: int,
        *,
        anchors=None,
        anchor_features=None,
        local_alpha: float = 1.0,
        strata_edges=None,
        rank_temperatures=None,
        score_floor: float = 0.05,
        allocation=None,
        min_per_stratum: int = 1,
        min_ess_frac: float | None = None,
        max_importance_weight: float | None = None,
        weight_clip: float | None = None,
        weight_power: float = 1.0,
        normalize_weights: bool = False,
    ):
        """Sample rank strata with exact p/q weights for a uniform class target."""

        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        anchors = self._as_numpy_optional(anchors)
        anchor_features = self._as_numpy_optional(anchor_features)

        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        weights = np.ones((bsz, n_samples), dtype=np.float32)
        ess = np.ones((bsz,), dtype=np.float64)
        max_weight = np.ones((bsz,), dtype=np.float64)
        mean_weight = np.ones((bsz,), dtype=np.float64)
        entropy = np.ones((bsz,), dtype=np.float64)
        sampled_rank = np.zeros((bsz,), dtype=np.float64)
        effective_alpha = np.zeros((bsz,), dtype=np.float64)
        expected_ess = np.ones((bsz,), dtype=np.float64)
        proposal_weight_max = np.ones((bsz,), dtype=np.float64)
        stratum_alloc = np.zeros((bsz, 4), dtype=np.float64)

        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
                continue

            anchor = anchors[i] if anchors is not None else None
            anchor_feature = anchor_features[i] if anchor_features is not None else None
            dist_pool, dist_anchor = self._distance_inputs(lbl, valid, anchor, anchor_feature)
            if dist_pool is None or dist_anchor is None:
                proposal = np.full((valid,), 1.0 / valid, dtype=np.float64)
                rank_lookup = np.arange(valid, dtype=np.float64)
            else:
                flat_candidates = dist_pool.astype(np.float32, copy=False).reshape(valid, -1)
                diff = flat_candidates - dist_anchor.astype(np.float32, copy=False).reshape(1, -1)
                distances = np.mean(diff * diff, axis=1)
                proposal, proposal_stats = rank_stratified_proposal_with_stats(
                    distances,
                    n_samples=n_samples,
                    local_alpha=local_alpha,
                    strata_edges=strata_edges,
                    rank_temperatures=rank_temperatures,
                    score_floor=score_floor,
                    allocation=allocation,
                    min_per_stratum=min_per_stratum,
                    min_ess_frac=min_ess_frac,
                    max_importance_weight=max_importance_weight,
                )
                effective_alpha[i] = proposal_stats["effective_alpha"]
                expected_ess[i] = proposal_stats["expected_ess_frac"]
                proposal_weight_max[i] = proposal_stats["proposal_weight_max"]
                for h in range(4):
                    stratum_alloc[i, h] = proposal_stats.get(f"stratum_alloc_{h}", 0.0)
                order = np.argsort(distances, kind="stable")
                rank_lookup = np.empty((valid,), dtype=np.float64)
                rank_lookup[order] = np.arange(valid, dtype=np.float64)

            idx = np.random.choice(valid, n_samples, replace=True, p=proposal)
            target_prob = 1.0 / valid
            row_weights = target_prob / proposal[idx]
            if weight_clip is not None and float(weight_clip) > 0.0:
                row_weights = np.minimum(row_weights, float(weight_clip))
            power = float(weight_power)
            if abs(power - 1.0) > 1e-12:
                row_weights = np.power(np.clip(row_weights, 1e-30, None), power)
            if normalize_weights:
                row_weights = row_weights / max(float(row_weights.mean()), 1e-8)

            sample_indices[i] = idx.astype(np.int32)
            weights[i] = row_weights.astype(np.float32)
            weight_sum = float(row_weights.sum())
            weight_sq_sum = float(np.sum(row_weights * row_weights))
            ess[i] = (weight_sum * weight_sum / max(weight_sq_sum, 1e-12)) / max(1, n_samples)
            max_weight[i] = float(row_weights.max())
            mean_weight[i] = float(row_weights.mean())
            entropy_denom = max(float(np.log(valid)), 1e-12)
            entropy[i] = float(-(proposal * np.log(np.clip(proposal, 1e-30, None))).sum() / entropy_denom)
            sampled_rank[i] = float(rank_lookup[idx].mean() / max(1, valid - 1))

        out = self.bank[labels[:, None], sample_indices]
        info = {
            "pos_sampler/ess_frac": float(ess.mean()),
            "pos_sampler/weight_max": float(max_weight.mean()),
            "pos_sampler/weight_mean": float(mean_weight.mean()),
            "pos_sampler/proposal_entropy": float(entropy.mean()),
            "pos_sampler/sampled_rank_frac": float(sampled_rank.mean()),
            "pos_sampler/local_alpha": float(local_alpha),
            "pos_sampler/effective_alpha": float(effective_alpha.mean()),
            "pos_sampler/expected_ess_frac": float(expected_ess.mean()),
            "pos_sampler/proposal_weight_max": float(proposal_weight_max.mean()),
            "pos_sampler/weight_power": float(weight_power),
        }
        for h in range(4):
            info[f"pos_sampler/stratum_alloc_{h}"] = float(stratum_alloc[:, h].mean())
        return torch.from_numpy(out), torch.from_numpy(weights), info
