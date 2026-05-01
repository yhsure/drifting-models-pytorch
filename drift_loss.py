from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list: Iterable[float] = (0.02, 0.05, 0.2),
):
    b, c_g, s = gen.shape

    if fixed_neg is None:
        fixed_neg = gen[:, :0, :]
    c_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        info = {}
        dist = cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / torch.clamp(targets_w.mean(), min=1e-8)
        info["scale"] = scale

        scale_inputs = torch.clamp(scale / (s**0.5), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=1e-3)

        mask_val = 100.0
        diag_mask = torch.eye(c_g, device=gen.device, dtype=torch.float32)
        block_mask = torch.nn.functional.pad(diag_mask, (0, c_n + fixed_pos.shape[1], 0, 0))
        block_mask = block_mask.unsqueeze(0)
        dist_normed = dist_normed + block_mask * mask_val

        force_across_r = torch.zeros_like(old_gen_scaled)
        split_idx = c_g + c_n

        for r in R_list:
            logits = -dist_normed / float(r)
            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
            affinity = affinity * targets_w[:, None, :]

            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = torch.sum(aff_pos, dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = torch.sum(aff_neg, dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            r_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)
            total_force_r = torch.einsum("biy,byx->bix", r_coeff, targets_scaled)

            total_coeffs = r_coeff.sum(dim=-1)
            total_force_r = total_force_r - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_r**2).mean()

            info[f"loss_{r}"] = f_norm_val
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_r = force_across_r + total_force_r / force_scale

        goal_scaled = old_gen_scaled + force_across_r

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = torch.mean(diff**2, dim=(-1, -2))
    info = {k: v.detach().mean() for k, v in info.items()}
    return loss, info


def _batched_log_prob(p: MixtureSameFamily, x: torch.Tensor) -> torch.Tensor:
    """Evaluate log_prob for x shaped as (..., sample_count, channel_dim)."""
    if x.ndim == 2:
        return p.log_prob(x)
    return p.log_prob(x.movedim(-2, 0)).movedim(0, -1)


def _log_mixture(x: torch.Tensor, p: MixtureSameFamily, q: MixtureSameFamily) -> torch.Tensor:
    return torch.stack([_batched_log_prob(p, x), _batched_log_prob(q, x)], dim=-1).logsumexp(-1) + math.log(0.5)


def _make_mog(locs: torch.Tensor, sigma: torch.Tensor) -> MixtureSameFamily:
    """Uniform MoG from (..., K, D) centers.

    Leading dimensions are independent index axes. K is the sample/mixture axis,
    and D is the Gaussian event/channel axis.
    """
    K = locs.shape[-2]
    return MixtureSameFamily(
        Categorical(logits=torch.zeros(*locs.shape[:-2], K, device=locs.device, dtype=locs.dtype)),
        Independent(Normal(loc=locs, scale=sigma * torch.ones_like(locs)), 1),
    )


def jsd_mog_loss(
    gen_features: torch.Tensor,
    real_features: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """JSD between generated and real MoGs over D-dimensional feature vectors.

    Inputs are shaped as (..., K, D). Leading dimensions are independent index
    axes, K is the sample/mixture axis, and D is the channel dimension.
    """
    D = gen_features.shape[-1]

    gen_f  = gen_features.float()
    real_f = real_features.float()
    sigma_q = sigma
    sigma_p = sigma_q.detach()

    p = _make_mog(real_f.reshape(*real_f.shape[:-1], D), sigma_p)
    q = _make_mog(gen_f.reshape(*gen_f.shape[:-1], D), sigma_q)

    x_p = real_f
    x_q = gen_f + sigma_q * torch.randn_like(gen_f)

    kl_pm = (_batched_log_prob(p, x_p) - _log_mixture(x_p, p, q)).mean()
    kl_qm = (_batched_log_prob(q, x_q) - _log_mixture(x_q, p, q)).mean()
    return 0.5 * (kl_pm + kl_qm)


def likelihood_mog_loss(
    gen_features: torch.Tensor,
    real_features: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """NLL of real feature samples under the generated-feature MoG.

    Inputs are shaped as (..., K, D). Leading dimensions are independent index
    axes, K is the sample/mixture axis, and D is the channel dimension.
    """
    D = gen_features.shape[-1]

    gen_f = gen_features.float()
    real_f = real_features.float()

    q = _make_mog(gen_f.reshape(*gen_f.shape[:-1], D), sigma)
    return -_batched_log_prob(q, real_f).mean()
