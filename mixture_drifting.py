# %%
"""Explicit-likelihood mixture with a one-step drifting generator on CIFAR-10.

This file is a deliberately separate experiment from ``mixture.py``.  It keeps
the explicit Gaussian mixture in frozen feature space, but replaces the
rectified-flow renderer with a single-pass generator trained by the drifting
loss from ``paper.tex``.

Two conditioning variants are supported:

``hard``
    Sample one component k ~ p(k | v(x)) and condition the generator on e_k.

``soft``
    Use the responsibility-weighted embedding sum_k p(k | v(x)) e_k.

For stronger runs, enable ``--component-drift-weight``.  This switches on the
component-local positive bank: each step selects a small set of mixture
components, draws several generated samples per component, and computes the
drifting loss against real images assigned to the same component.  This is the
mixture analogue of the per-class positive queues used in the drifting paper.
The default global drift path uses current-batch real positives and a uniform
generated-feature negative queue.  It deliberately does not use the earlier
nearest-neighbor positive enhancement.

Run a short comparison on the local GH200:

    .venv/bin/python mixture_drifting.py --variant hard --device cuda --steps 500
    .venv/bin/python mixture_drifting.py --variant soft --device cuda --steps 500
"""

# %%
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mixture import (
    Config as MixtureConfig,
)
from mixture import (
    FeatureGaussianMixture,
    FrozenFeatureEncoder,
    append_metric,
    build_ema_model,
    cycle,
    effective_ema_decay,
    ema_update,
    evaluate_image_samples_against_cifar,
    feature_descriptors,
    make_cifar_likelihood_dataset,
    parse_float_tuple,
    parse_int_tuple,
    resolve_device,
    save_tanh_grid,
    seed_everything,
    single_descriptor_drifting_loss,
)


# %%
@dataclass
class DriftMixtureConfig:
    variant: str = "hard"
    data_dir: str = "data"
    out_dir: str = "runs/cifar_likelihood_drift_v1"
    resume_path: str = ""
    seed: int = 7
    device: str = "auto"
    num_workers: int = 2
    max_train_examples: int = 0
    max_test_examples: int = 0

    steps: int = 2000
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    ema_decay: float = 0.999

    mode_count: int = 128
    cond_dim: int = 128
    unet_dim: int = 64
    unet_dim_mults: str = "1,2,4"
    feature_backbone: str = "resnet18"
    feature_stages: str = "2,4"
    no_std_features: bool = False
    feature_sigma: float = 5.0
    drifting_temps: str = "0.02,0.05,0.2"

    mixture_nll_weight: float = 0.04
    mixture_balance_weight: float = 0.03
    mixture_resp_temp: float = 0.6
    center_weight: float = 0.03
    global_drift_weight: float = 1.0
    class_conditional_drift: bool = False
    drift_negative_queue_size: int = 4096
    drift_negative_queue_samples: int = 256
    drift_include_pixel_descriptor: bool = False
    component_drift_weight: float = 0.0
    active_components: int = 16
    grouped_drift_weight: float = 0.5
    grouped_drift_components: int = 8
    grouped_drift_min_count: int = 4
    cond_dropout: float = 0.1

    log_every: int = 50
    sample_every: int = 500
    save_every: int = 500
    grid_size: int = 64
    preview_source: str = "ema"
    eval_samples: int = 512
    eval_batch_size: int = 256
    eval_nn_samples: int = 512


def parse_args() -> DriftMixtureConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for field_name, field_value in asdict(DriftMixtureConfig()).items():
        arg = "--" + field_name.replace("_", "-")
        value_type = type(field_value)
        if value_type is bool:
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=field_value)
        else:
            parser.add_argument(arg, type=value_type, default=field_value)
    cfg = DriftMixtureConfig(**vars(parser.parse_args()))
    if cfg.variant not in {"hard", "soft"}:
        raise ValueError("--variant must be either 'hard' or 'soft'")
    if cfg.preview_source not in {"model", "ema"}:
        raise ValueError("--preview-source must be either 'model' or 'ema'")
    return cfg


def as_mixture_config(cfg: DriftMixtureConfig) -> MixtureConfig:
    return MixtureConfig(
        pipeline="cifar_likelihood_flow",
        dataset="cifar10",
        data_dir=cfg.data_dir,
        out_dir=cfg.out_dir,
        seed=cfg.seed,
        device=cfg.device,
        num_workers=cfg.num_workers,
        max_train_examples=cfg.max_train_examples,
        max_test_examples=cfg.max_test_examples,
        batch_size=cfg.batch_size,
        mode_count=cfg.mode_count,
        feature_sigma=cfg.feature_sigma,
        feature_backbone=cfg.feature_backbone,
        feature_stages=cfg.feature_stages,
        unet_dim=cfg.unet_dim,
        unet_dim_mults=cfg.unet_dim_mults,
        cond_dim=cfg.cond_dim,
        eval_batch_size=cfg.eval_batch_size,
        eval_nn_samples=cfg.eval_nn_samples,
        flow_eval_samples=cfg.eval_samples,
    )


# %%
class ComponentDriftingGenerator(nn.Module):
    """Single-pass image generator conditioned by mixture components."""

    def __init__(self, dim: int, dim_mults: tuple[int, ...], cond_dim: int, component_count: int):
        super().__init__()
        try:
            from rectified_flow_pytorch import Unet
        except ImportError as exc:
            raise RuntimeError(
                "mixture_drifting.py needs rectified-flow-pytorch for its CIFAR UNet. "
                "Install it with: uv pip install --python .venv/bin/python rectified-flow-pytorch"
            ) from exc
        self.component_count = component_count
        self.null_index = component_count
        self.component_embed = nn.Embedding(component_count + 1, cond_dim)
        self.unet = Unet(
            dim=dim,
            channels=3,
            dim_mults=dim_mults,
            accept_time=True,
            accept_cond=True,
            dim_cond=cond_dim,
        )

    def hard_condition(self, components: torch.Tensor) -> torch.Tensor:
        return self.component_embed(components)

    def soft_condition(self, weights: torch.Tensor) -> torch.Tensor:
        return weights @ self.component_embed.weight[: self.component_count]

    def forward_with_cond(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        fixed_time = torch.zeros(noise.shape[0], device=noise.device, dtype=noise.dtype)
        return torch.tanh(self.unet(noise, fixed_time, cond=cond))

    def forward_hard(self, noise: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cond(noise, self.hard_condition(components))

    def forward_soft(self, noise: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cond(noise, self.soft_condition(weights))

    def null_components(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.full((n,), self.null_index, device=device, dtype=torch.long)


@torch.inference_mode()
def sample_component_drifting_generator(
    model: ComponentDriftingGenerator,
    mixture: FeatureGaussianMixture,
    n: int,
    device: torch.device,
    variant: str = "hard",
    resp_temp: float = 0.6,
) -> torch.Tensor:
    model.eval()
    mixture.eval()
    noise = torch.randn(n, 3, 32, 32, device=device)
    components = mixture.sample_components(n, device)
    if variant == "soft":
        feature_samples = mixture.means[components] + mixture.sigma * torch.randn(
            n,
            mixture.means.shape[1],
            device=device,
        )
        _, resp = mixture.nll_and_responsibilities(feature_samples, resp_temp)
        return model.forward_soft(noise, resp).clamp(-1.0, 1.0)
    return model.forward_hard(noise, components).clamp(-1.0, 1.0)


def compact_cifar_descriptors_grad(images: torch.Tensor, encoder: FrozenFeatureEncoder) -> torch.Tensor:
    feats = encoder(images)
    parts = []
    for stage_id in sorted(feats):
        feat = feats[stage_id]
        parts.append(feat.mean(dim=(2, 3)))
        parts.append(feat.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6))
    parts.append(F.adaptive_avg_pool2d(images, output_size=(4, 4)).flatten(1))
    parts.append(images.mean(dim=(2, 3)))
    parts.append(images.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6))
    return torch.cat(parts, dim=1)


class DescriptorQueue:
    """Uniform FIFO support queue for frozen feature descriptors."""

    def __init__(self, max_size: int):
        self.max_size = max(0, int(max_size))
        self.data: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.data is None else int(self.data.shape[0])

    def push(self, values: torch.Tensor) -> None:
        if self.max_size <= 0:
            return
        values = values.detach().float()
        if values.shape[0] > self.max_size:
            values = values[-self.max_size :]
        if self.data is None:
            self.data = values.contiguous()
            return
        self.data = torch.cat((self.data, values), dim=0)[-self.max_size :].contiguous()

    def sample(self, count: int) -> torch.Tensor | None:
        if count <= 0 or self.data is None or self.data.shape[0] == 0:
            return None
        idx = torch.randint(self.data.shape[0], (int(count),), device=self.data.device)
        return self.data[idx]


def compute_support_drifting_field(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    tau: float,
    self_negative_count: int = 0,
) -> torch.Tensor:
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)
    if self_negative_count == x.shape[0]:
        diag = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        dist_neg[:, :self_negative_count] = dist_neg[:, :self_negative_count].masked_fill(diag, 1e6)

    logits = torch.cat((-dist_pos / tau, -dist_neg / tau), dim=1)
    aff_row = logits.softmax(dim=-1)
    aff_col = logits.softmax(dim=-2)
    affinity = torch.sqrt((aff_row * aff_col).clamp_min(1e-12))

    n_pos = y_pos.shape[0]
    a_pos = affinity[:, :n_pos]
    a_neg = affinity[:, n_pos:]
    w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)
    return w_pos @ y_pos - w_neg @ y_neg


def single_descriptor_support_drifting_loss(
    generated: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temps: tuple[float, ...],
    self_negative_count: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    dim = generated.shape[1]
    sqrt_dim = dim**0.5

    support = torch.cat((positives, negatives), dim=0)
    scale = (torch.cdist(generated, support).mean() / sqrt_dim).detach().clamp_min(1e-6)
    generated_norm = generated / scale
    positives_norm = positives / scale
    negatives_norm = negatives / scale

    force = torch.zeros_like(generated_norm)
    info: dict[str, float] = {"scale": float(scale.item())}
    for tau in temps:
        tau_eff = float(tau) * sqrt_dim
        v = compute_support_drifting_field(
            generated_norm,
            positives_norm,
            negatives_norm,
            tau=tau_eff,
            self_negative_count=self_negative_count,
        )
        v_norm = v.square().mean().clamp_min(1e-8).sqrt()
        force = force + v / v_norm
        info[f"force_norm_tau_{tau}"] = float(v_norm.detach().item())

    target = (generated_norm + force).detach()
    return F.mse_loss(generated_norm, target), info


def descriptor_drifting_loss(
    generated: torch.Tensor,
    real: torch.Tensor,
    encoder: FrozenFeatureEncoder,
    temps: tuple[float, ...],
    use_std: bool,
    groups: list[torch.Tensor] | None = None,
    negative_queues: list[DescriptorQueue] | None = None,
    negative_queue_samples: int = 0,
    include_pixel_descriptor: bool = False,
    update_queues: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    gen_descs = feature_descriptors(
        generated,
        encoder=encoder,
        use_std=use_std,
        include_pixel_descriptor=include_pixel_descriptor,
    )
    with torch.no_grad():
        real_descs = feature_descriptors(
            real,
            encoder=encoder,
            use_std=use_std,
            include_pixel_descriptor=include_pixel_descriptor,
        )

    total = torch.zeros((), device=generated.device)
    scales = []
    if not groups:
        neg_queue_used = 0
        for desc_idx, (gen_desc, real_desc) in enumerate(zip(gen_descs, real_descs, strict=True)):
            negatives = gen_desc.detach().float()
            self_negative_count = gen_desc.shape[0]
            if negative_queues is not None and negative_queue_samples > 0:
                queued = negative_queues[desc_idx].sample(negative_queue_samples)
                if queued is not None:
                    neg_queue_used += int(queued.shape[0])
                    negatives = torch.cat((negatives, queued.to(gen_desc.device)), dim=0)
            loss_i, info_i = single_descriptor_support_drifting_loss(
                gen_desc.float(),
                real_desc.float(),
                negatives,
                temps=temps,
                self_negative_count=self_negative_count,
            )
            total = total + loss_i
            scales.append(info_i["scale"])
        if update_queues and negative_queues is not None:
            for queue, gen_desc in zip(negative_queues, gen_descs, strict=True):
                queue.push(gen_desc)
        return total, {
            "scale": float(sum(scales) / max(1, len(scales))),
            "neg_queue_used": float(neg_queue_used),
        }

    used = 0
    for idx in groups:
        if idx.numel() < 2:
            continue
        used += 1
        for gen_desc, real_desc in zip(gen_descs, real_descs, strict=True):
            loss_i, info_i = single_descriptor_drifting_loss(
                gen_desc[idx].float(),
                real_desc[idx].float(),
                temps=temps,
            )
            total = total + loss_i
            scales.append(info_i["scale"])
    if update_queues and negative_queues is not None:
        for queue, gen_desc in zip(negative_queues, gen_descs, strict=True):
            queue.push(gen_desc)
    if used == 0:
        return torch.zeros((), device=generated.device), {"scale": 0.0, "neg_queue_used": 0.0}
    return total / float(used), {"scale": float(sum(scales) / max(1, len(scales))), "neg_queue_used": 0.0}


def largest_component_groups(
    components: torch.Tensor,
    component_count: int,
    max_groups: int,
    min_count: int,
) -> list[torch.Tensor]:
    counts = torch.bincount(components.clamp_max(component_count - 1), minlength=component_count)
    valid = torch.nonzero(counts >= min_count, as_tuple=False).flatten()
    if valid.numel() == 0:
        return []
    order = counts[valid].argsort(descending=True)
    chosen = valid[order[:max_groups]]
    return [torch.nonzero(components == int(k), as_tuple=False).flatten() for k in chosen]


def label_groups(labels: torch.Tensor, min_count: int = 2) -> list[torch.Tensor]:
    groups = []
    for label in labels.unique(sorted=False):
        idx = torch.nonzero(labels == label, as_tuple=False).flatten()
        if idx.numel() >= min_count:
            groups.append(idx)
    return groups


class ComponentPositiveBank:
    """Fast sampler for real images assigned to each frozen-feature component."""

    def __init__(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        initial_means: torch.Tensor,
        device: torch.device,
        min_pool_size: int = 1,
    ):
        self.device = device
        self.images = images.to(device)
        self.component_count = int(initial_means.shape[0])
        with torch.no_grad():
            labels = torch.cdist(features.to(device).float(), initial_means.to(device).float()).argmin(dim=1)
        global_indices = torch.arange(labels.shape[0], device=device)
        self.pools: list[torch.Tensor] = []
        for component in range(self.component_count):
            pool = torch.nonzero(labels == component, as_tuple=False).flatten()
            if pool.numel() < min_pool_size:
                pool = global_indices
            self.pools.append(pool)

    def sample(self, components: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            components.shape[0],
            *self.images.shape[1:],
            device=self.device,
            dtype=self.images.dtype,
        )
        for component in components.unique(sorted=False):
            mask = components == component
            count = int(mask.sum().item())
            pool = self.pools[int(component.item()) % self.component_count]
            chosen = pool[torch.randint(pool.numel(), (count,), device=self.device)]
            out[mask] = self.images[chosen]
        return out


def sample_balanced_component_batch(
    mixture: FeatureGaussianMixture,
    batch_size: int,
    active_components: int,
    device: torch.device,
) -> torch.Tensor:
    component_count = mixture.component_count
    group_count = max(1, min(active_components, component_count, batch_size // 2))
    probs = mixture.logits.detach().softmax(dim=0)
    replace = group_count > component_count
    active = torch.multinomial(probs, group_count, replacement=replace).to(device)
    repeats = math.ceil(batch_size / group_count)
    components = active.repeat_interleave(repeats)[:batch_size]
    return components[torch.randperm(components.shape[0], device=device)]


def apply_condition_dropout(
    model: ComponentDriftingGenerator,
    components: torch.Tensor,
    resp: torch.Tensor,
    cfg: DriftMixtureConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.cond_dropout <= 0:
        return components, resp
    drop = torch.rand_like(components.float()) < cfg.cond_dropout
    dropped_components = torch.where(drop, model.null_components(components.shape[0], components.device), components)
    dropped_resp = resp.clone()
    dropped_resp[drop] = 0.0
    return dropped_components, dropped_resp


# %%
def train(cfg: DriftMixtureConfig, device: torch.device, out_dir: Path) -> None:
    mixture_cfg = as_mixture_config(cfg)
    dataset, feature_mean, feature_std, initial_means = make_cifar_likelihood_dataset(mixture_cfg, device, out_dir)
    image_tensor, _, feature_tensor = dataset.tensors
    positive_bank = ComponentPositiveBank(
        image_tensor,
        feature_tensor,
        initial_means,
        device,
        min_pool_size=max(2, cfg.grouped_drift_min_count),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    batches = cycle(loader)
    temps = parse_float_tuple(cfg.drifting_temps)

    model = ComponentDriftingGenerator(
        dim=cfg.unet_dim,
        dim_mults=parse_int_tuple(cfg.unet_dim_mults),
        cond_dim=cfg.cond_dim,
        component_count=cfg.mode_count,
    ).to(device)
    mixture = FeatureGaussianMixture(initial_means.to(device), sigma=cfg.feature_sigma).to(device)
    ema_model = build_ema_model(model).to(device)
    drift_encoder = FrozenFeatureEncoder(
        stage_ids=parse_int_tuple(cfg.feature_stages),
        backbone_name=cfg.feature_backbone,
    ).to(device).eval()
    likelihood_encoder = FrozenFeatureEncoder(stage_ids=(3, 4), backbone_name=cfg.feature_backbone).to(device).eval()
    negative_queues: list[DescriptorQueue] | None = None
    if cfg.drift_negative_queue_size > 0:
        with torch.no_grad():
            probe_descs = feature_descriptors(
                image_tensor[:1].to(device),
                encoder=drift_encoder,
                use_std=not cfg.no_std_features,
                include_pixel_descriptor=cfg.drift_include_pixel_descriptor,
            )
        negative_queues = [DescriptorQueue(cfg.drift_negative_queue_size) for _ in probe_descs]

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr},
            {"params": mixture.parameters(), "lr": cfg.lr},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    start_step = 0
    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
        mixture.load_state_dict(ckpt["mixture"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed {cfg.resume_path} at step {start_step}")

    metrics_path = out_dir / "metrics.jsonl"
    loss_ema = None
    pbar = tqdm(range(start_step + 1, cfg.steps + 1), desc=f"likelihood_drift_{cfg.variant}")
    for step in pbar:
        real, labels, feat = next(batches)
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feat = feat.to(device, non_blocking=True)

        nll, resp = mixture.nll_and_responsibilities(feat, cfg.mixture_resp_temp)
        nll_per_dim = nll / feat.shape[1]
        mean_resp = resp.mean(dim=0)
        balance = (mean_resp * (mean_resp * cfg.mode_count).clamp_min(1e-8).log()).sum()
        sampled_components = torch.multinomial(resp.detach(), num_samples=1).squeeze(1)
        dropped_components, dropped_resp = apply_condition_dropout(model, sampled_components, resp.detach(), cfg)

        global_drift = torch.zeros((), device=device)
        drift_info = {"scale": 0.0, "neg_queue_used": 0.0}
        generated = None
        if cfg.global_drift_weight > 0:
            noise = torch.randn_like(real)
            if cfg.variant == "hard":
                generated = model.forward_hard(noise, dropped_components)
            else:
                generated = model.forward_soft(noise, dropped_resp)

            global_groups = label_groups(labels) if cfg.class_conditional_drift else None
            global_drift, drift_info = descriptor_drifting_loss(
                generated=generated,
                real=real,
                encoder=drift_encoder,
                temps=temps,
                use_std=not cfg.no_std_features,
                groups=global_groups,
                negative_queues=negative_queues,
                negative_queue_samples=cfg.drift_negative_queue_samples,
                include_pixel_descriptor=cfg.drift_include_pixel_descriptor,
                update_queues=True,
            )

        component_drift = torch.zeros((), device=device)
        component_center_loss = torch.zeros((), device=device)
        component_count = 0
        if cfg.component_drift_weight > 0:
            component_batch = sample_balanced_component_batch(
                mixture,
                cfg.batch_size,
                cfg.active_components,
                device,
            )
            positive_images = positive_bank.sample(component_batch)
            local_components, _ = apply_condition_dropout(
                model,
                component_batch,
                F.one_hot(component_batch, num_classes=cfg.mode_count).float(),
                cfg,
            )
            local_generated = model.forward_hard(torch.randn_like(positive_images), local_components)
            local_groups = [
                torch.nonzero(component_batch == int(component), as_tuple=False).flatten()
                for component in component_batch.unique(sorted=False)
            ]
            component_drift, local_info = descriptor_drifting_loss(
                generated=local_generated,
                real=positive_images,
                encoder=drift_encoder,
                temps=temps,
                use_std=not cfg.no_std_features,
                groups=local_groups,
                include_pixel_descriptor=cfg.drift_include_pixel_descriptor,
            )
            component_count = len(local_groups)
            drift_info = local_info if drift_info["scale"] == 0.0 else drift_info
            if generated is None:
                generated = local_generated
            if cfg.center_weight > 0:
                local_feat = compact_cifar_descriptors_grad(local_generated, likelihood_encoder)
                local_feat = (local_feat - feature_mean.to(device)) / feature_std.to(device)
                local_target = mixture.means.detach()[component_batch.clamp_max(cfg.mode_count - 1)]
                component_center_loss = F.mse_loss(local_feat.float(), local_target.float())

        grouped_drift = torch.zeros((), device=device)
        group_count = 0
        if generated is not None and cfg.variant == "hard" and cfg.grouped_drift_weight > 0:
            groups = largest_component_groups(
                sampled_components,
                cfg.mode_count,
                cfg.grouped_drift_components,
                cfg.grouped_drift_min_count,
            )
            grouped_drift, _ = descriptor_drifting_loss(
                generated=generated,
                real=real,
                encoder=drift_encoder,
                temps=temps,
                use_std=not cfg.no_std_features,
                groups=groups,
                include_pixel_descriptor=cfg.drift_include_pixel_descriptor,
            )
            group_count = len(groups)

        center_loss = torch.zeros((), device=device)
        if generated is not None and cfg.center_weight > 0:
            gen_feat = compact_cifar_descriptors_grad(generated, likelihood_encoder)
            gen_feat = (gen_feat - feature_mean.to(device)) / feature_std.to(device)
            if cfg.variant == "hard":
                hard_components = sampled_components.clamp_max(cfg.mode_count - 1)
                target_feat = mixture.means.detach()[hard_components]
            else:
                target_feat = resp.detach() @ mixture.means.detach()
            center_loss = F.mse_loss(gen_feat.float(), target_feat.float())
        center_loss = center_loss + component_center_loss

        loss = (
            cfg.global_drift_weight * global_drift
            + cfg.component_drift_weight * component_drift
            + cfg.grouped_drift_weight * grouped_drift
            + cfg.center_weight * center_loss
            + cfg.mixture_nll_weight * nll_per_dim
            + cfg.mixture_balance_weight * balance
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *mixture.parameters()], cfg.grad_clip)
        opt.step()
        ema_update(ema_model, model, decay=effective_ema_decay(cfg.ema_decay, step))

        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        pbar.set_postfix(loss=f"{loss_ema:.3e}", nll=f"{float(nll_per_dim.detach().cpu()):.3f}")

        metric = {
            "step": step,
            "variant": cfg.variant,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "global_drift": float(global_drift.detach().cpu()),
            "component_drift": float(component_drift.detach().cpu()),
            "grouped_drift": float(grouped_drift.detach().cpu()),
            "center_loss": float(center_loss.detach().cpu()),
            "mixture_nll_per_dim": float(nll_per_dim.detach().cpu()),
            "balance": float(balance.detach().cpu()),
            "usage_entropy": float(mixture.usage_entropy().detach().cpu()),
            "hard_unique": int(sampled_components.unique().numel()),
            "class_group_count": int(len(label_groups(labels)) if cfg.class_conditional_drift else 0),
            "group_count": int(group_count),
            "component_count": int(component_count),
            "scale": drift_info["scale"],
            "neg_queue_used": drift_info.get("neg_queue_used", 0.0),
            "neg_queue_size": int(len(negative_queues[0]) if negative_queues else 0),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)

        if do_sample:
            sample_model = ema_model if cfg.preview_source == "ema" else model
            samples = sample_component_drifting_generator(
                sample_model,
                mixture,
                cfg.grid_size,
                device,
                variant=cfg.variant,
                resp_temp=cfg.mixture_resp_temp,
            )
            save_tanh_grid(samples.cpu(), out_dir / f"samples_step{step:06d}.png", nrow=int(math.sqrt(cfg.grid_size)))

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "mixture": mixture.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "config": asdict(cfg),
                },
                out_dir / f"cifar_likelihood_drift_{cfg.variant}.pt",
            )

        if do_log or do_save:
            append_metric(metrics_path, metric)

    sample_model = ema_model if cfg.preview_source == "ema" else model
    final = sample_component_drifting_generator(
        sample_model,
        mixture,
        cfg.grid_size,
        device,
        variant=cfg.variant,
        resp_temp=cfg.mixture_resp_temp,
    )
    save_tanh_grid(final.cpu(), out_dir / "samples_final.png", nrow=int(math.sqrt(cfg.grid_size)))
    if cfg.eval_samples > 0:
        eval_samples = sample_component_drifting_generator(
            sample_model,
            mixture,
            cfg.eval_samples,
            device,
            variant=cfg.variant,
            resp_temp=cfg.mixture_resp_temp,
        )
        metrics = evaluate_image_samples_against_cifar(
            eval_samples,
            mixture_cfg,
            device,
            out_dir,
            f"drift_{cfg.variant}",
        )
        print(json.dumps(metrics, indent=2))


# %%
def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    stamp = time.strftime("%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / f"{stamp}_{cfg.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"device={device}; out_dir={out_dir}")
    train(cfg, device, out_dir)
    print(f"Done. Samples, metrics, and checkpoints are in {out_dir}")


if __name__ == "__main__":
    main()
