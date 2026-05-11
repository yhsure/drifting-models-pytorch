from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.logging import log_for_0


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _as_plain_dict(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _selected_descriptor_keys(activations: Mapping[str, torch.Tensor], include_keys: list[str] | None = None) -> list[str]:
    if include_keys:
        missing = [key for key in include_keys if key not in activations]
        if missing:
            raise KeyError(f"Likelihood descriptor keys missing from activations: {missing}")
        return list(include_keys)

    keys = sorted(k for k in activations if k.endswith("_mean") or k.endswith("_std"))
    if keys:
        return keys
    if "global" in activations:
        return ["global"]
    return sorted(activations.keys())


def compact_descriptors_from_activations(
    activations: Mapping[str, torch.Tensor],
    *,
    include_keys: list[str] | None = None,
) -> torch.Tensor:
    """Pool frozen activation maps into the compact descriptor used by the mixture prior."""
    parts: list[torch.Tensor] = []
    for key in _selected_descriptor_keys(activations, include_keys=include_keys):
        value = activations[key]
        if value.ndim < 2:
            continue
        value = value.float()
        if value.ndim == 2:
            pooled = value
        else:
            pooled = value.reshape(value.shape[0], -1, value.shape[-1]).mean(dim=1)
        parts.append(pooled)
    if not parts:
        raise ValueError("No usable activations were available for likelihood descriptors.")
    return torch.cat(parts, dim=1)


@torch.no_grad()
def compute_likelihood_descriptors(
    samples: torch.Tensor,
    *,
    activation_fn,
    activation_kwargs: Mapping[str, Any],
    descriptor_keys: list[str] | None,
    device: torch.device,
) -> torch.Tensor:
    samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        activations = activation_fn(samples, **dict(activation_kwargs))
    return compact_descriptors_from_activations(activations, include_keys=descriptor_keys).float()


def _standardize_for_fit(raw: torch.Tensor, std_floor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(float(std_floor))
    return (raw - mean) / std, mean, std


@torch.no_grad()
def minibatch_kmeans(
    data: torch.Tensor,
    *,
    component_count: int,
    iterations: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if data.ndim != 2:
        raise ValueError(f"k-means data must be 2D, got shape={tuple(data.shape)}")
    n, dim = data.shape
    k = int(component_count)
    if n <= 0:
        raise ValueError("Cannot fit k-means with no descriptors.")
    data = data.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    init_idx = torch.randint(n, (k,), generator=generator, device=device)
    if n >= k:
        init_idx = torch.randperm(n, generator=generator, device=device)[:k]
    centers = data[init_idx].clone()
    counts = torch.zeros(k, device=device, dtype=torch.float32)
    batch_size = max(1, min(int(batch_size), n))

    for _ in tqdm(range(max(1, int(iterations))), desc="likelihood kmeans", disable=(_rank() != 0)):
        idx = torch.randint(n, (batch_size,), generator=generator, device=device)
        batch = data[idx]
        assign = torch.cdist(batch, centers).argmin(dim=1)
        batch_counts = torch.bincount(assign, minlength=k).float()
        valid = batch_counts > 0
        if not torch.any(valid):
            continue
        sums = torch.zeros(k, dim, device=device, dtype=torch.float32)
        sums.index_add_(0, assign, batch)
        means = sums[valid] / batch_counts[valid, None]
        eta = batch_counts[valid] / (counts[valid] + batch_counts[valid]).clamp_min(1.0)
        centers[valid] = centers[valid] * (1.0 - eta[:, None]) + means * eta[:, None]
        counts[valid] += batch_counts[valid]
    return centers


@torch.no_grad()
def class_log_probs_from_descriptors(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    *,
    sigma: float,
    num_classes: int,
    batch_size: int,
    eps: float,
    device: torch.device,
) -> torch.Tensor:
    descriptors = descriptors.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long).clamp(0, int(num_classes) - 1)
    centers = centers.to(device=device, dtype=torch.float32)
    k = centers.shape[0]
    accum = torch.zeros(int(num_classes), k, device=device, dtype=torch.float32)
    counts = torch.zeros(int(num_classes), device=device, dtype=torch.float32)
    sigma_sq = max(float(sigma), 1e-6) ** 2
    batch_size = max(1, int(batch_size))
    for start in tqdm(range(0, descriptors.shape[0], batch_size), desc="likelihood class weights", disable=(_rank() != 0)):
        end = min(start + batch_size, descriptors.shape[0])
        batch = descriptors[start:end]
        batch_labels = labels[start:end]
        logits = -torch.cdist(batch, centers).square() / (2.0 * sigma_sq)
        resp = torch.softmax(logits, dim=-1)
        accum.index_add_(0, batch_labels, resp)
        counts.index_add_(0, batch_labels, torch.ones_like(batch_labels, dtype=torch.float32))
    weights = accum / counts.clamp_min(1.0)[:, None]
    missing = counts <= 0
    if torch.any(missing):
        weights[missing] = 1.0 / float(k)
    weights = weights + float(eps)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights.log()


@torch.no_grad()
def estimate_feature_sigma(
    descriptors: torch.Tensor,
    centers: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    descriptors = descriptors.to(device=device, dtype=torch.float32)
    centers = centers.to(device=device, dtype=torch.float32)
    dim = max(1, int(centers.shape[1]))
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    batch_size = max(1, int(batch_size))
    for start in tqdm(range(0, descriptors.shape[0], batch_size), desc="likelihood sigma", disable=(_rank() != 0)):
        end = min(start + batch_size, descriptors.shape[0])
        min_dist_sq = torch.cdist(descriptors[start:end], centers).square().min(dim=1).values
        total += min_dist_sq.double().sum()
        count += int(min_dist_sq.numel())
    sigma_sq = (total / max(1, count) / float(dim)).clamp_min(1e-8)
    return float(torch.sqrt(sigma_sq).detach().cpu().item())


@torch.no_grad()
def collect_likelihood_fit_tensors(
    train_loader,
    *,
    preprocess_fn,
    activation_fn,
    activation_kwargs: Mapping[str, Any],
    descriptor_keys: list[str] | None,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    shard_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = train_loader.dataset
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    world = _world_size()
    rank = _rank()
    limit = len(dataset) if int(max_samples) <= 0 else min(int(max_samples), len(dataset))
    if world > 1:
        if shard_dir is None:
            raise ValueError("Distributed likelihood fitting requires shard_dir.")
        if limit < world:
            raise ValueError(
                "Distributed likelihood fitting requires at least one fit sample per rank: "
                f"fit_samples={limit}, world_size={world}."
            )
        if rank == 0:
            shard_dir.mkdir(parents=True, exist_ok=True)
            for path in shard_dir.glob("rank_*.pt"):
                path.unlink(missing_ok=True)
        dist.barrier()
        indices = torch.randperm(len(dataset), generator=generator)[:limit]
        local_indices = indices[rank::world].tolist()
        dataset = Subset(dataset, local_indices)

    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=(world == 1),
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=(generator if world == 1 else None),
        persistent_workers=(int(num_workers) > 0),
    )
    descs: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    total = 0
    for batch in tqdm(loader, desc="likelihood descriptors", disable=(_rank() != 0)):
        processed = preprocess_fn(batch)
        images = processed["images"]
        labels = processed["labels"].long()
        desc = compute_likelihood_descriptors(
            images,
            activation_fn=activation_fn,
            activation_kwargs=activation_kwargs,
            descriptor_keys=descriptor_keys,
            device=device,
        )
        remaining = limit - total
        if desc.shape[0] > remaining:
            desc = desc[:remaining]
            labels = labels[:remaining]
        descs.append(desc.cpu())
        labels_out.append(labels.cpu())
        total += int(desc.shape[0])
        if world == 1 and total >= limit:
            break
    if not descs:
        raise RuntimeError("Likelihood prior fitting did not collect any descriptors.")
    local_desc = torch.cat(descs, dim=0).float()
    local_labels = torch.cat(labels_out, dim=0).long()
    if world == 1:
        return local_desc, local_labels

    assert shard_dir is not None
    shard_path = shard_dir / f"rank_{rank:05d}.pt"
    tmp_path = shard_path.with_suffix(".pt.tmp")
    torch.save({"descriptors": local_desc, "labels": local_labels}, tmp_path)
    tmp_path.replace(shard_path)
    dist.barrier()

    if rank != 0:
        return torch.empty(0, local_desc.shape[1], dtype=torch.float32), torch.empty(0, dtype=torch.long)

    all_descs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for shard_rank in range(world):
        shard = torch.load(shard_dir / f"rank_{shard_rank:05d}.pt", map_location="cpu", weights_only=False)
        all_descs.append(shard["descriptors"].float())
        all_labels.append(shard["labels"].long())
    return torch.cat(all_descs, dim=0), torch.cat(all_labels, dim=0)


def anneal_value(start: float, final: float, step: int, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return float(final)
    alpha = min(max(float(step) / float(anneal_steps), 0.0), 1.0)
    return float(start) + alpha * (float(final) - float(start))


def anneal_int(start: int, final: int, step: int, anneal_steps: int) -> int:
    return int(round(anneal_value(float(start), float(final), step, anneal_steps)))


def _validate_cached_fit(
    payload: Mapping[str, Any],
    *,
    component_count: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    centers = payload["centers"].detach().float()
    feature_mean = payload["feature_mean"].detach().float()
    feature_std = payload["feature_std"].detach().float()
    class_log_probs = payload["class_log_probs"].detach().float()
    sigma = float(payload["sigma"])
    if centers.ndim != 2 or centers.shape[0] != int(component_count):
        raise ValueError(f"Cached likelihood centers have incompatible shape: {tuple(centers.shape)}")
    if feature_mean.shape != (centers.shape[1],) or feature_std.shape != (centers.shape[1],):
        raise ValueError("Cached likelihood feature stats do not match cached center dimension.")
    if class_log_probs.shape != (int(num_classes), int(component_count)):
        raise ValueError(f"Cached likelihood class logits have incompatible shape: {tuple(class_log_probs.shape)}")
    return centers, feature_mean, feature_std, class_log_probs, sigma


def _save_likelihood_fit_cache(
    path: Path,
    *,
    centers: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    class_log_probs: torch.Tensor,
    sigma: float,
    descriptor_keys: list[str] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(
        {
            "centers": centers.detach().cpu(),
            "feature_mean": feature_mean.detach().cpu(),
            "feature_std": feature_std.detach().cpu(),
            "class_log_probs": class_log_probs.detach().cpu(),
            "sigma": float(sigma),
            "descriptor_keys": list(descriptor_keys or []),
        },
        tmp,
    )
    tmp.replace(path)


def _broadcast_likelihood_fit(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float] | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if _world_size() == 1:
        if tensors is None:
            raise RuntimeError("Rank-zero likelihood tensors are missing in single-process mode.")
        return tensors

    if _rank() == 0:
        assert tensors is not None
        centers, feature_mean, feature_std, class_log_probs, sigma = tensors
        shape = torch.tensor(
            [centers.shape[0], centers.shape[1], class_log_probs.shape[0]],
            device=device,
            dtype=torch.long,
        )
        sigma_t = torch.tensor([float(sigma)], device=device, dtype=torch.float32)
    else:
        shape = torch.zeros(3, device=device, dtype=torch.long)
        sigma_t = torch.zeros(1, device=device, dtype=torch.float32)
    dist.broadcast(shape, src=0)
    dist.broadcast(sigma_t, src=0)
    k, dim, num_classes = [int(x) for x in shape.detach().cpu().tolist()]
    if _rank() != 0:
        centers = torch.empty(k, dim, device=device, dtype=torch.float32)
        feature_mean = torch.empty(dim, device=device, dtype=torch.float32)
        feature_std = torch.empty(dim, device=device, dtype=torch.float32)
        class_log_probs = torch.empty(num_classes, k, device=device, dtype=torch.float32)
    else:
        centers = centers.to(device=device)
        feature_mean = feature_mean.to(device=device)
        feature_std = feature_std.to(device=device)
        class_log_probs = class_log_probs.to(device=device)
    for tensor in (centers, feature_mean, feature_std, class_log_probs):
        dist.broadcast(tensor, src=0)
    return centers, feature_mean, feature_std, class_log_probs, float(sigma_t.detach().cpu().item())


def initialize_explicit_likelihood_prior(
    *,
    model,
    train_loader,
    preprocess_fn,
    activation_fn,
    config: Mapping[str, Any] | None,
    default_activation_kwargs: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> bool:
    cfg = _as_plain_dict(config)
    if not bool(cfg.get("enabled", False)):
        return False
    prior_model = _unwrap_model(model)
    if not hasattr(prior_model, "has_likelihood_prior") or prior_model.likelihood_component_count <= 0:
        raise ValueError("likelihood_prior.enabled=true requires model.likelihood_component_count > 0.")
    if prior_model.has_likelihood_prior() and not bool(cfg.get("force_refit", False)):
        log_for_0("Explicit likelihood prior already initialized; skipping fit.")
        return False

    component_count = int(cfg.get("component_count", prior_model.likelihood_component_count))
    if component_count != prior_model.likelihood_component_count:
        raise ValueError(
            f"likelihood_prior.component_count={component_count} does not match "
            f"model.likelihood_component_count={prior_model.likelihood_component_count}"
        )
    sigma = float(cfg.get("feature_sigma", cfg.get("sigma", prior_model.likelihood_sigma)))
    descriptor_keys = cfg.get("descriptor_keys", None)
    descriptor_keys = list(descriptor_keys) if descriptor_keys else None
    activation_kwargs = dict(default_activation_kwargs)
    activation_kwargs.update(dict(cfg.get("activation_kwargs", {})))

    cache_path = str(cfg.get("cache_path", "")).strip()
    cache_file = Path(cache_path).expanduser() if cache_path else None
    fit_tensors = None
    needs_fit = True
    force_refit = bool(cfg.get("force_refit", False))
    if _rank() == 0 and cache_file is not None and cache_file.is_file() and not force_refit:
        try:
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            fit_tensors = _validate_cached_fit(
                payload,
                component_count=component_count,
                num_classes=int(prior_model.num_classes),
            )
            log_for_0("Loaded explicit likelihood prior cache from %s", str(cache_file))
            needs_fit = False
        except (KeyError, ValueError) as exc:
            log_for_0("Ignoring incompatible explicit likelihood prior cache at %s: %s", str(cache_file), exc)
            fit_tensors = None
            needs_fit = True

    if _world_size() > 1:
        fit_flag = torch.tensor([1 if needs_fit else 0], device=device, dtype=torch.long)
        dist.broadcast(fit_flag, src=0)
        needs_fit = bool(int(fit_flag.detach().cpu().item()))

    if needs_fit:
        shard_dir_cfg = str(cfg.get("shard_dir", "")).strip()
        if shard_dir_cfg:
            shard_dir = Path(shard_dir_cfg).expanduser()
        elif cache_file is not None:
            shard_dir = cache_file.parent / "likelihood_prior_shards"
        else:
            shard_dir = None
        raw_desc, labels = collect_likelihood_fit_tensors(
            train_loader,
            preprocess_fn=preprocess_fn,
            activation_fn=activation_fn,
            activation_kwargs=activation_kwargs,
            descriptor_keys=descriptor_keys,
            device=device,
            max_samples=int(cfg.get("fit_samples", max(8192, component_count * 8))),
            batch_size=int(cfg.get("fit_batch_size", cfg.get("dataset_batch_size", 256))),
            num_workers=int(cfg.get("fit_num_workers", 0)),
            seed=seed,
            shard_dir=shard_dir,
        )
        if _rank() == 0:
            std_desc, feature_mean, feature_std = _standardize_for_fit(
                raw_desc,
                std_floor=float(cfg.get("descriptor_std_floor", 1e-4)),
            )
            expected_dim = int(getattr(prior_model, "likelihood_descriptor_dim", 0))
            if expected_dim > 0 and expected_dim != std_desc.shape[1]:
                raise ValueError(
                    f"Configured likelihood_descriptor_dim={expected_dim}, "
                    f"but descriptor extraction produced {std_desc.shape[1]}."
                )
            centers = minibatch_kmeans(
                std_desc,
                component_count=component_count,
                iterations=int(cfg.get("kmeans_iters", 40)),
                batch_size=int(cfg.get("kmeans_batch_size", 2048)),
                seed=seed,
                device=device,
            )
            if sigma <= 0:
                sigma = estimate_feature_sigma(
                    std_desc,
                    centers,
                    batch_size=int(cfg.get("responsibility_batch_size", 512)),
                    device=device,
                )
            class_log_probs = class_log_probs_from_descriptors(
                std_desc,
                labels,
                centers,
                sigma=sigma,
                num_classes=int(prior_model.num_classes),
                batch_size=int(cfg.get("responsibility_batch_size", 512)),
                eps=float(cfg.get("class_weight_eps", 1e-5)),
                device=device,
            )
            fit_tensors = (centers, feature_mean.to(device), feature_std.to(device), class_log_probs, sigma)
            if cache_file is not None:
                _save_likelihood_fit_cache(
                    cache_file,
                    centers=centers,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    class_log_probs=class_log_probs,
                    sigma=sigma,
                    descriptor_keys=descriptor_keys,
                )
                log_for_0("Saved explicit likelihood prior cache to %s", str(cache_file))
            log_for_0(
                "Fitted explicit likelihood prior: K=%d dim=%d samples=%d sigma=%.4g",
                component_count,
                int(std_desc.shape[1]),
                int(std_desc.shape[0]),
                sigma,
            )

    centers, feature_mean, feature_std, class_log_probs, sigma = _broadcast_likelihood_fit(fit_tensors, device=device)
    prior_model.set_likelihood_prior(
        centers=centers,
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_log_probs=class_log_probs,
        sigma=sigma,
    )
    return True


def likelihood_usage_entropy(class_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(class_logits.float(), dim=-1).mean(dim=0)
    return -(probs * probs.clamp_min(1e-8).log()).sum()


def likelihood_balance_loss(resp: torch.Tensor) -> torch.Tensor:
    mean_resp = resp.float().mean(dim=0)
    k = max(1, mean_resp.shape[0])
    return (mean_resp * (mean_resp * k).clamp_min(1e-8).log()).sum()


def likelihood_info(resp: torch.Tensor, topk_mass: float | None = None) -> dict[str, float]:
    entropy = -(resp.float() * resp.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
    out = {"likelihood/posterior_entropy": float(entropy.detach().cpu().item())}
    if topk_mass is not None:
        out["likelihood/topk_mass"] = float(topk_mass)
    return out
