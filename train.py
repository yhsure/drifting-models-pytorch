from __future__ import annotations

import argparse
import copy
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from einops import rearrange, repeat
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset.dataset import epoch0_sampler, get_postprocess_fn, infinite_sampler
from drift_loss import drift_loss
from memory_bank import ArrayMemoryBank
from models.mae_model import build_activation_function
from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.fid_util_ddp import evaluate_fid_ddp
from utils.hsdp_util import merge_data, set_global_mesh
from utils.init_util import maybe_init_state_params
from utils.logging import is_rank_zero, log_for_0
from utils.misc import load_config, maybe_compile, profile_func, run_init, sanitize_train_config, stamp_workdir
from utils.model_builder import build_model_dict

run_init()


@dataclass
class TrainState:
    step: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    ema_model: torch.nn.Module | None = None
    ema_decay: float = 0.999
    device: torch.device | None = None


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(torch.cuda.current_device())
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _generator_model_config(model) -> dict:
    model = _unwrap_model(model)
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "model_config"):
        return dict(model.model_config)
    return {}


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _linear_schedule(start: float, end: float, step: int, steps: int, start_step: int = 0) -> float:
    if steps <= 0:
        return float(end)
    frac = (int(step) - int(start_step)) / float(steps)
    frac = min(1.0, max(0.0, frac))
    return float(start) + frac * (float(end) - float(start))


def _sync_after_rank0_io(workdir: str, tag: str, timeout_s: float = 3600.0) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return

    sync_dir = Path(workdir).resolve() / "log" / "sync"
    marker = sync_dir / f"{tag}.done"
    if _rank() == 0:
        sync_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()), encoding="utf-8")
        return

    start = time.time()
    while not marker.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for rank-0 IO marker: {marker}")
        time.sleep(2.0)


def _make_bank_sync_group(scope: str):
    if not (dist.is_available() and dist.is_initialized()) or _world_size() <= 1:
        return None
    scope = str(scope or "none").strip().lower()
    if scope in {"", "none", "false", "0"}:
        return None
    if scope == "global":
        return dist.group.WORLD
    if scope != "node":
        raise ValueError(f"Unsupported pos_sampling.sync_bank_scope={scope!r}; expected none, node, or global.")

    world_size = _world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
    if local_world_size <= 0:
        local_world_size = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    local_world_size = min(max(1, local_world_size), world_size)
    rank = _rank()
    selected_group = None
    for start in range(0, world_size, local_world_size):
        ranks = list(range(start, min(start + local_world_size, world_size)))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            selected_group = group
    return selected_group


def _all_gather_tensor_for_bank(tensor: torch.Tensor, *, device: torch.device, group=None) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()) or _world_size() <= 1:
        return tensor.detach().cpu()
    gather_input = tensor.detach().to(device=device, non_blocking=True).contiguous()
    group_size = dist.get_world_size(group=group)
    gathered = [torch.empty_like(gather_input) for _ in range(group_size)]
    dist.all_gather(gathered, gather_input, group=group)
    return torch.cat(gathered, dim=0).cpu()


def _local_world_size() -> int:
    world_size = _world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
    if local_world_size <= 0:
        local_world_size = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    return min(max(1, local_world_size), world_size)


def _bank_checkpoint_scope(sync_scope: str, checkpoint_scope: str = "auto") -> str:
    checkpoint_scope = str(checkpoint_scope or "auto").strip().lower()
    if checkpoint_scope == "auto":
        sync_scope = str(sync_scope or "none").strip().lower()
        if sync_scope in {"global", "node"}:
            return sync_scope
        return "rank"
    if checkpoint_scope in {"none", "false", "0"}:
        return "rank"
    if checkpoint_scope not in {"rank", "node", "global"}:
        raise ValueError(
            f"Unsupported bank checkpoint scope={checkpoint_scope!r}; expected auto, rank, node, or global."
        )
    return checkpoint_scope


def _bank_checkpoint_identity(scope: str) -> tuple[str, int]:
    scope = str(scope).strip().lower()
    if scope == "global":
        return "global", 0
    if scope == "node":
        return "node", _rank() // _local_world_size()
    return "rank", _rank()


def _bank_checkpoint_path(workdir: str, step: int, scope: str) -> Path:
    kind, ident = _bank_checkpoint_identity(scope)
    return Path(workdir).resolve() / "memory_banks" / f"step_{int(step):09d}_banks_{kind}_{ident:03d}.pt"


def _bank_checkpoint_pattern(workdir: str, scope: str) -> str:
    kind, ident = _bank_checkpoint_identity(scope)
    return f"step_*_banks_{kind}_{ident:03d}.pt"


def _bank_checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("step_").split("_", 1)[0])
    except (IndexError, ValueError):
        return -1


def _latest_bank_checkpoint(workdir: str, scope: str, step: int) -> Path | None:
    bank_dir = Path(workdir).resolve() / "memory_banks"
    if not bank_dir.exists():
        return None
    candidates = [
        p
        for p in bank_dir.glob(_bank_checkpoint_pattern(workdir, scope))
        if _bank_checkpoint_step(p) <= int(step)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_bank_checkpoint_step)[-1]


def _restore_memory_banks(
    *,
    workdir: str,
    step: int,
    scope: str,
    positive_bank: ArrayMemoryBank,
    negative_bank: ArrayMemoryBank,
) -> bool:
    path = _latest_bank_checkpoint(workdir, scope, step)
    if path is None:
        log_for_0("No memory-bank checkpoint found for scope=%s at step<=%d", scope, int(step))
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    positive_bank.load_state_dict(payload["positive"])
    negative_bank.load_state_dict(payload["negative"])
    log_for_0("Restored memory banks from %s", str(path))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return True


def _save_memory_banks(
    *,
    workdir: str,
    step: int,
    scope: str,
    positive_bank: ArrayMemoryBank,
    negative_bank: ArrayMemoryBank,
    keep: int = 2,
) -> None:
    kind, ident = _bank_checkpoint_identity(scope)
    owner = scope == "rank" or (scope == "global" and _rank() == 0) or (
        scope == "node" and int(os.environ.get("LOCAL_RANK", "0") or 0) == 0
    )
    if owner:
        path = _bank_checkpoint_path(workdir, step, scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": int(step),
                "scope": scope,
                "identity": {"kind": kind, "id": ident},
                "positive": positive_bank.state_dict(),
                "negative": negative_bank.state_dict(),
            },
            path,
        )
        keep = int(keep)
        if keep > 0:
            ckpts = sorted(path.parent.glob(_bank_checkpoint_pattern(workdir, scope)), key=_bank_checkpoint_step)
            for old in ckpts[: max(0, len(ckpts) - keep)]:
                old.unlink(missing_ok=True)
        log_for_0("Saved memory banks to %s", str(path))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _strip_compile_prefix(name: str) -> str:
    return name.removeprefix("_orig_mod.")


def _named_module_tensors(model: torch.nn.Module):
    model = _unwrap_model(model)
    params = {_strip_compile_prefix(k): v for k, v in model.named_parameters()}
    buffers = {_strip_compile_prefix(k): v for k, v in model.named_buffers()}
    return params, buffers


def _update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params, ema_buffers = _named_module_tensors(ema_model)
    model_params, model_buffers = _named_module_tensors(model)

    if ema_params.keys() != model_params.keys():
        missing = sorted(ema_params.keys() - model_params.keys())
        extra = sorted(model_params.keys() - ema_params.keys())
        raise RuntimeError(f"EMA/model parameter mismatch: missing={missing[:5]}, extra={extra[:5]}")

    with torch.inference_mode():
        for name, ema_param in ema_params.items():
            param = model_params[name].detach().to(device=ema_param.device, dtype=ema_param.dtype)
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)

        for name, ema_buffer in ema_buffers.items():
            if name not in model_buffers:
                continue
            buffer = model_buffers[name].detach().to(device=ema_buffer.device, dtype=ema_buffer.dtype)
            ema_buffer.copy_(buffer)


def train_step(
    state: TrainState,
    labels,
    samples,
    negative_samples,
    positive_weights,
    negative_weights,
    feature_params,
    feature_apply,
    learning_rate_fn: Any = None,
    cfg_min=1.0,
    cfg_max=4.0,
    neg_cfg_pw=1.0,
    no_cfg_frac=0.0,
    gen_per_label=8,
    cfg_negative_count: int | None = None,
    activation_kwargs=dict(),
    loss_kwargs=dict(R_list=[0.02, 0.05, 0.2]),
    max_grad_norm=2.0,
):
    device = state.device
    labels = labels.to(device=device, dtype=torch.long)
    samples = samples.to(device=device, dtype=torch.float32)
    negative_samples = negative_samples.to(device=device, dtype=torch.float32)
    if positive_weights is None:
        positive_weights = torch.ones(samples.shape[:2], device=device, dtype=torch.float32)
    else:
        positive_weights = positive_weights.to(device=device, dtype=torch.float32)
    if negative_weights is None:
        negative_weights = torch.zeros(negative_samples.shape[:2], device=device, dtype=torch.float32)
    else:
        negative_weights = negative_weights.to(device=device, dtype=torch.float32)

    bsz = labels.shape[0]

    frac = torch.rand((bsz,), device=device)
    pw = 1 - neg_cfg_pw
    if abs(pw) < 1e-6:
        log_cfg_min = torch.log(torch.tensor(cfg_min, device=device))
        log_cfg_max = torch.log(torch.tensor(cfg_max, device=device))
        cfg = torch.exp(log_cfg_min + frac * (log_cfg_max - log_cfg_min))
    else:
        cfg = (cfg_min**pw + frac * (cfg_max**pw - cfg_min**pw)) ** (1 / pw)

    frac2 = torch.rand((bsz,), device=device)
    cfg = torch.where(frac2 < no_cfg_frac, torch.ones_like(cfg), cfg)

    n_pos, n_gen, n_uncond = samples.shape[1], gen_per_label, negative_samples.shape[1]
    if cfg_negative_count is None:
        cfg_negative_count = n_uncond
    cfg_negative_count = max(0, min(int(cfg_negative_count), n_uncond))
    neg_mass = negative_weights.clone()
    if cfg_negative_count > 0:
        uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, cfg_negative_count)
        neg_mass[:, :cfg_negative_count] = neg_mass[:, :cfg_negative_count] + uncond_w[:, None]

    neg_samples_input = torch.cat([samples, negative_samples], dim=1)
    neg_samples_input = rearrange(neg_samples_input, "b x h w c -> (b x) h w c")

    with torch.no_grad():
        sg_features_raw = feature_apply(neg_samples_input, **activation_kwargs)
        sg_features = {
            k: rearrange(v, "(b x) ... -> b x ...", b=bsz, x=n_pos + n_uncond)
            for k, v in sg_features_raw.items()
        }

    lr = learning_rate_fn(state.step)
    _set_lr(state.optimizer, lr)

    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    input_labels = repeat(labels, "b -> (b g)", g=gen_per_label)
    input_cfg = repeat(cfg, "b -> (b g)", g=gen_per_label)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen_samples = state.model(c=input_labels, cfg_scale=input_cfg)["samples"]
        gen_features_raw = feature_apply(gen_samples, **activation_kwargs)
        gen_features = {k: rearrange(v, "(b g) ... -> b g ...", b=bsz, g=n_gen) for k, v in gen_features_raw.items()}

    total_loss = torch.tensor(0.0, device=device)
    total_info = {}

    for k in sg_features.keys():
        feature_pos = sg_features[k][:, :n_pos]
        feature_gen = gen_features[k]
        feature_uncond = sg_features[k][:, n_pos:]

        feature_pos = rearrange(feature_pos, "b x f d -> (b f) x d")
        feature_gen = rearrange(feature_gen, "b x f d -> (b f) x d")
        feature_uncond = rearrange(feature_uncond, "b x f d -> (b f) x d")

        b_feat = feature_gen.shape[0]
        weight_neg = repeat(neg_mass, "b k -> (b f) k", f=b_feat // neg_mass.shape[0])
        weight_pos = repeat(positive_weights, "b x -> (b f) x", f=b_feat // positive_weights.shape[0])

        loss_k, info_k = drift_loss(
            gen=feature_gen,
            fixed_pos=feature_pos,
            fixed_neg=feature_uncond,
            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
            weight_pos=weight_pos,
            weight_neg=weight_neg,
            **loss_kwargs,
        )

        total_loss = total_loss + loss_k.mean()
        for k2, v2 in info_k.items():
            total_info[f"{k2}/{k}"] = float(v2.detach().cpu().item())

    total_loss.backward()
    g_norm = float(torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_grad_norm).item())
    state.optimizer.step()

    _update_ema_model(state.ema_model, state.model, state.ema_decay)

    metric = {k: float(v) for k, v in total_info.items()}
    metric["loss"] = float(total_loss.detach().cpu().item())
    metric["g_norm"] = g_norm
    metric["lr"] = float(lr)
    state.step += 1
    return state, metric


def generate_step(batch, params, apply_fn, postprocess_fn, cfg_scale=1.0):
    _, labels = batch
    if isinstance(params, torch.nn.Module):
        model = params
    else:
        model = params["model"] if isinstance(params, dict) and "model" in params else params
    device = next(model.parameters()).device if isinstance(model, torch.nn.Module) else torch.device("cpu")
    labels = labels.to(device=device, dtype=torch.long)
    if apply_fn is None:
        apply_fn = lambda m, y, cfg: m(c=y, cfg_scale=cfg)["samples"]  # noqa: E731
    if isinstance(model, torch.nn.Module):
        model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent_samples = apply_fn(model, labels, cfg_scale)
        return postprocess_fn(latent_samples).cpu()


def _log_training_samples(
    *,
    logger,
    eval_loader,
    model,
    postprocess_fn,
    cfg_scale: float,
    grid_size: int,
    seed: int,
) -> None:
    if _rank() != 0 or grid_size <= 0:
        return
    labels = []
    first_images = None
    for batch in epoch0_sampler(eval_loader):
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue
        if first_images is None:
            first_images = batch[0]
        labels.append(batch[1])
        if sum(int(x.shape[0]) for x in labels) >= grid_size:
            break
    if not labels:
        return
    sample_labels = torch.cat(labels, dim=0)[:grid_size]
    batch = (first_images, sample_labels)
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        samples = generate_step(
            batch,
            params=model,
            apply_fn=lambda m, y, cfg: m(c=y, cfg_scale=cfg, train=False)["samples"],  # noqa: E731
            postprocess_fn=postprocess_fn,
            cfg_scale=float(cfg_scale),
        )
    logger.log_dict({"samples/step": logger.step, "samples/cfg_scale": float(cfg_scale)})
    logger.log_image("samples/train", samples, max_images=grid_size, grid_cols=6)


def train_gen(
    model,
    optimizer,
    logger,
    eval_loader,
    train_loader,
    learning_rate_fn,
    preprocess_fn,
    postprocess_fn,
    dataset_name="imagenet256",
    train_batch_size=0,
    total_steps=100000,
    save_per_step=10000,
    eval_per_step=5000,
    eval_samples=50000,
    activation_fn=None,
    feature_params=None,
    ema_decay=0.999,
    seed=42,
    pos_per_sample=32,
    neg_per_sample=16,
    forward_dict=dict(
        gen_per_label=16,
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
    ),
    positive_bank_size=64,
    negative_bank_size=512,
    cfg_list=(1.0,),
    activation_kwargs=dict(
        patch_mean_size=[2, 4],
        patch_std_size=[2, 4],
        use_std=True,
        use_mean=True,
        every_k_block=2,
    ),
    max_grad_norm=2.0,
    loss_kwargs=dict(R_list=(0.02, 0.05, 0.2)),
    keep_last=2,
    keep_every=None,
    init_from="",
    push_per_step=0,
    push_at_resume=3000,
    workdir="runs",
    eval_on_step_one=True,
    run_eval=True,
    train_sample_per_step: int = 0,
    train_sample_grid_size: int = 36,
    train_sample_cfg: float | None = None,
    train_sample_seed: int = 1234,
    pos_sampling: dict[str, Any] | None = None,
    negative_sampling: dict[str, Any] | None = None,
    bank_checkpoint: dict[str, Any] | bool | None = None,
    compile_level: int = 2,
    profile: bool = False,
):
    torch.manual_seed(seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    ema_model = copy.deepcopy(model).to(device)

    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)

    if cfg_list is None:
        cfg_list = [1.0]
    elif isinstance(cfg_list, (list, tuple)):
        cfg_list = [float(cfg) for cfg in cfg_list]
    else:
        cfg_list = [float(cfg_list)]

    state = TrainState(
        step=0,
        model=model,
        optimizer=optimizer,
        ema_model=ema_model,
        ema_decay=ema_decay,
        device=device,
    )

    print(f"Restoring checkpoint from {workdir}")
    state = restore_checkpoint(state=state, workdir=workdir)
    print(f"Checkpoint restored (step={int(state.step)})")
    if int(state.step) == 0 and init_from:
        log_for_0("Initializing generator params from init_from=%s", init_from)
        state = maybe_init_state_params(
            state,
            model_type="generator",
            init_from=init_from,
            hf_cache_dir=HF_ROOT,
        )

    assert feature_params is not None, "feature_params must be provided for feature extraction"

    state.model = maybe_compile(state.model, compile_level)
    state.ema_model = maybe_compile(state.ema_model, compile_level)

    if _world_size() > 1:
        print(f"Wrapping model with DDP (world_size={_world_size()})...")
        state.model = DDP(
            state.model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        print("DDP ready.")

    print("Starting training loop...")
    step = int(state.step)
    initial_step = step
    pbar = (
        tqdm(range(step, total_steps), initial=step, total=total_steps)
        if is_rank_zero()
        else range(step, total_steps)
    )

    pos_sampling = pos_sampling or {}
    pos_strategy = str(pos_sampling.get("strategy", "uniform")).strip().lower()
    use_anchor_features = bool(pos_sampling.get("use_anchor_features", False))
    anchor_feature_key = str(pos_sampling.get("anchor_feature_key", "layer4_mean")).strip()
    sync_bank_across_ranks = bool(pos_sampling.get("sync_bank_across_ranks", False))
    bank_sync_scope = str(pos_sampling.get("sync_bank_scope", "none"))
    bank_sync_group = _make_bank_sync_group(bank_sync_scope)
    negative_sampling = negative_sampling or {}
    negative_strategy = str(negative_sampling.get("strategy", "none")).strip().lower()
    hard_negative_enabled = negative_strategy in {
        "interclass_hard",
        "hard_interclass",
        "feature_interclass",
        "importance_interclass",
    }
    bank_checkpoint_cfg = bank_checkpoint if isinstance(bank_checkpoint, dict) else {}
    if isinstance(bank_checkpoint, dict):
        bank_checkpoint_enabled = bool(bank_checkpoint_cfg.get("enabled", False))
        bank_checkpoint_keep = int(bank_checkpoint_cfg.get("keep", bank_checkpoint_cfg.get("keep_last", 2)))
    else:
        bank_checkpoint_enabled = bool(bank_checkpoint)
        bank_checkpoint_keep = 2
    bank_checkpoint_scope = _bank_checkpoint_scope(
        bank_sync_scope,
        str(bank_checkpoint_cfg.get("scope", "auto")) if isinstance(bank_checkpoint_cfg, dict) else "auto",
    )

    def _compute_bank_embed(images_in: torch.Tensor) -> torch.Tensor:
        images_dev = images_in.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = activation_fn(images_dev, **activation_kwargs)
        emb = feats.get(anchor_feature_key)
        if emb is None:
            emb = feats.get("layer4")
            if emb is not None:
                emb = emb.mean(dim=1, keepdim=True)
            else:
                emb = feats["global"]
        return emb.reshape(emb.shape[0], -1).detach().float().cpu()

    def _compute_generation_anchor(
        labels_in: torch.Tensor,
        *,
        anchor_model: torch.nn.Module,
        cfg_scale: float,
    ) -> torch.Tensor:
        labels_dev = labels_in.to(device=device, dtype=torch.long, non_blocking=True)
        cfg = torch.full((labels_dev.shape[0],), float(cfg_scale), device=device, dtype=torch.float32)
        was_training = bool(getattr(anchor_model, "training", False))
        anchor_model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            samples = anchor_model(c=labels_dev, cfg_scale=cfg, train=False)["samples"]
        if was_training:
            anchor_model.train()
        return samples.detach().float().cpu()

    memory_bank_positive = ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    if bank_checkpoint_enabled and int(state.step) > 0:
        _restore_memory_banks(
            workdir=workdir,
            step=int(state.step),
            scope=bank_checkpoint_scope,
            positive_bank=memory_bank_positive,
            negative_bank=memory_bank_negative,
        )
    train_iter = infinite_sampler(train_loader, step)

    for step in pbar:
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        goal = push_per_step
        if initial_step > 0 and step == initial_step:
            goal = push_at_resume * push_per_step
            print(f"pushing at resume: {goal}")

        push_images_parts = []
        push_labels_parts = []
        push_feature_parts = []
        image_features_last = None
        while True:
            batch = next(train_iter)
            processed_batch = preprocess_fn(batch)
            images = processed_batch["images"]
            labels = processed_batch["labels"]
            push_images_parts.append(images)
            push_labels_parts.append(labels)
            if use_anchor_features:
                image_features_last = _compute_bank_embed(images)
                push_feature_parts.append(image_features_last)
            n_push += images.shape[0]
            if n_push >= goal:
                break

        bank_images = torch.cat(push_images_parts, dim=0)
        bank_labels = torch.cat(push_labels_parts, dim=0)
        if sync_bank_across_ranks:
            bank_images = _all_gather_tensor_for_bank(bank_images, device=device, group=bank_sync_group)
            bank_labels = _all_gather_tensor_for_bank(bank_labels, device=device, group=bank_sync_group)
        if use_anchor_features:
            bank_features = torch.cat(push_feature_parts, dim=0)
            if sync_bank_across_ranks:
                bank_features = _all_gather_tensor_for_bank(bank_features, device=device, group=bank_sync_group)
            memory_bank_positive.add(bank_images, bank_labels, features=bank_features)
            memory_bank_negative.add(bank_images, bank_labels * 0, features=bank_features)
        else:
            memory_bank_positive.add(bank_images, bank_labels)
            memory_bank_negative.add(bank_images, bank_labels * 0)

        bsz_per_host = train_batch_size // max(1, _world_size())
        anchor_images = images
        anchor_features = image_features_last if use_anchor_features else None
        if bsz_per_host > 0:
            assert labels.shape[0] >= bsz_per_host, f"Labels shape {labels.shape[0]} < bsz_per_host {bsz_per_host}"
            select_indices = torch.randperm(labels.shape[0])[:bsz_per_host]
            labels = labels[select_indices]
            anchor_images = images[select_indices]
            if anchor_features is not None:
                anchor_features = anchor_features[select_indices.cpu()]

        pos_info = {}
        pos_anchor_info = {}
        pos_anchor_images = anchor_images
        pos_anchor_features = anchor_features
        pos_anchor_source = str(pos_sampling.get("anchor_source", "data")).strip().lower()
        if pos_anchor_source in {"ema_generation", "ema_gen", "generation", "gen"}:
            pos_anchor_cfg = float(pos_sampling.get("anchor_cfg", train_sample_cfg or cfg_list[-1]))
            pos_anchor_images = _compute_generation_anchor(
                labels,
                anchor_model=state.ema_model,
                cfg_scale=pos_anchor_cfg,
            )
            pos_anchor_features = _compute_bank_embed(pos_anchor_images) if use_anchor_features else None
            pos_anchor_info["pos_sampler/generated_anchor"] = 1.0
            pos_anchor_info["pos_sampler/generated_anchor_cfg"] = float(pos_anchor_cfg)
        else:
            pos_anchor_info["pos_sampler/generated_anchor"] = 0.0
        if pos_strategy in ("uniform", "none", ""):
            positive_samples = memory_bank_positive.sample(labels.detach().cpu().numpy(), n_samples=pos_per_sample)
            positive_weights = torch.ones((labels.shape[0], pos_per_sample), dtype=torch.float32)
        elif pos_strategy in ("importance_local_uniform", "local_uniform_importance", "weighted_local"):
            weight_power = _linear_schedule(
                float(pos_sampling.get("weight_power", 1.0)),
                float(pos_sampling.get("weight_power_final", pos_sampling.get("weight_power", 1.0))),
                step=step,
                steps=int(pos_sampling.get("weight_power_anneal_steps", pos_sampling.get("anneal_steps", 1))),
                start_step=int(
                    pos_sampling.get("weight_power_start_step", pos_sampling.get("anneal_start_step", initial_step))
                ),
            )
            alpha = _linear_schedule(
                float(pos_sampling.get("local_alpha", 0.75)),
                float(pos_sampling.get("local_alpha_final", pos_sampling.get("local_alpha", 0.75))),
                step=step,
                steps=int(pos_sampling.get("anneal_steps", max(1, total_steps))),
                start_step=int(pos_sampling.get("anneal_start_step", initial_step)),
            )
            positive_samples, positive_weights, pos_info = memory_bank_positive.sample_importance(
                labels.detach().cpu().numpy(),
                n_samples=pos_per_sample,
                anchors=pos_anchor_images,
                anchor_features=pos_anchor_features,
                local_alpha=alpha,
                rank_temperature=float(pos_sampling.get("rank_temperature", 16.0)),
                top_k=int(pos_sampling.get("top_k", 0)),
                min_ess_frac=pos_sampling.get("min_ess_frac", None),
                max_importance_weight=pos_sampling.get("max_importance_weight", None),
                weight_clip=pos_sampling.get("weight_clip", None),
                weight_power=weight_power,
                normalize_weights=bool(pos_sampling.get("normalize_weights", False)),
            )
        elif pos_strategy in ("importance_force_leverage", "force_leverage", "continuous_force"):
            weight_power = _linear_schedule(
                float(pos_sampling.get("weight_power", 1.0)),
                float(pos_sampling.get("weight_power_final", pos_sampling.get("weight_power", 1.0))),
                step=step,
                steps=int(pos_sampling.get("weight_power_anneal_steps", pos_sampling.get("anneal_steps", 1))),
                start_step=int(
                    pos_sampling.get("weight_power_start_step", pos_sampling.get("anneal_start_step", initial_step))
                ),
            )
            alpha = _linear_schedule(
                float(pos_sampling.get("local_alpha", 0.75)),
                float(pos_sampling.get("local_alpha_final", pos_sampling.get("local_alpha", 0.75))),
                step=step,
                steps=int(pos_sampling.get("anneal_steps", max(1, total_steps))),
                start_step=int(pos_sampling.get("anneal_start_step", initial_step)),
            )
            positive_samples, positive_weights, pos_info = memory_bank_positive.sample_importance_force_leverage(
                labels.detach().cpu().numpy(),
                n_samples=pos_per_sample,
                anchors=pos_anchor_images,
                anchor_features=pos_anchor_features,
                local_alpha=alpha,
                radii=pos_sampling.get("radii", loss_kwargs.get("R_list", [0.2, 0.05, 0.02])),
                score_floor=float(pos_sampling.get("score_floor", 0.05)),
                distance_power=float(pos_sampling.get("distance_power", 1.0)),
                kernel_power=float(pos_sampling.get("kernel_power", 1.0)),
                distance_scale=str(pos_sampling.get("distance_scale", "mean")),
                min_ess_frac=pos_sampling.get("min_ess_frac", None),
                max_importance_weight=pos_sampling.get("max_importance_weight", None),
                weight_clip=pos_sampling.get("weight_clip", None),
                weight_power=weight_power,
                normalize_weights=bool(pos_sampling.get("normalize_weights", False)),
            )
        elif pos_strategy in ("importance_feature_stratified", "stratified_feature", "neyman_feature"):
            weight_power = _linear_schedule(
                float(pos_sampling.get("weight_power", 1.0)),
                float(pos_sampling.get("weight_power_final", pos_sampling.get("weight_power", 1.0))),
                step=step,
                steps=int(pos_sampling.get("weight_power_anneal_steps", pos_sampling.get("anneal_steps", 1))),
                start_step=int(
                    pos_sampling.get("weight_power_start_step", pos_sampling.get("anneal_start_step", initial_step))
                ),
            )
            alpha = _linear_schedule(
                float(pos_sampling.get("local_alpha", 1.0)),
                float(pos_sampling.get("local_alpha_final", pos_sampling.get("local_alpha", 1.0))),
                step=step,
                steps=int(pos_sampling.get("anneal_steps", max(1, total_steps))),
                start_step=int(pos_sampling.get("anneal_start_step", initial_step)),
            )
            allocation = pos_sampling.get("allocation", None)
            allocation_final = pos_sampling.get("allocation_final", None)
            if (
                isinstance(allocation, (list, tuple))
                and isinstance(allocation_final, (list, tuple))
                and len(allocation) == len(allocation_final)
            ):
                allocation = [
                    _linear_schedule(
                        float(start_value),
                        float(end_value),
                        step=step,
                        steps=int(
                            pos_sampling.get(
                                "allocation_anneal_steps",
                                pos_sampling.get("anneal_steps", max(1, total_steps)),
                            )
                        ),
                        start_step=int(
                            pos_sampling.get(
                                "allocation_start_step",
                                pos_sampling.get("anneal_start_step", initial_step),
                            )
                        ),
                    )
                    for start_value, end_value in zip(allocation, allocation_final, strict=True)
                ]
            positive_samples, positive_weights, pos_info = memory_bank_positive.sample_importance_stratified(
                labels.detach().cpu().numpy(),
                n_samples=pos_per_sample,
                anchors=pos_anchor_images,
                anchor_features=pos_anchor_features,
                local_alpha=alpha,
                strata_edges=pos_sampling.get("strata_edges", None),
                rank_temperatures=pos_sampling.get("rank_temperatures", None),
                score_floor=float(pos_sampling.get("score_floor", 0.05)),
                allocation=allocation,
                min_per_stratum=int(pos_sampling.get("min_per_stratum", 1)),
                min_ess_frac=pos_sampling.get("min_ess_frac", None),
                max_importance_weight=pos_sampling.get("max_importance_weight", None),
                weight_clip=pos_sampling.get("weight_clip", None),
                weight_power=weight_power,
                normalize_weights=bool(pos_sampling.get("normalize_weights", False)),
            )
        elif pos_strategy in ("importance_feature_mix", "importance_queue_mix", "weighted_queue_mix"):
            weight_power = _linear_schedule(
                float(pos_sampling.get("weight_power", 1.0)),
                float(pos_sampling.get("weight_power_final", pos_sampling.get("weight_power", 1.0))),
                step=step,
                steps=int(pos_sampling.get("weight_power_anneal_steps", pos_sampling.get("anneal_steps", 1))),
                start_step=int(
                    pos_sampling.get("weight_power_start_step", pos_sampling.get("anneal_start_step", initial_step))
                ),
            )
            local_k = round(
                _linear_schedule(
                    float(pos_sampling.get("local_k", 48)),
                    float(pos_sampling.get("local_k_final", pos_sampling.get("local_k", 48))),
                    step=step,
                    steps=int(pos_sampling.get("anneal_steps", max(1, total_steps))),
                    start_step=int(pos_sampling.get("anneal_start_step", initial_step)),
                )
            )
            local_k = int(max(0, min(int(pos_per_sample), local_k)))
            global_r = int(pos_per_sample) - local_k
            positive_samples, positive_weights, pos_info = memory_bank_positive.sample_importance_mix(
                labels.detach().cpu().numpy(),
                n_samples=pos_per_sample,
                anchors=pos_anchor_images,
                anchor_features=pos_anchor_features,
                local_k=local_k,
                global_r=global_r,
                local_mode=str(pos_sampling.get("local_mode", "nearest")),
                weight_power=weight_power,
            )
        else:
            raise ValueError(f"Unknown pos_sampling.strategy={pos_strategy!r}")
        pos_info.update(pos_anchor_info)

        negative_samples = memory_bank_negative.sample((labels * 0).detach().cpu().numpy(), n_samples=neg_per_sample)
        cfg_negative_count = int(negative_samples.shape[1])
        extra_negative_weights = None
        neg_info = {}
        hard_neg_per_sample = int(
            negative_sampling.get("interclass_per_sample", negative_sampling.get("hard_per_sample", 0))
        )
        if hard_negative_enabled and hard_neg_per_sample > 0:
            anchor_info = {}
            hard_neg_anchor_images = anchor_images
            hard_neg_anchor_features = anchor_features
            hard_neg_anchor_source = str(negative_sampling.get("anchor_source", "data")).strip().lower()
            if hard_neg_anchor_source in {"ema_generation", "ema_gen", "generation", "gen"}:
                hard_neg_anchor_cfg = float(negative_sampling.get("anchor_cfg", train_sample_cfg or cfg_list[-1]))
                hard_neg_anchor_images = _compute_generation_anchor(
                    labels,
                    anchor_model=state.ema_model,
                    cfg_scale=hard_neg_anchor_cfg,
                )
                hard_neg_anchor_features = _compute_bank_embed(hard_neg_anchor_images) if use_anchor_features else None
                anchor_info["neg_sampler/generated_anchor"] = 1.0
                anchor_info["neg_sampler/generated_anchor_cfg"] = float(hard_neg_anchor_cfg)
            else:
                anchor_info["neg_sampler/generated_anchor"] = 0.0
            hard_neg_weight = _linear_schedule(
                float(negative_sampling.get("weight", 0.25)),
                float(negative_sampling.get("weight_final", negative_sampling.get("weight", 0.25))),
                step=step,
                steps=int(negative_sampling.get("weight_anneal_steps", negative_sampling.get("anneal_steps", 1))),
                start_step=int(
                    negative_sampling.get(
                        "weight_start_step",
                        negative_sampling.get("anneal_start_step", initial_step),
                    )
                ),
            )
            hard_neg_alpha = _linear_schedule(
                float(negative_sampling.get("local_alpha", 0.8)),
                float(negative_sampling.get("local_alpha_final", negative_sampling.get("local_alpha", 0.8))),
                step=step,
                steps=int(negative_sampling.get("anneal_steps", max(1, total_steps))),
                start_step=int(negative_sampling.get("anneal_start_step", initial_step)),
            )
            hard_neg_weight_power = _linear_schedule(
                float(negative_sampling.get("weight_power", 1.0)),
                float(negative_sampling.get("weight_power_final", negative_sampling.get("weight_power", 1.0))),
                step=step,
                steps=int(negative_sampling.get("weight_power_anneal_steps", negative_sampling.get("anneal_steps", 1))),
                start_step=int(
                    negative_sampling.get(
                        "weight_power_start_step",
                        negative_sampling.get("anneal_start_step", initial_step),
                    )
                ),
            )
            hard_negative_samples, hard_negative_weights, neg_info = memory_bank_positive.sample_interclass_importance(
                labels.detach().cpu().numpy(),
                n_samples=hard_neg_per_sample,
                anchors=hard_neg_anchor_images,
                anchor_features=hard_neg_anchor_features,
                candidate_pool_size=int(negative_sampling.get("candidate_pool_size", 2048)),
                local_alpha=hard_neg_alpha,
                rank_temperature=float(negative_sampling.get("rank_temperature", 32.0)),
                top_k=int(negative_sampling.get("top_k", 0)),
                min_ess_frac=negative_sampling.get("min_ess_frac", None),
                max_importance_weight=negative_sampling.get("max_importance_weight", None),
                weight_clip=negative_sampling.get("weight_clip", None),
                weight_power=hard_neg_weight_power,
                normalize_weights=bool(negative_sampling.get("normalize_weights", False)),
            )
            neg_info.update(anchor_info)
            hard_negative_weights = hard_negative_weights * float(hard_neg_weight)
            negative_samples = torch.cat([negative_samples, hard_negative_samples], dim=1)
            extra_negative_weights = torch.cat(
                [
                    torch.zeros((labels.shape[0], cfg_negative_count), dtype=torch.float32),
                    hard_negative_weights.float(),
                ],
                dim=1,
            )
            neg_info["neg_sampler/interclass_weight_scale"] = float(hard_neg_weight)
            neg_info["neg_sampler/interclass_per_sample"] = float(hard_neg_per_sample)

        merged_positive, merged_negative, merged_labels, merged_pos_weights, merged_neg_weights = merge_data(
            (positive_samples, negative_samples, labels, positive_weights, extra_negative_weights)
        )

        process_time = time.time() - start_time

        profile_metrics = {}
        if profile and step == initial_step:
            def _profile_train_step(
                state_arg,
                labels_arg,
                pos_arg,
                neg_arg,
                pos_weights_arg,
                neg_weights_arg,
                feature_arg,
            ):
                return train_step(
                    state_arg,
                    labels_arg,
                    pos_arg,
                    neg_arg,
                    pos_weights_arg,
                    neg_weights_arg,
                    feature_arg,
                    feature_apply=activation_fn,
                    learning_rate_fn=learning_rate_fn,
                    activation_kwargs=activation_kwargs,
                    loss_kwargs=loss_kwargs,
                    max_grad_norm=max_grad_norm,
                    cfg_negative_count=cfg_negative_count,
                    **forward_dict,
                )

            profile_metrics = profile_func(
                _profile_train_step,
                (
                    state,
                    merged_labels,
                    merged_positive,
                    merged_negative,
                    merged_pos_weights,
                    merged_neg_weights,
                    feature_params,
                ),
                name="train_step",
            )

        state, metrics = train_step(
            state,
            merged_labels,
            merged_positive,
            merged_negative,
            merged_pos_weights,
            merged_neg_weights,
            feature_params,
            feature_apply=activation_fn,
            learning_rate_fn=learning_rate_fn,
            activation_kwargs=activation_kwargs,
            loss_kwargs=loss_kwargs,
            max_grad_norm=max_grad_norm,
            cfg_negative_count=cfg_negative_count,
            **forward_dict,
        )

        total_time = time.time() - start_time
        metrics["total_time"] = total_time
        metrics["process_time"] = process_time
        metrics["kimg"] = (step + 1) * merged_positive.shape[0] / 1000.0
        metrics["forward_kimg"] = (step + 1) * merged_positive.shape[0] / 1000.0 * forward_dict["gen_per_label"]
        metrics.update(profile_metrics)
        metrics.update(pos_info)
        metrics.update(neg_info)

        logger.log_dict(metrics)

        now_step = step + 1
        do_train_sample = int(train_sample_per_step) > 0 and (
            now_step % int(train_sample_per_step) == 0 or now_step == total_steps
        )
        if do_train_sample:
            sample_cfg = cfg_list[0] if train_sample_cfg is None else float(train_sample_cfg)
            _log_training_samples(
                logger=logger,
                eval_loader=eval_loader,
                model=state.ema_model,
                postprocess_fn=postprocess_fn,
                cfg_scale=sample_cfg,
                grid_size=int(train_sample_grid_size),
                seed=int(train_sample_seed),
            )
            _sync_after_rank0_io(workdir, f"sample_step_{now_step:09d}")

        if now_step % save_per_step == 0 or now_step == total_steps:
            save_checkpoint(state, keep=keep_last, keep_every=keep_every, workdir=workdir)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="gen",
                model_config=_generator_model_config(model),
            )
            if bank_checkpoint_enabled:
                _save_memory_banks(
                    workdir=workdir,
                    step=now_step,
                    scope=bank_checkpoint_scope,
                    positive_bank=memory_bank_positive,
                    negative_bank=memory_bank_negative,
                    keep=bank_checkpoint_keep,
                )
            _sync_after_rank0_io(workdir, f"save_step_{now_step:09d}")

        do_step_one_eval = bool(eval_on_step_one) and now_step == 1
        do_eval = bool(run_eval) and ((now_step % eval_per_step == 0) or do_step_one_eval or (now_step == total_steps))
        if do_eval:
            is_sanity = do_step_one_eval
            n_samples = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            round_best_fid = float("inf")
            round_best_cfg = cfg_list[0]
            eval_cfg_list = cfg_list if not is_sanity else [cfg_list[0]]

            use_ddp_eval = _world_size() > 1
            eval_fn = evaluate_fid_ddp if use_ddp_eval else evaluate_fid
            for eval_cfg in eval_cfg_list:
                result = eval_fn(
                    dataset_name=dataset_name,
                    gen_func=generate_step,
                    gen_params={
                        "params": state.ema_model,
                        "apply_fn": lambda m, y, cfg: m(c=y, cfg_scale=cfg, train=False)["samples"],  # noqa: E731
                        "cfg_scale": eval_cfg,
                        "postprocess_fn": postprocess_fn,
                    },
                    eval_loader=eval_loader,
                    logger=logger if _rank() == 0 else None,
                    num_samples=n_samples,
                    log_folder=f"{folder_prefix}{eval_cfg}",
                    log_prefix=f"EMA_{state.ema_decay:g}",
                )
                if _rank() == 0:
                    fid_val = result.get("fid", float("inf"))
                    if fid_val < round_best_fid:
                        round_best_fid = fid_val
                        round_best_cfg = eval_cfg

            if not is_sanity and _rank() == 0:
                log_for_0("best_fid=%.4f best_cfg=%.1f (step=%d)", round_best_fid, round_best_cfg, now_step)
                logger.log_dict({"best_fid": round_best_fid, "best_cfg": round_best_cfg})
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    logger.finish()
    del model, optimizer, eval_loader, train_loader, state
    gc.collect()


def main_gen(config, output_dir="runs", profile=False):
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name

    from models.generator import DitGen

    set_global_mesh(config.get("hsdp_dim", min(8, max(1, _world_size()))))

    print("Building generator model...")
    model_dict = build_model_dict(config, DitGen, workdir=output_dir)
    print("Generator model built.")
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))
    postprocess_fn_noclip = get_postprocess_fn(
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        has_clip=False,
    )
    feature_cfg = model_dict.feature
    mae_path = str(feature_cfg.get("mae_path", "")).strip()
    if not mae_path and bool(feature_cfg.get("use_mae", True)):
        load_dict = feature_cfg.get("load_dict", {})
        if str(load_dict.get("source", "hf")).strip().lower() == "local":
            mae_path = str(load_dict.get("path", "")).strip()
        else:
            model_name = str(load_dict.get("hf_model_name", "")).strip()
            if model_name:
                mae_path = f"hf://{model_name}"
    if bool(feature_cfg.get("use_mae", True)) and not mae_path:
        raise ValueError(
            "feature.mae_path (or feature.load_dict.hf_model_name / feature.load_dict.path) "
            "is required when use_mae=true."
        )
    compile_level = int(config.get("compile", 2))
    print(f"Loading feature extractor from {mae_path}...")
    activation_fn, variables = build_activation_function(
        mae_path=mae_path,
        use_convnext=bool(feature_cfg.get("use_convnext", False)),
        use_mae=bool(feature_cfg.get("use_mae", True)),
        postprocess_fn=postprocess_fn_noclip,
        compile_level=compile_level,
        device=_device(),
    )
    print("Feature extractor loaded.")
    train_gen(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        dataset_name=model_dict.dataset_name,
        activation_fn=activation_fn,
        feature_params=variables,
        workdir=output_dir,
        compile_level=compile_level,
        profile=profile,
        **sanitize_train_config(config.train),
    )


def main(args):
    run_init()
    config = load_config(args.config)
    main_gen(config, output_dir=args.workdir, profile=args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gen/latent_ablation.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Run profiling on the first training step.",
    )
    args = parser.parse_args()
    args.workdir = stamp_workdir(args.workdir)
    args.output_dir = args.workdir

    main(args)
