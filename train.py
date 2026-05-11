from __future__ import annotations

import argparse
import copy
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange, repeat
from tqdm import tqdm

from dataset.dataset import epoch0_sampler, get_postprocess_fn, infinite_sampler
from drift_loss import drift_loss
from likelihood_prior import (
    anneal_int,
    anneal_value,
    compute_likelihood_descriptors,
    initialize_explicit_likelihood_prior,
    likelihood_balance_loss,
    likelihood_info,
    likelihood_usage_entropy,
)
from memory_bank import ArrayMemoryBank, ComponentArrayMemoryBank
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
    ema_model: Optional[torch.nn.Module] = None
    ema_decay: float = 0.999
    device: Optional[torch.device] = None


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
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _generator_model_config(model) -> dict:
    model = _unwrap_model(model)
    if hasattr(model, "model_config"):
        return dict(model.model_config)
    return {}


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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
    feature_params,
    feature_apply,
    learning_rate_fn: Any = None,
    cfg_min=1.0,
    cfg_max=4.0,
    neg_cfg_pw=1.0,
    no_cfg_frac=0.0,
    gen_per_label=8,
    activation_kwargs=dict(),
    loss_kwargs=dict(R_list=[0.02, 0.05, 0.2]),
    max_grad_norm=2.0,
    likelihood_descriptors=None,
    likelihood_components=None,
    likelihood_weights=None,
    likelihood_conditioning="none",
    likelihood_temperature=1.0,
    likelihood_component_dropout=0.0,
    likelihood_nll_weight=0.0,
    likelihood_balance_weight=0.0,
):
    device = state.device
    labels = labels.to(device=device, dtype=torch.long)
    samples = samples.to(device=device, dtype=torch.float32)
    negative_samples = negative_samples.to(device=device, dtype=torch.float32)
    if likelihood_descriptors is not None:
        likelihood_descriptors = likelihood_descriptors.to(device=device, dtype=torch.float32)
    if likelihood_components is not None:
        likelihood_components = likelihood_components.to(device=device, dtype=torch.long)
    if likelihood_weights is not None:
        likelihood_weights = likelihood_weights.to(device=device, dtype=torch.float32)
    likelihood_temperature_tensor = torch.as_tensor(likelihood_temperature, device=device, dtype=torch.float32).clamp_min(1e-6)

    bsz = labels.shape[0]

    frac = torch.rand((bsz,), device=device)
    pw = 1 - neg_cfg_pw
    if abs(pw) < 1e-6:
        cfg = torch.exp(torch.log(torch.tensor(cfg_min, device=device)) + frac * (torch.log(torch.tensor(cfg_max, device=device)) - torch.log(torch.tensor(cfg_min, device=device))))
    else:
        cfg = (cfg_min**pw + frac * (cfg_max**pw - cfg_min**pw)) ** (1 / pw)

    frac2 = torch.rand((bsz,), device=device)
    cfg = torch.where(frac2 < no_cfg_frac, torch.ones_like(cfg), cfg)

    uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, negative_samples.shape[1])
    n_pos, n_gen, n_uncond = samples.shape[1], gen_per_label, negative_samples.shape[1]

    neg_samples_input = torch.cat([samples, negative_samples], dim=1)
    neg_samples_input = rearrange(neg_samples_input, "b x h w c -> (b x) h w c")

    with torch.no_grad():
        sg_features_raw = feature_apply(neg_samples_input, **activation_kwargs)
        sg_features = {k: rearrange(v, "(b x) ... -> b x ...", b=bsz, x=n_pos + n_uncond) for k, v in sg_features_raw.items()}

    lr = learning_rate_fn(state.step)
    _set_lr(state.optimizer, lr)

    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    input_labels = repeat(labels, "b -> (b g)", g=gen_per_label)
    input_cfg = repeat(cfg, "b -> (b g)", g=gen_per_label)
    input_likelihood_components = None
    input_likelihood_weights = None
    if likelihood_components is not None:
        likelihood_components = likelihood_components.clone()
        if likelihood_weights is not None:
            likelihood_weights = likelihood_weights.clone()
        if likelihood_component_dropout > 0:
            prior_model = _unwrap_model(state.model)
            null_index = int(getattr(prior_model, "likelihood_null_index", 0))
            drop = torch.rand((bsz,), device=device) < float(likelihood_component_dropout)
            if likelihood_components.ndim == 1:
                likelihood_components = torch.where(drop, torch.full_like(likelihood_components, null_index), likelihood_components)
            else:
                likelihood_components[drop] = null_index
                if likelihood_weights is not None:
                    likelihood_weights[drop] = 0.0
                    likelihood_weights[drop, 0] = 1.0
        if likelihood_components.ndim == 1:
            input_likelihood_components = repeat(likelihood_components, "b -> (b g)", g=gen_per_label)
        else:
            input_likelihood_components = repeat(likelihood_components, "b k -> (b g) k", g=gen_per_label)
        if likelihood_weights is not None:
            input_likelihood_weights = repeat(likelihood_weights, "b k -> (b g) k", g=gen_per_label)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen_out = state.model(
            c=input_labels,
            cfg_scale=input_cfg,
            likelihood_components=input_likelihood_components,
            likelihood_weights=input_likelihood_weights,
            likelihood_conditioning=likelihood_conditioning,
            likelihood_temperature=likelihood_temperature_tensor,
            likelihood_loss_descriptors=(
                likelihood_descriptors
                if likelihood_descriptors is not None and (likelihood_nll_weight > 0 or likelihood_balance_weight > 0)
                else None
            ),
            likelihood_loss_labels=labels,
            likelihood_resp_temperature=likelihood_temperature_tensor,
        )
        gen_samples = gen_out["samples"]
        gen_features_raw = feature_apply(gen_samples, **activation_kwargs)
        gen_features = {k: rearrange(v, "(b g) ... -> b g ...", b=bsz, g=n_gen) for k, v in gen_features_raw.items()}

    total_loss = torch.tensor(0.0, device=device)
    total_info = {}

    if likelihood_descriptors is not None and (likelihood_nll_weight > 0 or likelihood_balance_weight > 0):
        prior_model = _unwrap_model(state.model)
        nll = gen_out["likelihood_nll"]
        resp = gen_out["likelihood_resp"]
        nll_per_dim = nll / max(1, int(likelihood_descriptors.shape[1]))
        balance = likelihood_balance_loss(resp)
        total_loss = total_loss + float(likelihood_nll_weight) * nll_per_dim
        total_loss = total_loss + float(likelihood_balance_weight) * balance
        total_info["likelihood/nll_per_dim"] = nll_per_dim.detach()
        total_info["likelihood/balance"] = balance.detach()
        total_info["likelihood/usage_entropy"] = likelihood_usage_entropy(prior_model.likelihood_class_logits).detach()
        for k_info, v_info in likelihood_info(resp).items():
            total_info[k_info] = torch.tensor(v_info, device=device)

    for k in sg_features.keys():
        feature_pos = sg_features[k][:, :n_pos]
        feature_gen = gen_features[k]
        feature_uncond = sg_features[k][:, n_pos:]

        feature_pos = rearrange(feature_pos, "b x f d -> (b f) x d")
        feature_gen = rearrange(feature_gen, "b x f d -> (b f) x d")
        feature_uncond = rearrange(feature_uncond, "b x f d -> (b f) x d")

        b_feat = feature_gen.shape[0]
        weight_neg = repeat(uncond_w, "b -> (b f) k", f=b_feat // uncond_w.shape[0], k=n_uncond)

        loss_k, info_k = drift_loss(
            gen=feature_gen,
            fixed_pos=feature_pos,
            fixed_neg=feature_uncond,
            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
            weight_pos=torch.ones_like(feature_pos[:, :, 0]),
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
    compile_level: int = 2,
    explicit_likelihood: dict | None = None,
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

    likelihood_cfg = dict(explicit_likelihood or {})
    likelihood_enabled = bool(likelihood_cfg.get("enabled", False))
    if likelihood_enabled:
        likelihood_cfg.setdefault("cache_path", str(Path(workdir).resolve() / "likelihood_prior.pt"))
        initialized = initialize_explicit_likelihood_prior(
            model=state.model,
            train_loader=train_loader,
            preprocess_fn=preprocess_fn,
            activation_fn=activation_fn,
            config=likelihood_cfg,
            default_activation_kwargs=activation_kwargs,
            device=device,
            seed=seed,
        )
        if initialized:
            if hasattr(state.ema_model, "resize_likelihood_prior_from_state_dict"):
                state.ema_model.resize_likelihood_prior_from_state_dict(state.model.state_dict())
            state.ema_model.load_state_dict(state.model.state_dict(), strict=False)

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
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    if train_batch_size > 0:
        world = max(1, _world_size())
        if train_batch_size < world or train_batch_size % world != 0:
            raise ValueError(
                "train_batch_size must be a global batch divisible by distributed world size: "
                f"train_batch_size={train_batch_size}, world_size={world}. "
                "Use train_batch_size=0 to consume each dataset batch without sub-sampling."
            )

    memory_bank_positive = (
        ComponentArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
        if likelihood_enabled
        else ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    )
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    train_iter = infinite_sampler(train_loader, step)
    likelihood_descriptor_keys = likelihood_cfg.get("descriptor_keys", None)
    likelihood_descriptor_keys = list(likelihood_descriptor_keys) if likelihood_descriptor_keys else None
    likelihood_activation_kwargs = dict(activation_kwargs)
    likelihood_activation_kwargs.update(dict(likelihood_cfg.get("activation_kwargs", {})))

    for step in pbar:
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        goal = push_per_step
        if initial_step > 0 and step == initial_step:
            goal = push_at_resume * push_per_step
            print(f"pushing at resume: {goal}")

        while True:
            batch = next(train_iter)
            processed_batch = preprocess_fn(batch)
            images = processed_batch["images"]
            labels = processed_batch["labels"]
            if likelihood_enabled:
                with torch.no_grad():
                    push_desc = compute_likelihood_descriptors(
                        images,
                        activation_fn=activation_fn,
                        activation_kwargs=likelihood_activation_kwargs,
                        descriptor_keys=likelihood_descriptor_keys,
                        device=device,
                    )
                    prior_model = _unwrap_model(state.model)
                    push_components = prior_model.likelihood_assign(
                        push_desc,
                        labels.to(device=device, dtype=torch.long),
                        temperature=float(likelihood_cfg.get("assignment_temperature", 1.0)),
                    )
                memory_bank_positive.add(
                    images.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                    push_components.detach().cpu().numpy(),
                )
            else:
                memory_bank_positive.add(images.detach().cpu().numpy(), labels.detach().cpu().numpy())
            memory_bank_negative.add(images.detach().cpu().numpy(), (labels * 0).detach().cpu().numpy())
            n_push += images.shape[0]
            if n_push >= goal:
                break

        anchor_samples = images
        bsz_per_host = train_batch_size // max(1, _world_size())
        if bsz_per_host > 0:
            assert labels.shape[0] >= bsz_per_host, f"Labels shape {labels.shape[0]} < bsz_per_host {bsz_per_host}"
            select_indices = torch.randperm(labels.shape[0])[:bsz_per_host]
            labels = labels[select_indices]
            anchor_samples = anchor_samples[select_indices]

        likelihood_descriptors = None
        likelihood_components = None
        likelihood_weights = None
        likelihood_step_info = {}
        likelihood_conditioning = str(likelihood_cfg.get("conditioning", "none" if not likelihood_enabled else "sparse_soft"))
        likelihood_temperature = float(likelihood_cfg.get("temperature", 1.0))
        if likelihood_enabled:
            likelihood_temperature = anneal_value(
                float(likelihood_cfg.get("temp_start", likelihood_cfg.get("temperature", 1.0))),
                float(likelihood_cfg.get("temp_final", likelihood_cfg.get("temperature", 1.0))),
                step,
                int(likelihood_cfg.get("temp_anneal_steps", 0)),
            )
            likelihood_topk = anneal_int(
                int(likelihood_cfg.get("topk_start", likelihood_cfg.get("topk", 1))),
                int(likelihood_cfg.get("topk_final", likelihood_cfg.get("topk", 1))),
                step,
                int(likelihood_cfg.get("topk_anneal_steps", likelihood_cfg.get("temp_anneal_steps", 0))),
            )
            with torch.no_grad():
                likelihood_descriptors = compute_likelihood_descriptors(
                    anchor_samples,
                    activation_fn=activation_fn,
                    activation_kwargs=likelihood_activation_kwargs,
                    descriptor_keys=likelihood_descriptor_keys,
                    device=device,
                )
                prior_model = _unwrap_model(state.model)
                likelihood_components, likelihood_weights, likelihood_step_info = (
                    prior_model.likelihood_topk_responsibilities(
                        likelihood_descriptors,
                        labels.to(device=device, dtype=torch.long),
                        temperature=likelihood_temperature,
                        topk=likelihood_topk,
                    )
                )
            positive_samples = memory_bank_positive.sample_likelihood(
                labels.detach().cpu().numpy(),
                likelihood_components.detach().cpu().numpy(),
                likelihood_weights.detach().cpu().numpy(),
                n_samples=pos_per_sample,
                uniform_fraction=float(likelihood_cfg.get("uniform_same_class_positive_fraction", 0.2)),
            )
            if pos_per_sample > 0:
                positive_samples[:, 0] = anchor_samples.detach().cpu()
        else:
            positive_samples = memory_bank_positive.sample(labels.detach().cpu().numpy(), n_samples=pos_per_sample)
        negative_samples = memory_bank_negative.sample((labels * 0).detach().cpu().numpy(), n_samples=neg_per_sample)

        if likelihood_enabled:
            (
                merged_positive,
                merged_negative,
                merged_labels,
                merged_likelihood_descriptors,
                merged_likelihood_components,
                merged_likelihood_weights,
            ) = merge_data(
                (
                    positive_samples,
                    negative_samples,
                    labels,
                    likelihood_descriptors,
                    likelihood_components,
                    likelihood_weights,
                )
            )
        else:
            merged_positive, merged_negative, merged_labels = merge_data((positive_samples, negative_samples, labels))
            merged_likelihood_descriptors = None
            merged_likelihood_components = None
            merged_likelihood_weights = None

        process_time = time.time() - start_time

        profile_metrics = {}
        if profile and step == initial_step:
            profile_metrics = profile_func(
                lambda s, l, p, n, fp: train_step(
                    s,
                    l,
                    p,
                    n,
                    fp,
                    feature_apply=activation_fn,
                    learning_rate_fn=learning_rate_fn,
                    activation_kwargs=activation_kwargs,
                    loss_kwargs=loss_kwargs,
                    max_grad_norm=max_grad_norm,
                    likelihood_descriptors=merged_likelihood_descriptors,
                    likelihood_components=merged_likelihood_components,
                    likelihood_weights=merged_likelihood_weights,
                    likelihood_conditioning=likelihood_conditioning,
                    likelihood_temperature=likelihood_temperature,
                    likelihood_component_dropout=float(likelihood_cfg.get("component_dropout", 0.0)),
                    likelihood_nll_weight=float(likelihood_cfg.get("nll_weight", 0.0)),
                    likelihood_balance_weight=float(likelihood_cfg.get("balance_weight", 0.0)),
                    **forward_dict,
                ),
                (state, merged_labels, merged_positive, merged_negative, feature_params),
                name="train_step",
            )

        state, metrics = train_step(
            state,
            merged_labels,
            merged_positive,
            merged_negative,
            feature_params,
            feature_apply=activation_fn,
            learning_rate_fn=learning_rate_fn,
            activation_kwargs=activation_kwargs,
            loss_kwargs=loss_kwargs,
            max_grad_norm=max_grad_norm,
            likelihood_descriptors=merged_likelihood_descriptors,
            likelihood_components=merged_likelihood_components,
            likelihood_weights=merged_likelihood_weights,
            likelihood_conditioning=likelihood_conditioning,
            likelihood_temperature=likelihood_temperature,
            likelihood_component_dropout=float(likelihood_cfg.get("component_dropout", 0.0)),
            likelihood_nll_weight=float(likelihood_cfg.get("nll_weight", 0.0)),
            likelihood_balance_weight=float(likelihood_cfg.get("balance_weight", 0.0)),
            **forward_dict,
        )

        total_time = time.time() - start_time
        metrics["total_time"] = total_time
        metrics["process_time"] = process_time
        metrics["kimg"] = (step + 1) * merged_positive.shape[0] / 1000.0
        metrics["forward_kimg"] = (step + 1) * merged_positive.shape[0] / 1000.0 * forward_dict["gen_per_label"]
        metrics.update(profile_metrics)
        if likelihood_enabled:
            metrics["likelihood/temp"] = float(likelihood_temperature)
            metrics["likelihood/topk"] = float(likelihood_step_info.get("topk", 0.0))
            metrics["likelihood/topk_mass"] = float(likelihood_step_info.get("topk_mass", 0.0))
            metrics["likelihood/posterior_entropy_anchor"] = float(likelihood_step_info.get("posterior_entropy", 0.0))

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
        raise ValueError("feature.mae_path (or feature.load_dict.hf_model_name / feature.load_dict.path) is required when use_mae=true.")
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
        explicit_likelihood=config.get("likelihood_prior", {}),
        profile=profile,
        **sanitize_train_config(config.train),
    )


def main(args):
    run_init()
    config = load_config(args.config)
    main_gen(config, output_dir=args.workdir, profile=args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gen/latent_ablation.yaml", help="Path to configuration file.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    parser.add_argument("--profile", action="store_true", default=False, help="Run profiling on the first training step.")
    args = parser.parse_args()
    args.workdir = stamp_workdir(args.workdir)
    args.output_dir = args.workdir

    main(args)
