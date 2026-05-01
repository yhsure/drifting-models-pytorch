from __future__ import annotations

import argparse
import copy
import gc
import math
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
from drift_loss import drift_loss, jsd_mog_loss, likelihood_mog_loss
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
    mog_log_sigmas: dict[str, torch.nn.Parameter] | None = None
    mog_log_sigma_init: dict[str, torch.Tensor] | None = None
    optimizer_state_init: dict | None = None


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
    if hasattr(model, "model_config"):
        return dict(model.model_config)
    return {}


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _add_mog_param_group(
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
) -> None:
    existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
    new_params = [p for p in params if id(p) not in existing]
    if new_params:
        optimizer.add_param_group({"params": new_params, "weight_decay": 0.0})


def _ensure_mog_log_sigmas(state: TrainState, sg_features: dict[str, torch.Tensor]) -> None:
    if state.mog_log_sigmas is None:
        state.mog_log_sigmas = {}

    init_payload = state.mog_log_sigma_init or {}
    ordered_keys = [key for key in init_payload if key in sg_features]
    ordered_keys.extend(key for key in sg_features if key not in init_payload)
    for key in ordered_keys:
        if key in state.mog_log_sigmas:
            continue
        feat = sg_features[key]
        dim = int(feat.shape[-1])
        if key in init_payload:
            init = torch.as_tensor(init_payload[key], dtype=torch.float32, device=state.device).reshape(-1)
            if init.numel() == 1:
                init = init.repeat(dim)
            elif init.numel() != dim:
                raise ValueError(f"mog_log_sigmas[{key!r}] has {init.numel()} values, expected {dim}.")
        else:
            init = torch.full((dim,), math.log(float(dim) ** 0.5), dtype=torch.float32, device=state.device)
        param = torch.nn.Parameter(init.detach().clone())
        state.mog_log_sigmas[key] = param

    _add_mog_param_group(state.optimizer, list(state.mog_log_sigmas.values()))
    if state.optimizer_state_init is not None:
        state.optimizer.load_state_dict(state.optimizer_state_init)
        state.optimizer_state_init = None


def _sync_mog_sigma_grads(state: TrainState) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if not state.mog_log_sigmas:
        return
    world_size = dist.get_world_size()
    for param in state.mog_log_sigmas.values():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def _grad_clip_params(state: TrainState) -> list[torch.nn.Parameter]:
    params = list(state.model.parameters())
    if state.mog_log_sigmas:
        params.extend(state.mog_log_sigmas.values())
    return params


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
    lambda_drift=1.0,
    lambda_mog=0.0,
    mog_loss_type="jsd",
    sigma_latent=1.0,
):
    device = state.device
    do_drift = lambda_drift > 0.0
    do_mog = lambda_mog > 0.0
    mog_loss_type = str(mog_loss_type).lower()
    if do_mog and mog_loss_type not in {"jsd", "likelihood", "nll"}:
        raise ValueError(f"Unsupported mog_loss_type={mog_loss_type!r}. Expected 'jsd' or 'likelihood'.")
    labels = labels.to(device=device, dtype=torch.long)
    samples = samples.to(device=device, dtype=torch.float32)
    negative_samples = negative_samples.to(device=device, dtype=torch.float32)

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

    uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, negative_samples.shape[1])
    n_pos, n_gen, n_uncond = samples.shape[1], gen_per_label, negative_samples.shape[1]

    neg_samples_input = torch.cat([samples, negative_samples], dim=1)
    neg_samples_input = rearrange(neg_samples_input, "b x h w c -> (b x) h w c")

    with torch.no_grad():
        sg_features_raw = feature_apply(neg_samples_input, **activation_kwargs)
        sg_features = {
            k: rearrange(v, "(b x) ... -> b x ...", b=bsz, x=n_pos + n_uncond)
            for k, v in sg_features_raw.items()
        }

    if do_mog:
        _ensure_mog_log_sigmas(state, sg_features)

    lr = learning_rate_fn(state.step)
    _set_lr(state.optimizer, lr)

    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    input_labels = repeat(labels, "b -> (b g)", g=gen_per_label)
    input_cfg = repeat(cfg, "b -> (b g)", g=gen_per_label)
    gen_features = None
    gen_features_mog = None
    if do_drift or do_mog:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if do_drift and do_mog:
                two_stage = state.model(
                    c=input_labels, cfg_scale=input_cfg, two_stage=True, sigma_latent=sigma_latent,
                )
                gen_features_drift_raw = feature_apply(two_stage["gen_drift"], **activation_kwargs)
                gen_features_mog_raw = feature_apply(two_stage["gen_mog"], **activation_kwargs)
                gen_features = {
                    k: rearrange(v, "(b g) ... -> b g ...", b=bsz, g=n_gen)
                    for k, v in gen_features_drift_raw.items()
                }
                gen_features_mog = {
                    k: rearrange(v, "(b g) ... -> b g ...", b=bsz, g=n_gen)
                    for k, v in gen_features_mog_raw.items()
                }
            else:
                gen_samples = state.model(c=input_labels, cfg_scale=input_cfg)["samples"]
                gen_features_raw = feature_apply(gen_samples, **activation_kwargs)
                features = {
                    k: rearrange(v, "(b g) ... -> b g ...", b=bsz, g=n_gen)
                    for k, v in gen_features_raw.items()
                }
                if do_drift:
                    gen_features = features
                if do_mog:
                    gen_features_mog = features

    total_loss = torch.tensor(0.0, device=device)
    total_info = {}

    for k in sg_features.keys():
        feature_pos = sg_features[k][:, :n_pos]
        feature_pos = rearrange(feature_pos, "b x f d -> (b f) x d")

        if do_drift:
            feature_gen = gen_features[k]
            feature_uncond = sg_features[k][:, n_pos:]

            # x is the sample axis. b and f index independent label/feature-position spaces.
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

            drift_k = loss_k.mean()
            total_loss = total_loss + lambda_drift * drift_k
            total_info[f"drift_loss/{k}"] = float(drift_k.detach().cpu().item())
            for k2, v2 in info_k.items():
                total_info[f"{k2}/{k}"] = float(v2.detach().cpu().item())

        if do_mog:
            feature_gen_mog = rearrange(gen_features_mog[k], "b x f d -> (b f) x d")
            sigma_k = state.mog_log_sigmas[k].exp()
            if mog_loss_type == "jsd":
                mog_k = jsd_mog_loss(feature_gen_mog, feature_pos, sigma=sigma_k)
            else:
                mog_k = likelihood_mog_loss(feature_gen_mog, feature_pos, sigma=sigma_k)
            total_loss = total_loss + lambda_mog * mog_k
            total_info[f"mog_loss/{k}"] = float(mog_k.detach().cpu().item())
            sigma_info = sigma_k.detach().float()
            total_info[f"mog_sigma/{k}"] = float(sigma_info.mean().cpu().item())
            total_info[f"mog_sigma_min/{k}"] = float(sigma_info.min().cpu().item())
            total_info[f"mog_sigma_max/{k}"] = float(sigma_info.max().cpu().item())

    if total_loss.requires_grad:
        total_loss.backward()
        _sync_mog_sigma_grads(state)
        g_norm = float(torch.nn.utils.clip_grad_norm_(_grad_clip_params(state), max_grad_norm).item())
        state.optimizer.step()
    else:
        g_norm = 0.0

    with torch.inference_mode():
        base_model = state.model.module if hasattr(state.model, "module") else state.model
        for p_ema, p in zip(state.ema_model.parameters(), base_model.parameters()):
            p_ema.mul_(state.ema_decay).add_(p, alpha=(1.0 - state.ema_decay))

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
    profile: bool = False,
    lambda_drift: float = 1.0,
    lambda_mog: float = 0.0,
    mog_loss_type: str = "jsd",
    sigma_latent: float = 1.0,
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
    if is_rank_zero():
        pbar = tqdm(range(step, total_steps), initial=step, total=total_steps)
    else:
        pbar = range(step, total_steps)

    memory_bank_positive = ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    train_iter = infinite_sampler(train_loader, step)

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
            memory_bank_positive.add(images.detach().cpu().numpy(), labels.detach().cpu().numpy())
            memory_bank_negative.add(images.detach().cpu().numpy(), (labels * 0).detach().cpu().numpy())
            n_push += images.shape[0]
            if n_push >= goal:
                break

        bsz_per_host = train_batch_size // max(1, _world_size())
        if bsz_per_host > 0:
            assert labels.shape[0] >= bsz_per_host, f"Labels shape {labels.shape[0]} < bsz_per_host {bsz_per_host}"
            select_indices = torch.randperm(labels.shape[0])[:bsz_per_host]
            labels = labels[select_indices]

        positive_samples = memory_bank_positive.sample(labels.detach().cpu().numpy(), n_samples=pos_per_sample)
        negative_samples = memory_bank_negative.sample((labels * 0).detach().cpu().numpy(), n_samples=neg_per_sample)

        merged_positive, merged_negative, merged_labels = merge_data((positive_samples, negative_samples, labels))

        process_time = time.time() - start_time

        profile_metrics = {}
        if profile and step == initial_step:
            profile_metrics = profile_func(
                lambda s, labels_, p, n, fp: train_step(
                    s,
                    labels_,
                    p,
                    n,
                    fp,
                    feature_apply=activation_fn,
                    learning_rate_fn=learning_rate_fn,
                    activation_kwargs=activation_kwargs,
                    loss_kwargs=loss_kwargs,
                    max_grad_norm=max_grad_norm,
                    lambda_drift=lambda_drift,
                    lambda_mog=lambda_mog,
                    mog_loss_type=mog_loss_type,
                    sigma_latent=sigma_latent,
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
            lambda_drift=lambda_drift,
            lambda_mog=lambda_mog,
            mog_loss_type=mog_loss_type,
            sigma_latent=sigma_latent,
            **forward_dict,
        )

        total_time = time.time() - start_time
        metrics["total_time"] = total_time
        metrics["process_time"] = process_time
        metrics["kimg"] = (step + 1) * merged_positive.shape[0] / 1000.0
        metrics["forward_kimg"] = (step + 1) * merged_positive.shape[0] / 1000.0 * forward_dict["gen_per_label"]
        metrics.update(profile_metrics)

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
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if now_step % save_per_step == 0 or now_step == total_steps:
            save_checkpoint(state, keep=keep_last, workdir=workdir)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="gen",
                model_config=_generator_model_config(model),
            )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

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
