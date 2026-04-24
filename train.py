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

from dataset.dataset import get_postprocess_fn, infinite_sampler
from drift_loss import drift_loss
from memory_bank import ArrayMemoryBank
from models.mae_model import build_activation_function
from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
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
    return model.module if hasattr(model, "module") else model


def _generator_model_config(model) -> dict:
    model = _unwrap_model(model)
    if hasattr(model, "model_config"):
        return dict(model.model_config)
    return {}


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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
):
    device = state.device
    labels = labels.to(device=device, dtype=torch.long)
    samples = samples.to(device=device, dtype=torch.float32)
    negative_samples = negative_samples.to(device=device, dtype=torch.float32)

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
    compile_level: int = 2,
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

    state = restore_checkpoint(state=state, workdir=workdir)
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
        state.model = DDP(
            state.model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    log_for_0("Starting training loop...")
    step = int(state.step)
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)

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
        if step == initial_step:
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
            if _rank() == 0:
                is_sanity = do_step_one_eval
                n_samples = 500 if is_sanity else eval_samples
                folder_prefix = "sanity" if is_sanity else "CFG"
                round_best_fid = float("inf")
                round_best_cfg = cfg_list[0]
                eval_cfg_list = cfg_list if not is_sanity else [cfg_list[0]]

                for eval_cfg in eval_cfg_list:
                    result = evaluate_fid(
                        dataset_name=dataset_name,
                        gen_func=generate_step,
                        gen_params={
                            "params": state.ema_model,
                            "rng": 0,
                            "apply_fn": lambda m, y, cfg: m(c=y, cfg_scale=cfg, train=False)["samples"],  # noqa: E731
                            "cfg_scale": eval_cfg,
                            "postprocess_fn": postprocess_fn,
                        },
                        eval_loader=eval_loader,
                        logger=logger,
                        num_samples=n_samples,
                        log_folder=f"{folder_prefix}{eval_cfg}",
                        log_prefix=f"EMA_{state.ema_decay:g}",
                        rng_eval=0,
                    )
                    fid_val = result.get("fid", float("inf"))
                    if fid_val < round_best_fid:
                        round_best_fid = fid_val
                        round_best_cfg = eval_cfg

                if not is_sanity:
                    log_for_0("best_fid=%.4f best_cfg=%.1f (step=%d)", round_best_fid, round_best_cfg, now_step)
                    logger.log_dict({"best_fid": round_best_fid, "best_cfg": round_best_cfg})
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    logger.finish()
    del model, optimizer, eval_loader, train_loader, state
    gc.collect()


def main_gen(config, output_dir="runs"):
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name

    from models.generator import DitGen

    set_global_mesh(config.get("hsdp_dim", min(8, max(1, _world_size()))))

    model_dict = build_model_dict(config, DitGen, workdir=output_dir)
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
    activation_fn, variables = build_activation_function(
        mae_path=mae_path,
        use_convnext=bool(feature_cfg.get("use_convnext", False)),
        use_mae=bool(feature_cfg.get("use_mae", True)),
        postprocess_fn=postprocess_fn_noclip,
        compile_level=compile_level,
    )
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
        **sanitize_train_config(config.train),
    )


def main(args):
    run_init()
    config = load_config(args.config)
    main_gen(config, output_dir=args.workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gen/latent_ablation.yaml", help="Path to configuration file.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    args = parser.parse_args()
    args.workdir = stamp_workdir(args.workdir)
    args.output_dir = args.workdir

    main(args)
