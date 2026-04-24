from __future__ import annotations

import argparse
import copy
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm

from dataset.dataset import epoch0_sampler, infinite_sampler
from models.mae_model import MAEResNet
from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from utils.env import HF_ROOT
from utils.hsdp_util import set_global_mesh
from utils.init_util import maybe_init_state_params
from utils.logging import is_rank_zero, log_for_0
from utils.misc import load_config, profile_func, run_init, stamp_workdir
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


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def input_dict(batch):
    return {"x": batch["images"], "labels": batch["labels"]}


def train_step(
    state: TrainState,
    batch,
    *,
    rng_init,
    forward_dict: dict,
    step_keys=("dropout", "masking"),
    learning_rate_fn: Any,
    preprocess_fn: Any,
    max_grad_norm: float = 2.0,
):
    del rng_init, step_keys
    batch = preprocess_fn(batch)
    batch = {k: v.to(state.device) for k, v in batch.items()}
    forward_kwargs = input_dict(batch)

    lr = learning_rate_fn(state.step)
    _set_lr(state.optimizer, lr)

    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    loss_vec, metric = state.model(
        **forward_kwargs,
        **forward_dict,
        train=True,
    )
    loss = loss_vec.mean()
    loss.backward()
    g_norm = float(torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_grad_norm).item())
    state.optimizer.step()

    with torch.no_grad():
        if state.ema_model is not None:
            for p_ema, p in zip(state.ema_model.parameters(), state.model.parameters()):
                p_ema.mul_(state.ema_decay).add_(p, alpha=(1.0 - state.ema_decay))

    out_metric = {k: float(v.mean().detach().cpu().item()) for k, v in metric.items()}
    out_metric["loss"] = float(loss.detach().cpu().item())
    out_metric["lr"] = float(lr)
    out_metric["g_norm"] = g_norm
    state.step += 1
    return state, out_metric


def eval_step(
    params,
    batch,
    rng_step,
    *,
    apply_fn,
    forward_dict,
    step_keys=("dropout", "masking"),
    preprocess_fn: Any,
):
    del rng_step, step_keys
    model = apply_fn
    device = next(model.parameters()).device
    batch = preprocess_fn(batch)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        loss, metric = model(
            **input_dict(batch),
            **forward_dict,
            train=False,
        )
        metric["loss"] = loss
    return metric


def eval_loop(
    state: TrainState,
    eval_loader,
    eval_step_func,
    *,
    eval_samples=5000,
    forward_dict=None,
    use_ema=False,
    rng_eval=None,
    ema_to_params_func=lambda x: x,
):
    del rng_eval, ema_to_params_func
    forward_dict = forward_dict or {}
    params = state.ema_model if use_ema else state.model
    if params is None:
        params = state.model
    params.eval()

    all_metrics = {}
    n_samples = 0

    for batch in epoch0_sampler(eval_loader):
        bsz = batch[0].shape[0]
        if n_samples + bsz > eval_samples:
            keep = eval_samples - n_samples
            batch = (batch[0][:keep], batch[1][:keep])
            bsz = keep
        metric = eval_step_func(
            params,
            batch,
            0,
            forward_dict=dict(forward_dict),
        )
        for k, v in metric.items():
            cur = float(v.mean().detach().cpu().item())
            all_metrics[k] = all_metrics.get(k, 0.0) + cur * bsz
        n_samples += bsz
        if n_samples >= eval_samples:
            break

    if n_samples == 0:
        return {}
    return {k: v / n_samples for k, v in all_metrics.items()}


def train_mae(
    *,
    model,
    optimizer,
    logger,
    eval_loader,
    train_loader,
    learning_rate_fn,
    forward_dict,
    eval_forward_dict,
    preprocess_fn,
    postprocess_fn,
    total_steps=100000,
    save_per_step=10000,
    eval_per_step=2000,
    eval_samples=5000,
    ema_decay=0.999,
    seed=42,
    finetune_last_steps=0,
    warmup_finetune=1000,
    finetune_cls=0.5,
    max_grad_norm=2.0,
    keep_every=500000,
    keep_last=2,
    init_from="",
    workdir="runs",
    model_config=None,
):
    del postprocess_fn, seed, keep_every

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ema_model = copy.deepcopy(model).to(device)

    state = TrainState(
        step=0,
        model=model,
        optimizer=optimizer,
        ema_model=ema_model,
        ema_decay=float(ema_decay[0] if isinstance(ema_decay, (list, tuple)) else ema_decay),
        device=device,
    )

    state = restore_checkpoint(state=state, workdir=workdir)
    if int(state.step) == 0 and init_from:
        log_for_0("Initializing MAE params from init_from=%s", init_from)
        state = maybe_init_state_params(
            state,
            model_type="mae",
            init_from=init_from,
            hf_cache_dir=HF_ROOT,
        )

    eval_step_jit = lambda params, batch, rng_step, forward_dict: eval_step(  # noqa: E731
        params,
        batch,
        rng_step,
        apply_fn=params,
        preprocess_fn=preprocess_fn,
        forward_dict=forward_dict,
    )

    forward_zeros_dict = copy.deepcopy(forward_dict)
    forward_zeros_dict["mask_ratio_min"] = 0.0
    forward_zeros_dict["mask_ratio_max"] = 0.0

    log_for_0("Starting MAE training loop...")
    step = int(state.step)
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    train_iter = infinite_sampler(train_loader, step)

    start_finetune_step = total_steps - finetune_last_steps
    start_time_all = time.time()
    for step in pbar:
        start_time = time.time()
        logger.set_step(step)

        batch = next(train_iter)
        finish_prepare = time.time()

        cur_dict = dict(copy.deepcopy(forward_dict))
        if step >= start_finetune_step:
            cur_dict["lambda_cls"] = finetune_cls * min(1.0, (step - start_finetune_step) / max(1, warmup_finetune))

        profile_metrics = {}
        if step == initial_step:
            profile_metrics = profile_func(
                lambda s, b, fd: train_step(
                    s,
                    b,
                    rng_init=0,
                    forward_dict=fd,
                    learning_rate_fn=learning_rate_fn,
                    preprocess_fn=preprocess_fn,
                    max_grad_norm=max_grad_norm,
                ),
                (state, batch, cur_dict),
                name="train_step",
            )

        state, metrics = train_step(
            state,
            batch,
            rng_init=0,
            forward_dict=cur_dict,
            learning_rate_fn=learning_rate_fn,
            preprocess_fn=preprocess_fn,
            max_grad_norm=max_grad_norm,
        )

        finish_train = time.time()
        metrics["kimg"] = (step - initial_step + 1) * batch[0].shape[0] / 1000.0
        metrics["time/total"] = finish_train - start_time
        metrics["time/prepare"] = finish_prepare - start_time
        metrics["time/train"] = finish_train - finish_prepare
        metrics["time/per_step"] = (finish_train - start_time_all) / (step - initial_step + 1)
        metrics.update(profile_metrics)
        logger.log_dict(metrics)

        if (step + 1) % eval_per_step == 0:
            eval_metrics = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=eval_forward_dict,
                use_ema=False,
            )
            logger.log_dict_dir("eval", eval_metrics)

            eval_metrics_ema = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=eval_forward_dict,
                use_ema=True,
            )
            logger.log_dict_dir(f"eval_ema_{state.ema_decay:g}", eval_metrics_ema)

            eval_metrics_nomask = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=forward_zeros_dict,
                use_ema=False,
            )
            logger.log_dict_dir("eval_nomask", eval_metrics_nomask)

            eval_metrics_nomask_ema = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=forward_zeros_dict,
                use_ema=True,
            )
            logger.log_dict_dir(f"eval_ema_{state.ema_decay:g}_nomask", eval_metrics_nomask_ema)

        now_step = step + 1
        if (now_step in [total_steps, start_finetune_step]) or (now_step % save_per_step == 0 and now_step < start_finetune_step):
            save_checkpoint(state, keep=keep_last, workdir=workdir)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="mae",
                model_config=model_config,
            )

    logger.finish()
    del model, optimizer, eval_loader, train_loader, state
    gc.collect()


def main_mae(config, output_dir="runs"):
    set_global_mesh(config.get("hsdp_dim", 1))
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name

    model_dict = build_model_dict(config, MAEResNet, workdir=output_dir)
    train_mae(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        model_config=dict(config.model),
        workdir=output_dir,
        **config.train,
    )


def main(args):
    config = load_config(args.config)
    main_mae(config, output_dir=args.workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to MAE config.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    args = parser.parse_args()
    args.workdir = stamp_workdir(args.workdir)
    args.output_dir = args.workdir
    main(args)
