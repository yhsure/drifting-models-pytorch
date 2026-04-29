from __future__ import annotations

import time

import numpy as np
import torch
import torch.distributed as dist
from pytorch_fid.fid_score import calculate_frechet_distance
from torchmetrics.image.inception import InceptionScore

from dataset.dataset import epoch0_sampler
from utils.fid_util import _compute_features, _load_ref_stats, _to_uint8


def _gather_tensors(tensor: torch.Tensor, device: torch.device, world_size: int) -> torch.Tensor:
    local_size = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)

    padded = torch.zeros((max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=device)
    padded[: tensor.shape[0]] = tensor

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    return torch.cat([g[: s.item()] for g, s in zip(gathered, all_sizes)], dim=0)


def evaluate_fid_ddp(
    dataset_name,
    gen_func,
    gen_params,
    eval_loader,
    logger,
    num_samples=5000,
    log_folder="fid",
    log_prefix="gen_model",
    eval_prc_recall=False,
    eval_isc=True,
    eval_fid=True,
):
    start = time.time()
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    # Different noise per rank even when class labels are identical across ranks
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)

    local_n = num_samples // world_size + (1 if rank < num_samples % world_size else 0)

    eval_iter = epoch0_sampler(eval_loader)
    all_samples = []
    cur = 0
    for batch in eval_iter:
        gen_samples = gen_func(batch, **gen_params)
        if isinstance(gen_samples, torch.Tensor):
            gen_samples = gen_samples.detach().cpu()
            if gen_samples.dtype in (torch.bfloat16, torch.float16):
                gen_samples = gen_samples.float()
            gen_samples = gen_samples.numpy()
        all_samples.append(_to_uint8(gen_samples))
        cur += gen_samples.shape[0]
        if cur >= local_n:
            break
    local_samples = np.concatenate(all_samples, axis=0)[:local_n]

    metrics: dict[str, float] = {}

    if eval_fid:
        ref = _load_ref_stats(dataset_name)
        local_feats = _compute_features(local_samples, device=device)
        local_feats_t = torch.from_numpy(local_feats).to(device)
        all_feats_t = (
            _gather_tensors(local_feats_t, device, world_size)
            if is_dist and world_size > 1
            else local_feats_t
        )
        if rank == 0:
            all_feats = all_feats_t.cpu().numpy()[:num_samples].astype(np.float64)
            metrics["fid"] = float(
                calculate_frechet_distance(
                    ref["mu"], ref["sigma"],
                    np.mean(all_feats, axis=0),
                    np.cov(all_feats, rowvar=False),
                )
            )

    if eval_isc:
        samples_chw = local_samples
        if samples_chw.shape[-1] == 3:
            samples_chw = samples_chw.transpose(0, 3, 1, 2)
        metric = InceptionScore(normalize=False).to(device)
        with torch.inference_mode():
            for i in range(0, len(samples_chw), 128):
                x = torch.from_numpy(samples_chw[i : i + 128]).to(device=device, dtype=torch.uint8)
                metric.update(x)
        mean, std = metric.compute()
        if rank == 0:
            metrics["isc_mean"] = float(mean.item())
            metrics["isc_std"] = float(std.item())

    if rank == 0:
        metrics["fid_time"] = float(time.time() - start)
        if logger is not None:
            logger.log_dict({
                f"{log_folder}/{log_prefix}_{k}": v for k, v in metrics.items()
            } | {
                f"{log_folder}/step": getattr(logger, "step", 0),
            })
            if log_folder == "eval":
                logger.log_dict({
                    "samples/step": getattr(logger, "step", 0),
                    "samples/final_fid": metrics.get("fid", float("nan")),
                    "samples/final_isc_mean": metrics.get("isc_mean", float("nan")),
                })
                logger.log_image("samples/final_eval", local_samples[:36], max_images=36, grid_cols=6)
            else:
                logger.log_image(f"{log_folder}/{log_prefix}_viz", local_samples[:36], max_images=36, grid_cols=6)

    if is_dist:
        dist.barrier()

    return metrics
