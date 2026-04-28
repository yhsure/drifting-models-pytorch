from __future__ import annotations

import time

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torchmetrics.image.inception import InceptionScore

from dataset.dataset import epoch0_sampler
from utils.env import IMAGENET_FID_NPZ, IMAGENET_PR_NPZ
from utils.logging import log_for_0

_DATASET_STATS = {
    "imagenet256": IMAGENET_FID_NPZ,
}
_PR_REF_PATH = IMAGENET_PR_NPZ
INCEPTION_NET = None


def _canonical_dataset_name(name: str) -> str:
    n = name.lower()
    if "imagenet256" in n:
        return "imagenet256"
    raise ValueError(f"Only ImageNet is supported now, got: {name}")


def _to_uint8(samples):
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=0.0)
    return (samples * 255).clip(0, 255).astype(np.uint8)

def _compute_features(samples_uint8: np.ndarray, device: torch.device, batch_size: int = 200) -> np.ndarray:
    if samples_uint8.shape[-1] == 3:
        samples_uint8 = samples_uint8.transpose(0, 3, 1, 2)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    feats = []
    with torch.inference_mode():
        for i in range(0, samples_uint8.shape[0], batch_size):
            x = torch.from_numpy(samples_uint8[i : i + batch_size]).to(device=device, dtype=torch.float32) / 255.0
            pred = model(x)[0]
            pred = pred.squeeze(-1).squeeze(-1)
            feats.append(pred.cpu().numpy())
    return np.concatenate(feats, axis=0)


def _compute_stats(
    samples_uint8: np.ndarray,
    num_samples: int,
    *,
    compute_logits: bool,
    compute_features: bool,
    masks=None,
):
    del compute_logits, compute_features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if masks is None:
        masks = np.ones((len(samples_uint8),), dtype=np.float32)
    keep = masks > 0.5
    valid = samples_uint8[keep][:num_samples]
    feats = _compute_features(valid, device=device)
    feats64 = feats.astype(np.float64)
    return {
        "mu": np.mean(feats64, axis=0),
        "sigma": np.cov(feats64, rowvar=False),
        "features": feats,
    }


def _compute_inception_score_from_images(samples_uint8: np.ndarray, device: torch.device):
    if samples_uint8.shape[-1] == 3:
        samples_uint8 = samples_uint8.transpose(0, 3, 1, 2)
    metric = InceptionScore(normalize=False).to(device)
    with torch.inference_mode():
        for i in range(0, len(samples_uint8), 128):
            x = torch.from_numpy(samples_uint8[i : i + 128]).to(device=device, dtype=torch.uint8)
            metric.update(x)
    mean, std = metric.compute()
    return float(mean.item()), float(std.item())


def _load_ref_stats(dataset_name: str):
    canon = _canonical_dataset_name(dataset_name)
    path = _DATASET_STATS[canon]
    data = np.load(path)
    if "ref_mu" in data:
        return {"mu": data["ref_mu"], "sigma": data["ref_sigma"]}
    return {"mu": data["mu"], "sigma": data["sigma"]}


def evaluate_fid(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_iter = epoch0_sampler(eval_loader)
    all_samples = []
    cur = 0
    for i, batch in enumerate(eval_iter):
        _ = i
        gen_samples = gen_func(batch, **gen_params)
        if isinstance(gen_samples, torch.Tensor):
            gen_samples = gen_samples.detach().cpu().numpy()
        all_samples.append(_to_uint8(gen_samples))
        cur += gen_samples.shape[0]
        if cur >= num_samples:
            break

    samples = np.concatenate(all_samples, axis=0)[:num_samples]

    metrics: dict[str, float] = {}
    if eval_fid:
        ref = _load_ref_stats(dataset_name)
        stats = _compute_stats(samples, num_samples, compute_logits=eval_isc, compute_features=eval_prc_recall)
        metrics["fid"] = float(calculate_frechet_distance(ref["mu"], ref["sigma"], stats["mu"], stats["sigma"]))

    if eval_isc:
        mean, std = _compute_inception_score_from_images(samples, device=device)
        metrics["isc_mean"] = mean
        metrics["isc_std"] = std

    if eval_prc_recall:
        if _PR_REF_PATH and _PR_REF_PATH != "/path/to/imagenet_val_prc_arr0.npz":
            metrics["precision"] = float("nan")
            metrics["recall"] = float("nan")
        else:
            log_for_0("PR reference path not configured; skipping precision/recall.")

    metrics["fid_time"] = float(time.time() - start)
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
        logger.log_image("samples/final_eval", samples[:36], max_images=36, grid_cols=6)
    else:
        logger.log_image(f"{log_folder}/{log_prefix}_viz", samples[:36], max_images=36, grid_cols=6)
    return metrics
