from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL


_vae_cache = {}


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_prefetched_vae() -> tuple[str, bool]:
    explicit = os.environ.get("SDVAE_PATH") or os.environ.get("DRIFT_SDVAE_PATH")
    if explicit:
        return explicit, True

    hf_home_value = os.environ.get("HF_HOME") or os.environ.get("HF_ROOT")
    if hf_home_value:
        hub_root = Path(hf_home_value).expanduser() / "hub" / "models--stabilityai--sd-vae-ft-mse" / "snapshots"
        if hub_root.is_dir():
            snapshots = sorted(p for p in hub_root.iterdir() if (p / "config.json").exists())
            if snapshots:
                return str(snapshots[-1]), True

    return "stabilityai/sd-vae-ft-mse", False


def vae_enc_decode():
    cache_key = ("vae_enc_decode",)
    if cache_key in _vae_cache:
        return _vae_cache[cache_key]

    device = _get_device()
    vae_path, local_only = _resolve_prefetched_vae()
    vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=local_only).to(device)
    vae.eval()

    @torch.inference_mode()
    def _encode_fn(images, model=vae):
        if isinstance(images, np.ndarray):
            images_t = torch.from_numpy(images)
        else:
            images_t = images
        images_t = images_t.to(device=device, dtype=torch.float32)
        dist = model.encode(images_t).latent_dist
        latents = dist.sample() * 0.18215
        return latents.permute(0, 2, 3, 1).detach().cpu()

    @torch.inference_mode()
    def _decode_fn(latents, model=vae):
        if isinstance(latents, np.ndarray):
            latents_t = torch.from_numpy(latents)
        else:
            latents_t = latents
        latents_t = latents_t.to(device=device, dtype=torch.float32)
        if latents_t.ndim != 4:
            raise ValueError(f"Expected 4D latents, got {latents_t.shape}")
        if latents_t.shape[1] != 4:
            latents_t = latents_t.permute(0, 3, 1, 2)
        out = model.decode(latents_t / 0.18215).sample
        return out.detach().cpu()

    result = (partial(_encode_fn), partial(_decode_fn))
    _vae_cache[cache_key] = result
    return result
