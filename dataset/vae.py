from __future__ import annotations

from functools import partial

import numpy as np
import torch
from diffusers import AutoencoderKL


_vae_cache = {}


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vae_enc_decode(replicate_params: bool = True):
    del replicate_params
    cache_key = ("vae_enc_decode",)
    if cache_key in _vae_cache:
        return _vae_cache[cache_key]

    device = _get_device()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    @torch.no_grad()
    def _encode_fn(images, rng=None, model=vae):
        del rng
        if isinstance(images, np.ndarray):
            images_t = torch.from_numpy(images)
        else:
            images_t = images
        images_t = images_t.to(device=device, dtype=torch.float32)
        dist = model.encode(images_t).latent_dist
        latents = dist.sample() * 0.18215
        return latents.permute(0, 2, 3, 1).detach().cpu()

    @torch.no_grad()
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
