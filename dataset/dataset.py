"""ImageNet-only dataset pipeline for Drift release (PyTorch)."""

from __future__ import annotations

import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if not os.environ.get("TORCHINDUCTOR_CACHE_DIR") or not os.environ.get("TORCH_HOME"):
    _torch_cache_root = Path.cwd() / ".torch-cache"
    _torch_cache_root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_torch_cache_root / "torchinductor")
    if not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = str(_torch_cache_root / "torch")
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset.latent import LatentDataset
from dataset.vae import vae_enc_decode
from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH
from utils.logging import log_for_0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _build_transforms(resolution: int, use_aug: bool, split: str):
    if use_aug and split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _build_imagenet_dataset(*, resolution: int, use_aug: bool, use_cache: bool, split: str):
    if use_cache:
        return LatentDataset(root=os.path.join(IMAGENET_CACHE_PATH, split))

    transform = _build_transforms(resolution, use_aug=use_aug, split=split)
    return ImageFolder(root=os.path.join(IMAGENET_PATH, split), transform=transform)


def _build_lazy_decode_fn():
    decode_fn = None

    def _decode(images):
        nonlocal decode_fn
        if decode_fn is None:
            _, decode_fn_local = vae_enc_decode()
            decode_fn = decode_fn_local
        return decode_fn(images)

    return _decode


def worker_init_fn(worker_id: int, rank: int) -> None:
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_imagenet_split(
    *,
    resolution: int,
    batch_size: int,
    split: str,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
):
    ds = _build_imagenet_dataset(
        resolution=resolution,
        use_aug=use_aug,
        use_cache=use_cache,
        split=split,
    )
    log_for_0(ds)

    rank = _rank()
    sampler = DistributedSampler(ds, num_replicas=_world_size(), rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=(split == "train"),
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    if use_latent or use_cache:
        decode_fn = _build_lazy_decode_fn() if use_cache else vae_enc_decode()[1]

        if use_cache:

            def preprocess_fn(batch):
                cached_latent, label = batch
                if not isinstance(cached_latent, torch.Tensor):
                    cached_latent = torch.as_tensor(cached_latent)
                if not isinstance(label, torch.Tensor):
                    label = torch.as_tensor(label)
                return {"images": cached_latent.float(), "labels": label.long()}

        else:
            encode_fn, _ = vae_enc_decode()

            def preprocess_fn(batch):
                image, label = batch
                if not isinstance(image, torch.Tensor):
                    image = torch.as_tensor(image)
                if not isinstance(label, torch.Tensor):
                    label = torch.as_tensor(label)
                return {"images": encode_fn(image).float(), "labels": label.long()}

        def postprocess_fn(images):
            if not isinstance(images, torch.Tensor):
                images = torch.as_tensor(images)
            return torch.clamp((decode_fn(images) + 1) / 2, 0, 1)

        return loader, preprocess_fn, postprocess_fn

    def preprocess_fn(batch):
        image, label = batch
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if not isinstance(label, torch.Tensor):
            label = torch.as_tensor(label)
        return {"images": image.permute(0, 2, 3, 1).float(), "labels": label.long()}

    def postprocess_fn(images):
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        return torch.clamp((images + 1) / 2, 0, 1).permute(0, 3, 1, 2)

    return loader, preprocess_fn, postprocess_fn


def get_postprocess_fn(*, use_aug: bool = False, use_latent: bool = False, use_cache: bool = False, has_clip: bool = True):
    if use_latent or use_cache:
        decode_fn = _build_lazy_decode_fn() if use_cache else vae_enc_decode()[1]

        def postprocess(images):
            if not isinstance(images, torch.Tensor):
                images_t = torch.as_tensor(images)
            else:
                images_t = images
            out = (decode_fn(images_t) + 1) / 2
            return torch.clamp(out, 0, 1) if has_clip else out

        return postprocess

    if use_aug or (not use_latent and not use_cache):

        def postprocess(images):
            if not isinstance(images, torch.Tensor):
                images_t = torch.as_tensor(images)
            else:
                images_t = images
            out = (images_t + 1) / 2
            out = torch.clamp(out, 0, 1) if has_clip else out
            return out.permute(0, 3, 1, 2)

        return postprocess

    raise ValueError("Unsupported dataset flags.")


def infinite_sampler(it, start_step: int = 0):
    step_per_epoch = len(it)
    epoch_idx = start_step // step_per_epoch
    if hasattr(it, "sampler") and hasattr(it.sampler, "set_epoch"):
        it.sampler.set_epoch(epoch_idx)
    skip_batches = start_step % step_per_epoch
    while True:
        for i, batch in enumerate(it):
            if skip_batches > 0 and i < skip_batches:
                continue
            yield batch
        skip_batches = 0
        epoch_idx += 1
        if hasattr(it, "sampler") and hasattr(it.sampler, "set_epoch"):
            it.sampler.set_epoch(epoch_idx)


def epoch0_sampler(it):
    if hasattr(it, "sampler") and hasattr(it.sampler, "set_epoch"):
        it.sampler.set_epoch(0)
    for batch in it:
        yield batch
