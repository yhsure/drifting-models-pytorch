"""Latent cache dataset and cache builder for ImageNet release workflows (PyTorch)."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

_torch_cache_root = Path.cwd() / ".torch-cache"
_torch_cache_root.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_torch_cache_root / "torchinductor")
os.environ["TORCH_HOME"] = str(_torch_cache_root / "torch")
Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

from torchvision import datasets, transforms
from tqdm import tqdm

from dataset.vae import vae_enc_decode
from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH


@dataclass(frozen=True)
class _CacheWriteItem:
    output_path: str
    moments: np.ndarray
    moments_flip: np.ndarray


def _write_cache_file(item: _CacheWriteItem) -> None:
    output_path = Path(item.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
    torch.save(
        {
            "moments": item.moments,
            "moments_flip": item.moments_flip,
        },
        tmp_path,
    )
    os.replace(tmp_path, output_path)


class LatentDataset(datasets.DatasetFolder):
    def __init__(self, root: str):
        super().__init__(root=root, loader=str, extensions=(".pt",))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        moments = data["moments"] if torch.rand(1) < 0.5 else data["moments_flip"]
        return np.asarray(moments), target


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _center_crop_256(image: Image.Image) -> Image.Image:
    return center_crop_arr(image, 256)


class OriginalImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        rel_path = os.path.join(*path.split(os.path.sep)[-2:])
        return sample, target, rel_path


def _prepare_batch_data(images: torch.Tensor) -> np.ndarray:
    """Convert `(B,C,H,W)` tensor to host numpy for cache encoding."""
    return images.detach().cpu().numpy()


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def create_cached_dataset(
    local_batch_size: int,
    target_path: str,
    data_path: str,
    *,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    save_workers: int = 0,
) -> None:
    del save_workers
    encode_fn, _ = vae_enc_decode(replicate_params=False)

    Path(target_path, "train").mkdir(parents=True, exist_ok=True)
    Path(target_path, "val").mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Lambda(_center_crop_256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    world = _world_size()
    rank = _rank()

    for split in ("train", "val"):
        dataset = OriginalImageFolder(os.path.join(data_path, split), transform=transform)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": local_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
            "sampler": sampler,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = True
        loader = torch.utils.data.DataLoader(**loader_kwargs)

        for samples, _, rel_paths in tqdm(loader, total=len(loader), desc=f"cache:{split}:rank{rank}"):
            samples_np = _prepare_batch_data(samples)
            moments = encode_fn(samples_np).numpy()
            moments_flip = encode_fn(np.flip(samples_np, axis=3).copy()).numpy()

            for i, rel_path in enumerate(rel_paths):
                output_path = str(Path(target_path, split, rel_path).with_suffix(".pt"))
                _write_cache_file(
                    _CacheWriteItem(
                        output_path=output_path,
                        moments=np.asarray(moments[i]),
                        moments_flip=np.asarray(moments_flip[i]),
                    )
                )


def build_cache_from_args(args: argparse.Namespace) -> None:
    create_cached_dataset(
        local_batch_size=int(args.local_batch_size),
        target_path=args.target_path,
        data_path=args.data_path,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        save_workers=int(args.save_workers),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ImageNet latent cache files for release generator configs.")
    parser.add_argument("--data-path", default=IMAGENET_PATH, help="ImageNet root containing train/ and val/.")
    parser.add_argument("--target-path", default=IMAGENET_CACHE_PATH, help="Output cache root for latent .pt files.")
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=128,
        help="Per-process cache batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory for the cache build.")
    parser.add_argument(
        "--save-workers",
        type=int,
        default=0,
        help="Reserved for API compatibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    build_cache_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
