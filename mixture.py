# %%
"""Latent-space stochastic mode mixture for Fashion MNIST and CIFAR-10.

This is a compact, self-contained implementation of the model principles in
the prompt, specialized to small image datasets:

1. train or load a small VAE and use its latent space as model space;
2. map noise eps to latent modes m = f_theta(eps), using either an MLP prior
   or a categorical codebook prior initialized from encoded training images;
3. sample local latents u = m + diag(std_psi(m)) eta;
4. compare real and generated latents in a feature space phi(z), either identity
   or a tiny classifier feature trained on frozen VAE latents;
5. optimize the Gaussian-mixture negative log likelihood of real features.

Run a small training job:
    .venv/bin/python mixture.py --device cuda

Run a quick GH200 smoke test:

    .venv/bin/python mixture.py --device cuda --max-train-examples 512 \
        --max-test-examples 512 --vae-epochs 1 --feature-epochs 1 \
        --steps 10 --mode-count 32 --batch-size 128

Run the CIFAR-10 UNet drifting branch:

    .venv/bin/python mixture.py --pipeline cifar_unet --dataset cifar10 \
        --device cuda --steps 5000

Run the CIFAR-10 particle-refiner branch:

    .venv/bin/python mixture.py --pipeline cifar_particle_refiner --dataset cifar10 \
        --device cuda --steps 2000

Run the CIFAR-10 drifting image-particle branch:

    .venv/bin/python mixture.py --pipeline cifar_drift_particles --dataset cifar10 \
        --device cuda --steps 5000

Run a CIFAR-10 rectified-flow baseline:

    .venv/bin/python mixture.py --pipeline cifar_rectified_flow --dataset cifar10 \
        --device cuda --steps 20000

Run a CIFAR-10 mixture-prior rectified flow:

    .venv/bin/python mixture.py --pipeline cifar_mixture_flow --dataset cifar10 \
        --device cuda --mode-count 512 --steps 20000

Run a CIFAR-10 class-mixture rectified flow:

    .venv/bin/python mixture.py --pipeline cifar_class_flow --dataset cifar10 \
        --device cuda --steps 20000

Run a CIFAR-10 explicit likelihood-mixture rectified flow:

    .venv/bin/python mixture.py --pipeline cifar_likelihood_flow --dataset cifar10 \
        --device cuda --mode-count 128 --steps 20000

Try a newer lucidrains flow objective inside the same likelihood-mixture model:

    .venv/bin/python mixture.py --pipeline cifar_likelihood_flow --dataset cifar10 \
        --device cuda --mode-count 128 --flow-objective mean

Evaluate a particle-refiner checkpoint:

    .venv/bin/python mixture.py --pipeline eval_cifar_particle_refiner --dataset cifar10 \
        --device cuda --eval-ckpt runs/.../cifar_particle_refiner.pt
"""

# %%
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50
from torchvision.utils import save_image
from tqdm import tqdm


# %%
@dataclass
class Config:
    pipeline: str = "latent_mixture"
    data_dir: str = "data"
    dataset: str = "fashion_mnist"
    out_dir: str = "runs/mixture"
    cache_dir: str = "runs/mixture_cache"
    seed: int = 7
    device: str = "auto"
    num_workers: int = 2
    max_train_examples: int = 0
    max_test_examples: int = 0

    latent_dim: int = 0
    vae_arch: str = "auto"
    vae_epochs: int = 12
    vae_batch_size: int = 256
    vae_lr: float = 1e-3
    vae_weight_decay: float = 0.0
    vae_beta: float = 0.01
    vae_recon_loss: str = "auto"
    vae_l1_weight: float = 0.0
    vae_perceptual_weight: float = 0.0
    vae_edge_weight: float = 0.0
    vae_sample_train: bool = False
    vae_ckpt: str = ""
    force_retrain_vae: bool = False

    feature_space: str = "classifier"
    feature_dim: int = 32
    feature_epochs: int = 6
    feature_lr: float = 1e-3
    feature_std_floor: float = 1e-4
    feature_ckpt: str = ""
    force_retrain_feature: bool = False
    feature_backbone: str = "resnet18"

    mode_family: str = "codebook"
    eps_dim: int = 32
    hidden_dim: int = 256
    steps: int = 3000
    batch_size: int = 256
    lr: float = 2e-4
    mode_count: int = 4096
    local_samples_per_mode: int = 1
    feature_sigma: float = 1.0
    local_min_std: float = 0.002
    local_max_std: float = 0.05
    std_l2_weight: float = 0.0
    codebook_anchor_weight: float = 0.05
    codebook_lr_scale: float = 1.0
    grad_clip: float = 5.0

    log_every: int = 50
    sample_every: int = 500
    save_every: int = 500
    grid_size: int = 64
    probe_samples: int = 4096
    sample_temperature: float = 0.75

    unet_dim: int = 64
    unet_dim_mults: str = "1,2,4"
    drifting_temps: str = "0.02,0.05,0.2"
    feature_stages: str = "2,4"
    no_std_features: bool = False
    ema_decay: float = 0.999
    preview_source: str = "ema"
    weight_decay: float = 1e-4
    particle_topk: int = 4
    particle_temp: float = 0.08
    particle_noise: float = 0.08
    particle_feature_size: int = 8
    particle_l1_weight: float = 1.0
    particle_mse_weight: float = 0.25
    particle_perceptual_weight: float = 0.1
    particle_residual_scale: float = 0.35
    eval_ckpt: str = ""
    eval_samples: int = 10000
    eval_batch_size: int = 256
    eval_nn_samples: int = 2048
    eval_fid_split: str = "test"
    drift_particle_init: str = "noise"
    drift_pixel_l2_weight: float = 0.0
    drift_tv_weight: float = 0.0
    flow_sample_steps: int = 32
    flow_eval_samples: int = 2048
    prior_std: float = 0.65
    prior_lr_scale: float = 1.0
    prior_tv_weight: float = 0.0
    prior_repulsion_weight: float = 0.0
    cond_dim: int = 128
    mixture_nll_weight: float = 0.02
    mixture_balance_weight: float = 0.02
    mixture_resp_temp: float = 0.7
    cond_dropout: float = 0.1
    flow_guidance_scale: float = 1.25
    likelihood_conditioning: str = "hard"
    flow_objective: str = "manual"
    resume_ckpt: str = ""


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for field_name, field_value in asdict(Config()).items():
        arg = "--" + field_name.replace("_", "-")
        value_type = type(field_value)
        if value_type is bool:
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=field_value)
        else:
            parser.add_argument(arg, type=value_type, default=field_value)
    cfg = Config(**vars(parser.parse_args()))
    supplied_args = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}

    def was_supplied(field_name: str) -> bool:
        arg = "--" + field_name.replace("_", "-")
        return arg in supplied_args or f"--no-{field_name.replace('_', '-')}" in supplied_args

    if cfg.pipeline not in {
        "latent_mixture",
        "cifar_unet",
        "cifar_particle_refiner",
        "cifar_drift_particles",
        "cifar_rectified_flow",
        "cifar_mixture_flow",
        "cifar_class_flow",
        "cifar_likelihood_flow",
        "eval_cifar_particle_refiner",
        "eval_cifar_likelihood_flow",
    }:
        raise ValueError(
            "--pipeline must be 'latent_mixture', 'cifar_unet', "
            "'cifar_particle_refiner', 'cifar_drift_particles', 'cifar_rectified_flow', "
            "'cifar_mixture_flow', 'cifar_class_flow', 'cifar_likelihood_flow', "
            "'eval_cifar_particle_refiner', or 'eval_cifar_likelihood_flow'"
        )
    if cfg.dataset not in {"fashion_mnist", "cifar10"}:
        raise ValueError("--dataset must be either 'fashion_mnist' or 'cifar10'")
    if cfg.vae_arch not in {"auto", "conv", "residual", "residual8"}:
        raise ValueError("--vae-arch must be 'auto', 'conv', 'residual', or 'residual8'")
    if cfg.vae_recon_loss not in {"auto", "bce", "mse"}:
        raise ValueError("--vae-recon-loss must be 'auto', 'bce', or 'mse'")
    if cfg.feature_space not in {"identity", "classifier"}:
        raise ValueError("--feature-space must be either 'identity' or 'classifier'")
    if cfg.likelihood_conditioning not in {"hard", "soft"}:
        raise ValueError("--likelihood-conditioning must be either 'hard' or 'soft'")
    if cfg.flow_objective not in {"manual", "nano", "mean", "split_mean", "lsd"}:
        raise ValueError("--flow-objective must be one of: 'manual', 'nano', 'mean', 'split_mean', 'lsd'")
    if cfg.eval_fid_split not in {"test", "train"}:
        raise ValueError("--eval-fid-split must be either 'test' or 'train'")
    if cfg.feature_backbone not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError("--feature-backbone must be one of: 'resnet18', 'resnet34', 'resnet50'")
    if cfg.mode_family not in {"codebook", "mlp"}:
        raise ValueError("--mode-family must be either 'codebook' or 'mlp'")
    if cfg.preview_source not in {"model", "ema"}:
        raise ValueError("--preview-source must be either 'model' or 'ema'")
    cifar_only_pipelines = {
        "cifar_unet",
        "cifar_particle_refiner",
        "cifar_drift_particles",
        "cifar_rectified_flow",
        "cifar_mixture_flow",
        "cifar_class_flow",
        "cifar_likelihood_flow",
        "eval_cifar_particle_refiner",
        "eval_cifar_likelihood_flow",
    }
    if cfg.pipeline in cifar_only_pipelines and cfg.dataset != "cifar10":
        raise ValueError(f"--pipeline {cfg.pipeline} currently supports only --dataset cifar10")
    if cfg.drift_particle_init not in {"noise", "data"}:
        raise ValueError("--drift-particle-init must be either 'noise' or 'data'")
    if cfg.latent_dim <= 0:
        cfg.latent_dim = 32 if cfg.dataset == "fashion_mnist" else 2048
    defaults = Config()
    if cfg.dataset == "cifar10":
        if cfg.vae_arch == defaults.vae_arch and not was_supplied("vae_arch"):
            cfg.vae_arch = "residual8"
        if cfg.vae_epochs == defaults.vae_epochs and not was_supplied("vae_epochs"):
            cfg.vae_epochs = 30
        if cfg.vae_lr == defaults.vae_lr and not was_supplied("vae_lr"):
            cfg.vae_lr = 2e-3
        if cfg.vae_beta == defaults.vae_beta and not was_supplied("vae_beta"):
            cfg.vae_beta = 0.0
        if cfg.vae_l1_weight == defaults.vae_l1_weight and not was_supplied("vae_l1_weight"):
            cfg.vae_l1_weight = 1.0
        if cfg.vae_perceptual_weight == defaults.vae_perceptual_weight and not was_supplied("vae_perceptual_weight"):
            cfg.vae_perceptual_weight = 0.5
        if cfg.vae_edge_weight == defaults.vae_edge_weight and not was_supplied("vae_edge_weight"):
            cfg.vae_edge_weight = 0.5
        if cfg.feature_dim == defaults.feature_dim and not was_supplied("feature_dim"):
            cfg.feature_dim = 64
        if cfg.feature_std_floor == defaults.feature_std_floor and not was_supplied("feature_std_floor"):
            cfg.feature_std_floor = 1.0
        if cfg.mode_count == defaults.mode_count and not was_supplied("mode_count"):
            cfg.mode_count = 8192
        if cfg.feature_sigma == defaults.feature_sigma and not was_supplied("feature_sigma"):
            cfg.feature_sigma = 5.0
        if cfg.local_min_std == defaults.local_min_std and not was_supplied("local_min_std"):
            cfg.local_min_std = 0.0
        if cfg.local_max_std == defaults.local_max_std and not was_supplied("local_max_std"):
            cfg.local_max_std = 0.0
        if cfg.codebook_anchor_weight == defaults.codebook_anchor_weight and not was_supplied("codebook_anchor_weight"):
            cfg.codebook_anchor_weight = 1.0
        if cfg.codebook_lr_scale == defaults.codebook_lr_scale and not was_supplied("codebook_lr_scale"):
            cfg.codebook_lr_scale = 0.0
        if cfg.sample_temperature == defaults.sample_temperature and not was_supplied("sample_temperature"):
            cfg.sample_temperature = 0.7
        if cfg.pipeline == "cifar_likelihood_flow" and cfg.mode_count == 8192:
            cfg.mode_count = 128
    return cfg


# %%
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle(loader: Iterable):
    while True:
        yield from loader


def training_size_tag(cfg: Config) -> str:
    return f"n{cfg.max_train_examples}" if cfg.max_train_examples > 0 else "full"


def default_vae_ckpt_path(cfg: Config, out_dir: Path) -> Path:
    if not cfg.cache_dir:
        return out_dir / f"{cfg.dataset}_vae.pt"
    sample_tag = "sample" if cfg.vae_sample_train else "det"
    return (
        Path(cfg.cache_dir)
        / (
            f"{cfg.dataset}_vae_{cfg.vae_arch}_{training_size_tag(cfg)}_z{cfg.latent_dim}_e{cfg.vae_epochs}"
            f"_lr{cfg.vae_lr:g}_wd{cfg.vae_weight_decay:g}_b{cfg.vae_beta:g}_r{cfg.vae_recon_loss}_l1{cfg.vae_l1_weight:g}"
            f"_p{cfg.vae_perceptual_weight:g}_edge{cfg.vae_edge_weight:g}_{sample_tag}.pt"
        )
    )


def default_feature_ckpt_path(cfg: Config, out_dir: Path) -> Path:
    if not cfg.cache_dir:
        return out_dir / f"{cfg.dataset}_latent_feature.pt"
    return (
        Path(cfg.cache_dir)
        / (
            f"{cfg.dataset}_phi_{cfg.feature_space}_{training_size_tag(cfg)}_z{cfg.latent_dim}"
            f"_d{cfg.feature_dim}_arch{cfg.vae_arch}_vae{cfg.vae_epochs}_vlr{cfg.vae_lr:g}"
            f"_vwd{cfg.vae_weight_decay:g}_b{cfg.vae_beta:g}"
            f"_r{cfg.vae_recon_loss}_l1{cfg.vae_l1_weight:g}_p{cfg.vae_perceptual_weight:g}"
            f"_edge{cfg.vae_edge_weight:g}_std{cfg.feature_std_floor:g}_e{cfg.feature_epochs}.pt"
        )
    )


def image_shape(cfg: Config) -> tuple[int, int, int]:
    if cfg.dataset == "fashion_mnist":
        return 1, 28, 28
    return 3, 32, 32


def make_image_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    if cfg.dataset == "fashion_mnist":
        train_ds: Dataset = datasets.FashionMNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_ds: Dataset = datasets.FashionMNIST(root=cfg.data_dir, train=False, download=True, transform=transform)
    else:
        train_ds = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=transform)
    if cfg.max_train_examples > 0:
        train_ds = Subset(train_ds, list(range(min(cfg.max_train_examples, len(train_ds)))))
    if cfg.max_test_examples > 0:
        test_ds = Subset(test_ds, list(range(min(cfg.max_test_examples, len(test_ds)))))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.vae_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.vae_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# %%
class FashionMNISTVAE(nn.Module):
    """Small VAE used as the Fashion MNIST equivalent of the pretrained SDVAE."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.SiLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encode_stats(x)
        if not sample:
            return mu
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(z.shape[0], 64, 7, 7)
        return self.decoder(h)

    def decode(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("decode temperature must be positive")
        return torch.sigmoid(self.decode_logits(z) / temperature)

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode_logits(z), mu, logvar


class CIFAR10VAE(nn.Module):
    """Small RGB VAE used as the CIFAR-10 decodable model space."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(192, 384, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(384 * 4 * 4, 768),
            nn.SiLU(),
        )
        self.mu = nn.Linear(768, latent_dim)
        self.logvar = nn.Linear(768, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 768),
            nn.SiLU(),
            nn.Linear(768, 384 * 4 * 4),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(96, 3, kernel_size=4, stride=2, padding=1),
        )

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encode_stats(x)
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(z.shape[0], 384, 4, 4)
        return self.decoder(h)

    def decode(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("decode temperature must be positive")
        return torch.sigmoid(self.decode_logits(z) / temperature)

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode_logits(z), mu, logvar


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CIFAR10ResidualVAE(nn.Module):
    """Higher-capacity deterministic-friendly CIFAR autoencoder with VAE-compatible methods."""

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 384),
            ResidualBlock(384),
            nn.Flatten(),
            nn.Linear(384 * 4 * 4, 1024),
            nn.SiLU(),
        )
        self.mu = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 384 * 4 * 4),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            ResidualBlock(384),
            UpsampleBlock(384, 256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encode_stats(x)
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(z.shape[0], 384, 4, 4)
        return self.decoder(h)

    def decode(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("decode temperature must be positive")
        return torch.sigmoid(self.decode_logits(z) / temperature)

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode_logits(z), mu, logvar


class CIFAR10Residual8VAE(nn.Module):
    """CIFAR autoencoder with an 8x8 spatial bottleneck for sharper small images."""

    def __init__(self, latent_dim: int = 2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            ResidualBlock(64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            ResidualBlock(256),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 2048),
            nn.SiLU(),
        )
        self.mu = nn.Linear(2048, latent_dim)
        self.logvar = nn.Linear(2048, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 256 * 8 * 8),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encode_stats(x)
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(z.shape[0], 256, 8, 8)
        return self.decoder(h)

    def decode(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("decode temperature must be positive")
        return torch.sigmoid(self.decode_logits(z) / temperature)

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode_logits(z), mu, logvar


class CIFARPerceptualLoss(nn.Module):
    """Frozen low-level ResNet loss for CIFAR autoencoder sharpness."""

    def __init__(self):
        super().__init__()
        try:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        for p in self.parameters():
            p.requires_grad_(False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x.clamp(0.0, 1.0) - mean) / std

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(self.normalize(x))
        f1 = self.layer1(h)
        f2 = self.layer2(f1)
        return f1, f2

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recon_features = self.features(recon)
        with torch.no_grad():
            target_features = self.features(target)
        return sum(F.l1_loss(r, t) for r, t in zip(recon_features, target_features, strict=True))


def build_imagenet_resnet(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet18":
        model_fn = resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1
    elif backbone_name == "resnet34":
        model_fn = resnet34
        weights = ResNet34_Weights.IMAGENET1K_V1
    elif backbone_name == "resnet50":
        model_fn = resnet50
        weights = ResNet50_Weights.IMAGENET1K_V2
    else:
        raise ValueError(f"unknown ResNet backbone: {backbone_name}")

    try:
        return model_fn(weights=weights)
    except Exception:
        return model_fn(weights=None)


def vae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    recon_loss: str,
    l1_weight: float = 0.0,
    perceptual_weight: float = 0.0,
    edge_weight: float = 0.0,
    perceptual_loss: CIFARPerceptualLoss | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon_img = torch.sigmoid(logits)
    if recon_loss == "mse":
        recon = F.mse_loss(recon_img, x, reduction="none").flatten(1).sum(dim=1).mean()
    else:
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(dim=1).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    l1 = F.l1_loss(recon_img, x, reduction="none").flatten(1).sum(dim=1).mean()
    if perceptual_loss is not None and perceptual_weight > 0:
        perceptual = perceptual_loss(recon_img, x)
    else:
        perceptual = torch.zeros((), device=x.device, dtype=x.dtype)
    edge = image_edge_loss(recon_img, x) if edge_weight > 0 else torch.zeros((), device=x.device, dtype=x.dtype)
    total = recon + beta * kl + l1_weight * l1 + perceptual_weight * perceptual + edge_weight * edge
    return total, {"recon": recon, "kl": kl, "l1": l1, "perceptual": perceptual, "edge": edge}


def image_edge_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    recon_dx = recon[..., :, 1:] - recon[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    recon_dy = recon[..., 1:, :] - recon[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    dx_loss = F.l1_loss(recon_dx, target_dx, reduction="none").flatten(1).sum(dim=1).mean()
    dy_loss = F.l1_loss(recon_dy, target_dy, reduction="none").flatten(1).sum(dim=1).mean()
    return dx_loss + dy_loss


def resolved_vae_recon_loss(cfg: Config) -> str:
    if cfg.vae_recon_loss == "auto":
        return "mse" if cfg.dataset == "cifar10" else "bce"
    return cfg.vae_recon_loss


def build_vae(cfg: Config) -> nn.Module:
    if cfg.dataset == "fashion_mnist":
        return FashionMNISTVAE(latent_dim=cfg.latent_dim)
    if cfg.vae_arch == "residual8":
        return CIFAR10Residual8VAE(latent_dim=cfg.latent_dim)
    if cfg.vae_arch == "residual":
        return CIFAR10ResidualVAE(latent_dim=cfg.latent_dim)
    return CIFAR10VAE(latent_dim=cfg.latent_dim)


def train_or_load_vae(cfg: Config, train_loader: DataLoader, device: torch.device, out_dir: Path) -> nn.Module:
    vae = build_vae(cfg).to(device)
    ckpt_path = Path(cfg.vae_ckpt) if cfg.vae_ckpt else default_vae_ckpt_path(cfg, out_dir)
    if ckpt_path.exists() and not cfg.force_retrain_vae:
        ckpt = torch.load(ckpt_path, map_location=device)
        vae.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        vae.eval()
        print(f"Loaded VAE from {ckpt_path}")
        return vae

    if cfg.vae_epochs <= 0:
        print("Using randomly initialized VAE because --vae-epochs <= 0 and no checkpoint was found.")
        return vae

    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_weight_decay)
    recon_loss_name = resolved_vae_recon_loss(cfg)
    perceptual_loss = None
    if cfg.dataset == "cifar10" and cfg.vae_perceptual_weight > 0:
        perceptual_loss = CIFARPerceptualLoss().to(device).eval()
    vae.train()
    for epoch in range(cfg.vae_epochs):
        losses = []
        recon_losses = []
        l1_losses = []
        perceptual_losses = []
        edge_losses = []
        pbar = tqdm(train_loader, desc=f"vae epoch {epoch + 1}/{cfg.vae_epochs}")
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            logits, mu, logvar = vae(x, sample=cfg.vae_sample_train)
            loss, parts = vae_loss(
                logits,
                x,
                mu,
                logvar,
                beta=cfg.vae_beta,
                recon_loss=recon_loss_name,
                l1_weight=cfg.vae_l1_weight,
                perceptual_weight=cfg.vae_perceptual_weight,
                edge_weight=cfg.vae_edge_weight,
                perceptual_loss=perceptual_loss,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(parts["recon"].detach().cpu()))
            l1_losses.append(float(parts["l1"].detach().cpu()))
            perceptual_losses.append(float(parts["perceptual"].detach().cpu()))
            edge_losses.append(float(parts["edge"].detach().cpu()))
            pbar.set_postfix(
                loss=sum(losses[-50:]) / len(losses[-50:]),
                recon=sum(recon_losses[-50:]) / len(recon_losses[-50:]),
                l1=sum(l1_losses[-50:]) / len(l1_losses[-50:]),
                perc=sum(perceptual_losses[-50:]) / len(perceptual_losses[-50:]),
                edge=sum(edge_losses[-50:]) / len(edge_losses[-50:]),
            )
    vae.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": vae.state_dict(), "config": asdict(cfg)}, ckpt_path)
    print(f"Saved VAE to {ckpt_path}")
    return vae


# %%
@torch.inference_mode()
def encode_dataset(vae: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> TensorDataset:
    latents, labels = [], []
    vae.eval()
    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        z = vae.encode(x, sample=False).cpu()
        latents.append(z)
        labels.append(y.cpu())
    return TensorDataset(torch.cat(latents, dim=0), torch.cat(labels, dim=0))


class IdentityFeature(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class LatentClassifierFeature(nn.Module):
    """A tiny optional phi(z): train classifier, use its hidden layer as features."""

    def __init__(self, latent_dim: int, feature_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, feature_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(feature_dim, 10)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.trunk(z)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward(z))


def train_feature_model(cfg: Config, latent_ds: TensorDataset, device: torch.device, out_dir: Path) -> nn.Module:
    if cfg.feature_space == "identity":
        return IdentityFeature().to(device)

    feature = LatentClassifierFeature(cfg.latent_dim, cfg.feature_dim).to(device)
    ckpt_path = Path(cfg.feature_ckpt) if cfg.feature_ckpt else default_feature_ckpt_path(cfg, out_dir)
    if ckpt_path.exists() and not cfg.force_retrain_feature:
        feature.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        feature.eval()
        print(f"Loaded latent feature model from {ckpt_path}")
        return feature

    loader = DataLoader(
        latent_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    opt = torch.optim.AdamW(feature.parameters(), lr=cfg.feature_lr)
    feature.train()
    for epoch in range(cfg.feature_epochs):
        correct = 0
        total = 0
        losses = []
        pbar = tqdm(loader, desc=f"feature epoch {epoch + 1}/{cfg.feature_epochs}")
        for z, y in pbar:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = feature.logits(z)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += int((logits.argmax(dim=1) == y).sum())
            total += int(y.numel())
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=sum(losses[-50:]) / len(losses[-50:]), acc=correct / max(1, total))
    feature.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": feature.state_dict(), "config": asdict(cfg)}, ckpt_path)
    print(f"Saved latent feature model to {ckpt_path}")
    return feature


@torch.inference_mode()
def fit_feature_stats(
    phi: nn.Module,
    latent_ds: TensorDataset,
    device: torch.device,
    batch_size: int,
    std_floor: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(latent_ds, batch_size=batch_size, shuffle=False)
    feats = []
    phi.eval()
    for z, _ in loader:
        feats.append(phi(z.to(device)).cpu())
    all_feats = torch.cat(feats, dim=0)
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0).clamp_min(std_floor)
    return mean, std


# %%
class ModeGenerator(nn.Module):
    def __init__(self, eps_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(eps_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        return self.net(eps)


class CodebookGenerator(nn.Module):
    """Categorical f_theta(eps): eps selects a learnable latent-space mode."""

    def __init__(self, initial_modes: torch.Tensor):
        super().__init__()
        self.modes = nn.Parameter(initial_modes.detach().clone())
        self.register_buffer("initial_modes", initial_modes.detach().clone())

    @property
    def mode_count(self) -> int:
        return self.modes.shape[0]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.modes[indices]

    def all_modes(self) -> torch.Tensor:
        return self.modes

    def anchor_loss(self) -> torch.Tensor:
        return (self.modes - self.initial_modes).pow(2).sum(dim=1).mean()


class DiagonalCovariance(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, min_std: float, max_std: float):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, mode: torch.Tensor) -> torch.Tensor:
        raw = self.net(mode)
        return self.min_std + (self.max_std - self.min_std) * torch.sigmoid(raw)


class StochasticModeMixture(nn.Module):
    def __init__(self, cfg: Config, initial_modes: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        if cfg.mode_family == "codebook":
            if initial_modes is None:
                initial_modes = torch.randn(cfg.mode_count, cfg.latent_dim)
            self.generator = CodebookGenerator(initial_modes)
        else:
            self.generator = ModeGenerator(cfg.eps_dim, cfg.latent_dim, cfg.hidden_dim)
        self.covariance = DiagonalCovariance(cfg.latent_dim, cfg.hidden_dim, cfg.local_min_std, cfg.local_max_std)

    def sample_modes(self, n_modes: int) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(self.generator, CodebookGenerator):
            if n_modes == self.generator.mode_count:
                return self.generator.all_modes()
            if n_modes < self.generator.mode_count:
                indices = torch.randperm(self.generator.mode_count, device=device)[:n_modes]
            else:
                indices = torch.randint(self.generator.mode_count, (n_modes,), device=device)
            return self.generator(indices)

        eps = torch.randn(n_modes, self.cfg.eps_dim, device=device)
        return self.generator(eps)

    def anchor_loss(self) -> torch.Tensor:
        if isinstance(self.generator, CodebookGenerator):
            return self.generator.anchor_loss()
        return torch.zeros((), device=next(self.parameters()).device)

    def sample_latents(
        self,
        n_modes: int,
        samples_per_mode: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modes = self.sample_modes(n_modes)
        std = self.covariance(modes)
        if samples_per_mode > 1:
            modes = modes[:, None, :].expand(n_modes, samples_per_mode, -1).reshape(n_modes * samples_per_mode, -1)
            std = std[:, None, :].expand(n_modes, samples_per_mode, -1).reshape(n_modes * samples_per_mode, -1)
        eta = torch.randn_like(modes)
        latents = modes + std * eta
        return latents, modes, std

    @torch.inference_mode()
    def sample_images(self, vae: nn.Module, n: int) -> torch.Tensor:
        z, _, _ = self.sample_latents(n_modes=n, samples_per_mode=1)
        return vae.decode(z, temperature=self.cfg.sample_temperature)


def standardized_features(phi: nn.Module, z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (phi(z) - mean.to(z.device)) / std.to(z.device)


def gaussian_mixture_nll(real_features: torch.Tensor, gen_features: torch.Tensor, sigma: float) -> torch.Tensor:
    """NLL under equally weighted isotropic Gaussian components in feature space."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    feat_dim = real_features.shape[-1]
    diff = real_features[:, None, :] - gen_features[None, :, :]
    sq_dist = diff.pow(2).sum(dim=-1)
    log_component = -0.5 * sq_dist / (sigma * sigma)
    log_component = log_component - feat_dim * math.log(sigma) - 0.5 * feat_dim * math.log(2.0 * math.pi)
    return -(torch.logsumexp(log_component, dim=1) - math.log(gen_features.shape[0])).mean()


def append_metric(path: Path, metric: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metric) + "\n")


# %%
@torch.inference_mode()
def save_reconstruction_grid(
    vae: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: Path,
    cfg: Config,
    n: int = 32,
) -> None:
    vae.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    with torch.inference_mode():
        recon = vae.decode(vae.encode(x, sample=False))
    c, h, w = image_shape(cfg)
    paired = torch.stack([x, recon], dim=1).reshape(-1, c, h, w)
    save_image(paired, path, nrow=8)


@torch.inference_mode()
def save_sample_grid(model: StochasticModeMixture, vae: nn.Module, path: Path, n: int) -> None:
    model.eval()
    vae.eval()
    samples = model.sample_images(vae, n)
    save_image(samples.cpu(), path, nrow=int(math.sqrt(n)))


def evaluate_nll(
    model: StochasticModeMixture,
    phi: nn.Module,
    latent_ds: TensorDataset,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> float:
    loader = DataLoader(latent_ds, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    phi.eval()
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        gen_z, _, _ = model.sample_latents(cfg.mode_count, cfg.local_samples_per_mode)
        gen_feat = standardized_features(phi, gen_z, feature_mean, feature_std)
        for z, _ in loader:
            z = z.to(device)
            real_feat = standardized_features(phi, z, feature_mean, feature_std)
            batch_nll = gaussian_mixture_nll(real_feat, gen_feat, cfg.feature_sigma)
            loss_sum += float(batch_nll.cpu()) * z.shape[0]
            count += z.shape[0]
    return loss_sum / max(1, count)


@torch.inference_mode()
def generation_probe(model: StochasticModeMixture, phi: nn.Module, cfg: Config) -> dict:
    if cfg.probe_samples <= 0:
        return {}

    model.eval()
    phi.eval()
    gen_z, _, local_std = model.sample_latents(cfg.probe_samples, 1)
    metric = {
        "probe_std_mean": float(local_std.mean().cpu()),
        "probe_std_min": float(local_std.min().cpu()),
        "probe_std_max": float(local_std.max().cpu()),
    }

    logits_fn = getattr(phi, "logits", None)
    if callable(logits_fn):
        probs = logits_fn(gen_z).softmax(dim=-1).mean(dim=0)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum()
        metric["probe_class_entropy"] = float(entropy.cpu())
        metric["probe_class_probs"] = [float(x) for x in probs.cpu()]

    return metric


def initialize_modes_from_latents(latent_ds: TensorDataset, cfg: Config, device: torch.device) -> torch.Tensor | None:
    if cfg.mode_family != "codebook":
        return None

    latents = latent_ds.tensors[0]
    if cfg.mode_count <= latents.shape[0]:
        indices = torch.randperm(latents.shape[0])[: cfg.mode_count]
        modes = latents[indices]
    else:
        repeats = math.ceil(cfg.mode_count / latents.shape[0])
        modes = latents.repeat((repeats, 1))[: cfg.mode_count]
        modes = modes + 0.01 * torch.randn_like(modes)
    return modes.to(device)


def train_mixture(
    cfg: Config,
    vae: nn.Module,
    phi: nn.Module,
    latent_ds: TensorDataset,
    test_latent_ds: TensorDataset,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
    out_dir: Path,
) -> StochasticModeMixture:
    latent_loader = DataLoader(
        latent_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    batches = cycle(latent_loader)
    initial_modes = initialize_modes_from_latents(latent_ds, cfg, device)
    model = StochasticModeMixture(cfg, initial_modes=initial_modes).to(device)
    param_groups: list[dict] = [{"params": model.covariance.parameters(), "lr": cfg.lr}]
    if isinstance(model.generator, CodebookGenerator):
        if cfg.codebook_lr_scale > 0:
            param_groups.append({"params": [model.generator.modes], "lr": cfg.lr * cfg.codebook_lr_scale})
        else:
            model.generator.modes.requires_grad_(False)
    else:
        param_groups.append({"params": model.generator.parameters(), "lr": cfg.lr})
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr)
    metrics_path = out_dir / "metrics.jsonl"

    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)

    pbar = tqdm(range(1, cfg.steps + 1), desc="mixture")
    for step in pbar:
        real_z, _ = next(batches)
        real_z = real_z.to(device, non_blocking=True)

        with torch.no_grad():
            real_feat = standardized_features(phi, real_z, feature_mean, feature_std)

        gen_z, _, local_std = model.sample_latents(cfg.mode_count, cfg.local_samples_per_mode)
        gen_feat = standardized_features(phi, gen_z, feature_mean, feature_std)
        nll = gaussian_mixture_nll(real_feat, gen_feat, cfg.feature_sigma)
        std_l2 = local_std.pow(2).mean()
        anchor = model.anchor_loss()
        loss = nll + cfg.std_l2_weight * std_l2 + cfg.codebook_anchor_weight * anchor

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        metric = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "std_l2": float(std_l2.detach().cpu()),
            "anchor": float(anchor.detach().cpu()),
            "std_mean": float(local_std.mean().detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        pbar.set_postfix(nll=metric["nll"], std=metric["std_mean"])

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)

        if cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0):
            save_sample_grid(model, vae, out_dir / f"samples_step{step:06d}.png", cfg.grid_size)

        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)
        if do_save:
            test_nll = evaluate_nll(model, phi, test_latent_ds, feature_mean, feature_std, cfg, device)
            metric["test_nll"] = test_nll
            metric.update(generation_probe(model, phi, cfg))
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "step": step,
                    "test_nll": test_nll,
                    "generation_probe": {
                        k: v
                        for k, v in metric.items()
                        if k.startswith("probe_")
                    },
                },
                out_dir / f"{cfg.dataset}_mixture.pt",
            )
            print(f"step {step}: test_nll={test_nll:.4f}; saved {out_dir / f'{cfg.dataset}_mixture.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    return model


# %%
def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return values


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return values


class DriftingUNetGenerator(nn.Module):
    """Reference-style rectified-flow UNet generator for CIFAR-10 pixels."""

    def __init__(self, dim: int, dim_mults: tuple[int, ...]):
        super().__init__()
        try:
            from rectified_flow_pytorch import Unet
        except ImportError as exc:
            raise RuntimeError(
                "The CIFAR UNet branch needs rectified-flow-pytorch. "
                "Install it with: uv pip install --python .venv/bin/python rectified-flow-pytorch"
            ) from exc
        self.unet = Unet(dim=dim, channels=3, dim_mults=dim_mults, accept_time=False)

    def forward(self, noise_img: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.unet(noise_img))


def build_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    one_minus = 1.0 - decay
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, p in model_params.items():
        ema_params[name].mul_(decay).add_(p.detach(), alpha=one_minus)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, b in model_buffers.items():
        ema_buffers[name].copy_(b)


def effective_ema_decay(target_decay: float, step: int) -> float:
    warm = (1.0 + float(step)) / (10.0 + float(step))
    return min(float(target_decay), warm)


class FrozenFeatureEncoder(nn.Module):
    """Frozen multi-scale ResNet descriptor for the CIFAR UNet feature geometry."""

    def __init__(self, stage_ids: tuple[int, ...] = (2, 4), backbone_name: str = "resnet18"):
        super().__init__()
        self.stage_ids = stage_ids
        self.backbone_name = backbone_name
        backbone = build_imagenet_resnet(backbone_name)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        x = (images + 1.0) * 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        feats: dict[int, torch.Tensor] = {}
        x = self.stem(x)
        x = self.layer1(x)
        feats[1] = x
        x = self.layer2(x)
        feats[2] = x
        x = self.layer3(x)
        feats[3] = x
        x = self.layer4(x)
        feats[4] = x
        return {sid: feats[sid] for sid in self.stage_ids}


def feature_descriptors(
    images: torch.Tensor,
    encoder: FrozenFeatureEncoder,
    use_std: bool = True,
    include_pixel_descriptor: bool = False,
) -> list[torch.Tensor]:
    feats = encoder(images)
    descs: list[torch.Tensor] = []

    for feat in feats.values():
        parts = [feat.mean(dim=(2, 3))]
        if use_std:
            parts.append(feat.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6))
        parts.append(F.adaptive_avg_pool2d(feat, output_size=(2, 2)).flatten(1))
        descs.append(torch.cat(parts, dim=1))
    if include_pixel_descriptor:
        descs.append((images**2).mean(dim=(2, 3)))
    return descs


def compute_drifting_field(x: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor, tau: float) -> torch.Tensor:
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)
    if x.shape[0] == y_neg.shape[0]:
        diag = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(diag, 1e6)

    logits = torch.cat((-dist_pos / tau, -dist_neg / tau), dim=1)
    aff_row = logits.softmax(dim=-1)
    aff_col = logits.softmax(dim=-2)
    affinity = torch.sqrt((aff_row * aff_col).clamp_min(1e-12))

    n_pos = y_pos.shape[0]
    a_pos = affinity[:, :n_pos]
    a_neg = affinity[:, n_pos:]
    w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)
    return w_pos @ y_pos - w_neg @ y_neg


def single_descriptor_drifting_loss(
    generated: torch.Tensor,
    real: torch.Tensor,
    temps: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    dim = generated.shape[1]
    sqrt_dim = dim**0.5

    all_samples = torch.cat((real, generated.detach()), dim=0)
    scale = (torch.cdist(generated, all_samples).mean() / sqrt_dim).detach().clamp_min(1e-6)
    generated_norm = generated / scale
    real_norm = real / scale

    force = torch.zeros_like(generated_norm)
    info: dict[str, float] = {"scale": float(scale.item())}
    for tau in temps:
        tau_eff = float(tau) * sqrt_dim
        v = compute_drifting_field(generated_norm, real_norm, generated_norm, tau=tau_eff)
        v_norm = v.square().mean().clamp_min(1e-8).sqrt()
        force = force + v / v_norm
        info[f"force_norm_tau_{tau}"] = float(v_norm.detach().item())

    target = (generated_norm + force).detach()
    return F.mse_loss(generated_norm, target), info


def feature_space_drifting_loss(
    gen_images: torch.Tensor,
    pos_images: torch.Tensor,
    encoder: FrozenFeatureEncoder,
    temps: tuple[float, ...],
    use_std: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    gen_descs = feature_descriptors(gen_images, encoder=encoder, use_std=use_std)
    with torch.no_grad():
        pos_descs = feature_descriptors(pos_images, encoder=encoder, use_std=use_std)

    total_loss = torch.zeros((), device=gen_images.device)
    scales = []
    for g, p in zip(gen_descs, pos_descs, strict=True):
        loss_i, info_i = single_descriptor_drifting_loss(g.float(), p.float(), temps=temps)
        total_loss = total_loss + loss_i
        scales.append(info_i["scale"])
    return total_loss, {"scale": float(sum(scales) / max(1, len(scales)))}


def save_tanh_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    save_image((images.clamp(-1.0, 1.0) + 1.0) * 0.5, path, nrow=nrow)


def make_cifar_unet_loader(cfg: Config) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset: Dataset = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=transform)
    if cfg.max_train_examples > 0:
        dataset = Subset(dataset, list(range(min(cfg.max_train_examples, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def train_cifar_unet(cfg: Config, device: torch.device, out_dir: Path) -> None:
    temps = parse_float_tuple(cfg.drifting_temps)
    loader = make_cifar_unet_loader(cfg)
    batches = cycle(loader)

    model = DriftingUNetGenerator(dim=cfg.unet_dim, dim_mults=parse_int_tuple(cfg.unet_dim_mults)).to(device)
    ema_model = build_ema_model(model).to(device)
    feature_encoder = FrozenFeatureEncoder(
        stage_ids=parse_int_tuple(cfg.feature_stages),
        backbone_name=cfg.feature_backbone,
    ).to(device).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    real_preview, _ = next(iter(loader))
    save_tanh_grid(real_preview[: cfg.grid_size], out_dir / "real_grid.png", nrow=int(math.sqrt(cfg.grid_size)))

    metrics_path = out_dir / "metrics.jsonl"
    loss_ema = None
    pbar = tqdm(range(1, cfg.steps + 1), desc="cifar_unet")
    for step in pbar:
        real, _ = next(batches)
        real = real.to(device, non_blocking=True)
        gen = model(torch.randn_like(real))
        loss, info = feature_space_drifting_loss(
            gen_images=gen,
            pos_images=real,
            encoder=feature_encoder,
            temps=temps,
            use_std=not cfg.no_std_features,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(ema_model, model, decay=effective_ema_decay(cfg.ema_decay, step))

        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.96 * loss_ema + 0.04 * loss_val
        pbar.set_postfix(loss=f"{loss_ema:.3e}", scale=f"{info['scale']:.2e}")
        metric = {
            "step": step,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "scale": info["scale"],
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)
        if do_sample:
            sample_model = ema_model if cfg.preview_source == "ema" else model
            sample_model.eval()
            with torch.no_grad():
                vis = sample_model(torch.randn(cfg.grid_size, 3, 32, 32, device=device))
            save_tanh_grid(vis.cpu(), out_dir / f"samples_step{step:06d}.png", nrow=int(math.sqrt(cfg.grid_size)))

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "config": asdict(cfg),
                    "temps": temps,
                },
                out_dir / "cifar_unet.pt",
            )
            print(f"step {step}: loss_ema={loss_ema:.4e}; saved {out_dir / 'cifar_unet.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    sample_model = ema_model if cfg.preview_source == "ema" else model
    sample_model.eval()
    with torch.no_grad():
        final = sample_model(torch.randn(cfg.grid_size, 3, 32, 32, device=device))
    save_tanh_grid(final.cpu(), out_dir / "samples_final.png", nrow=int(math.sqrt(cfg.grid_size)))


# %%
class ParticleRefinerUNet(nn.Module):
    """UNet renderer that turns local image-particle barycenters into CIFAR samples."""

    def __init__(self, dim: int, dim_mults: tuple[int, ...], residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        try:
            from rectified_flow_pytorch import Unet
        except ImportError as exc:
            raise RuntimeError(
                "The particle-refiner branch needs rectified-flow-pytorch. "
                "Install it with: uv pip install --python .venv/bin/python rectified-flow-pytorch"
            ) from exc
        self.unet = Unet(dim=dim, channels=3, dim_mults=dim_mults, accept_time=False)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.unet(coarse))
        return (coarse + self.residual_scale * residual).clamp(-1.0, 1.0)


def particle_features(images: torch.Tensor, size: int) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(images, output_size=(size, size))
    return pooled.flatten(1)


@torch.no_grad()
def collect_particle_bank(loader: DataLoader, cfg: Config, device: torch.device) -> torch.Tensor:
    particles = []
    needed = cfg.mode_count
    for x, _ in loader:
        particles.append(x.to(device, non_blocking=True))
        if sum(t.shape[0] for t in particles) >= needed:
            break
    bank = torch.cat(particles, dim=0)[:needed]
    if bank.shape[0] < needed:
        repeats = math.ceil(needed / bank.shape[0])
        bank = bank.repeat((repeats, 1, 1, 1))[:needed]
    return bank.contiguous()


@torch.no_grad()
def local_particle_barycenter(
    real: torch.Tensor,
    particle_bank: torch.Tensor,
    particle_bank_features: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_features = particle_features(real, cfg.particle_feature_size)
    dists = torch.cdist(real_features.float(), particle_bank_features.float())
    top_dist, top_idx = dists.topk(k=min(cfg.particle_topk, particle_bank.shape[0]), largest=False, dim=1)
    weights = (-top_dist / cfg.particle_temp).softmax(dim=1)
    neighbors = particle_bank[top_idx]
    coarse = (weights[:, :, None, None, None] * neighbors).sum(dim=1)
    return coarse, top_idx[:, 0]


def particle_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_encoder: FrozenFeatureEncoder | None,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    loss = cfg.particle_l1_weight * l1 + cfg.particle_mse_weight * mse
    perceptual = torch.zeros((), device=pred.device)
    if feature_encoder is not None and cfg.particle_perceptual_weight > 0:
        pred_features = feature_encoder(pred)
        with torch.no_grad():
            target_features = feature_encoder(target)
        perceptual = sum(
            F.l1_loss(pred_features[key], target_features[key])
            for key in pred_features
        )
        loss = loss + cfg.particle_perceptual_weight * perceptual
    return loss, {
        "l1": float(l1.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "perceptual": float(perceptual.detach().cpu()),
    }


@torch.inference_mode()
def save_particle_refiner_samples(
    model: nn.Module,
    particle_bank: torch.Tensor,
    cfg: Config,
    path: Path,
) -> None:
    model.eval()
    n = cfg.grid_size
    indices = torch.randint(particle_bank.shape[0], (n,), device=particle_bank.device)
    coarse = particle_bank[indices]
    if cfg.particle_noise > 0:
        coarse = (coarse + cfg.particle_noise * torch.randn_like(coarse)).clamp(-1.0, 1.0)
    samples = model(coarse)
    save_tanh_grid(samples.cpu(), path, nrow=int(math.sqrt(n)))


def train_cifar_particle_refiner(cfg: Config, device: torch.device, out_dir: Path) -> None:
    loader = make_cifar_unet_loader(cfg)
    batches = cycle(loader)

    particle_bank = collect_particle_bank(loader, cfg, device)
    particle_bank_features = particle_features(particle_bank, cfg.particle_feature_size)
    grid_nrow = int(math.sqrt(cfg.grid_size))
    save_tanh_grid(particle_bank[: cfg.grid_size].cpu(), out_dir / "particle_grid.png", nrow=grid_nrow)

    real_preview, _ = next(iter(loader))
    real_preview = real_preview.to(device)
    coarse_preview, _ = local_particle_barycenter(
        real_preview[: cfg.grid_size],
        particle_bank,
        particle_bank_features,
        cfg,
    )
    save_tanh_grid(real_preview[: cfg.grid_size].cpu(), out_dir / "real_grid.png", nrow=grid_nrow)
    save_tanh_grid(coarse_preview.cpu(), out_dir / "coarse_grid.png", nrow=grid_nrow)

    model = ParticleRefinerUNet(
        dim=cfg.unet_dim,
        dim_mults=parse_int_tuple(cfg.unet_dim_mults),
        residual_scale=cfg.particle_residual_scale,
    ).to(device)
    ema_model = build_ema_model(model).to(device)
    feature_encoder = None
    if cfg.particle_perceptual_weight > 0:
        feature_encoder = FrozenFeatureEncoder(
            stage_ids=parse_int_tuple(cfg.feature_stages),
            backbone_name=cfg.feature_backbone,
        ).to(device).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    metrics_path = out_dir / "metrics.jsonl"

    loss_ema = None
    pbar = tqdm(range(1, cfg.steps + 1), desc="particle_refiner")
    for step in pbar:
        real, _ = next(batches)
        real = real.to(device, non_blocking=True)
        with torch.no_grad():
            coarse, nearest = local_particle_barycenter(real, particle_bank, particle_bank_features, cfg)
            noisy_coarse = coarse
            if cfg.particle_noise > 0:
                noisy_coarse = (coarse + cfg.particle_noise * torch.randn_like(coarse)).clamp(-1.0, 1.0)

        pred = model(noisy_coarse)
        loss, parts = particle_reconstruction_loss(pred, real, feature_encoder, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(ema_model, model, decay=effective_ema_decay(cfg.ema_decay, step))

        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.96 * loss_ema + 0.04 * loss_val
        pbar.set_postfix(loss=f"{loss_ema:.3f}", l1=f"{parts['l1']:.3f}")
        metric = {
            "step": step,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "l1": parts["l1"],
            "mse": parts["mse"],
            "perceptual": parts["perceptual"],
            "nearest_unique": int(nearest.unique().numel()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)

        if do_sample:
            sample_model = ema_model if cfg.preview_source == "ema" else model
            save_particle_refiner_samples(sample_model, particle_bank, cfg, out_dir / f"samples_step{step:06d}.png")
            sample_model.eval()
            with torch.no_grad():
                recon_preview = sample_model(coarse_preview)
            save_tanh_grid(recon_preview.cpu(), out_dir / f"recon_step{step:06d}.png", nrow=grid_nrow)

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "particle_bank": particle_bank.cpu(),
                    "particle_bank_features": particle_bank_features.cpu(),
                    "config": asdict(cfg),
                },
                out_dir / "cifar_particle_refiner.pt",
            )
            print(f"step {step}: loss_ema={loss_ema:.4f}; saved {out_dir / 'cifar_particle_refiner.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    sample_model = ema_model if cfg.preview_source == "ema" else model
    save_particle_refiner_samples(sample_model, particle_bank, cfg, out_dir / "samples_final.png")


# %%
class DriftingImageParticles(nn.Module):
    """A bank of generated image particles optimized by the drifting field."""

    def __init__(self, particle_count: int, init_images: torch.Tensor | None = None):
        super().__init__()
        if init_images is None:
            raw = torch.randn(particle_count, 3, 32, 32) * 0.6
        else:
            init = init_images.clamp(-0.999, 0.999)
            raw = torch.atanh(init)
        self.raw_particles = nn.Parameter(raw)

    def images(self) -> torch.Tensor:
        return torch.tanh(self.raw_particles)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.raw_particles.device
        if n <= self.raw_particles.shape[0]:
            indices = torch.randperm(self.raw_particles.shape[0], device=device)[:n]
        else:
            indices = torch.randint(self.raw_particles.shape[0], (n,), device=device)
        return torch.tanh(self.raw_particles[indices]), indices


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    dx = images[..., :, 1:] - images[..., :, :-1]
    dy = images[..., 1:, :] - images[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


@torch.no_grad()
def collect_drift_particle_init(loader: DataLoader, cfg: Config, device: torch.device) -> torch.Tensor | None:
    if cfg.drift_particle_init == "noise":
        return None
    images = []
    for x, _ in loader:
        images.append(x.to(device, non_blocking=True))
        if sum(t.shape[0] for t in images) >= cfg.mode_count:
            break
    init = torch.cat(images, dim=0)[: cfg.mode_count]
    if init.shape[0] < cfg.mode_count:
        repeats = math.ceil(cfg.mode_count / init.shape[0])
        init = init.repeat((repeats, 1, 1, 1))[: cfg.mode_count]
    return init


@torch.inference_mode()
def save_drift_particle_grid(model: DriftingImageParticles, path: Path, n: int) -> None:
    images, _ = model.sample(n)
    save_tanh_grid(images.cpu(), path, nrow=int(math.sqrt(n)))


def train_cifar_drift_particles(cfg: Config, device: torch.device, out_dir: Path) -> None:
    temps = parse_float_tuple(cfg.drifting_temps)
    loader = make_cifar_unet_loader(cfg)
    batches = cycle(loader)
    init_images = collect_drift_particle_init(loader, cfg, device)
    model = DriftingImageParticles(cfg.mode_count, init_images=init_images).to(device)
    feature_encoder = FrozenFeatureEncoder(
        stage_ids=parse_int_tuple(cfg.feature_stages),
        backbone_name=cfg.feature_backbone,
    ).to(device).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    real_preview, _ = next(iter(loader))
    save_tanh_grid(real_preview[: cfg.grid_size], out_dir / "real_grid.png", nrow=int(math.sqrt(cfg.grid_size)))
    save_drift_particle_grid(model, out_dir / "samples_step000000.png", cfg.grid_size)

    metrics_path = out_dir / "metrics.jsonl"
    loss_ema = None
    pbar = tqdm(range(1, cfg.steps + 1), desc="drift_particles")
    for step in pbar:
        real, _ = next(batches)
        real = real.to(device, non_blocking=True)
        gen, indices = model.sample(real.shape[0])
        drift_loss, info = feature_space_drifting_loss(
            gen_images=gen,
            pos_images=real,
            encoder=feature_encoder,
            temps=temps,
            use_std=not cfg.no_std_features,
        )
        pixel_l2 = gen.pow(2).mean()
        tv = total_variation_loss(gen)
        loss = drift_loss + cfg.drift_pixel_l2_weight * pixel_l2 + cfg.drift_tv_weight * tv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.96 * loss_ema + 0.04 * loss_val
        pbar.set_postfix(loss=f"{loss_ema:.3e}", scale=f"{info['scale']:.2e}")
        metric = {
            "step": step,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "drift_loss": float(drift_loss.detach().cpu()),
            "pixel_l2": float(pixel_l2.detach().cpu()),
            "tv": float(tv.detach().cpu()),
            "scale": info["scale"],
            "particle_unique": int(indices.unique().numel()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)
        if do_sample:
            save_drift_particle_grid(model, out_dir / f"samples_step{step:06d}.png", cfg.grid_size)

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "particles": model.images().detach().cpu(),
                    "config": asdict(cfg),
                    "temps": temps,
                },
                out_dir / "cifar_drift_particles.pt",
            )
            print(f"step {step}: loss_ema={loss_ema:.4e}; saved {out_dir / 'cifar_drift_particles.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    save_drift_particle_grid(model, out_dir / "samples_final.png", cfg.grid_size)


# %%
class RectifiedFlowUNet(nn.Module):
    """CIFAR rectified-flow velocity model."""

    def __init__(self, dim: int, dim_mults: tuple[int, ...]):
        super().__init__()
        try:
            from rectified_flow_pytorch import Unet
        except ImportError as exc:
            raise RuntimeError(
                "The rectified-flow branch needs rectified-flow-pytorch. "
                "Install it with: uv pip install --python .venv/bin/python rectified-flow-pytorch"
            ) from exc
        self.unet = Unet(dim=dim, channels=3, dim_mults=dim_mults, accept_time=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t)


class ClassConditionalRectifiedFlowUNet(nn.Module):
    """CIFAR class-mixture rectified-flow velocity model."""

    def __init__(self, dim: int, dim_mults: tuple[int, ...], cond_dim: int, num_classes: int = 10):
        super().__init__()
        try:
            from rectified_flow_pytorch import Unet
        except ImportError as exc:
            raise RuntimeError(
                "The class-flow branch needs rectified-flow-pytorch. "
                "Install it with: uv pip install --python .venv/bin/python rectified-flow-pytorch"
            ) from exc
        self.label_embed = nn.Embedding(num_classes, cond_dim)
        self.unet = Unet(
            dim=dim,
            channels=3,
            dim_mults=dim_mults,
            accept_time=True,
            accept_cond=True,
            dim_cond=cond_dim,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t, cond=self.label_embed(labels))


class ComponentConditionalRectifiedFlowUNet(nn.Module):
    """Velocity model conditioned on an explicit learned mixture component."""

    def __init__(
        self,
        dim: int,
        dim_mults: tuple[int, ...],
        cond_dim: int,
        component_count: int,
        accept_dest_time: bool = False,
    ):
        super().__init__()
        try:
            from rectified_flow_pytorch import Unet
        except ImportError as exc:
            raise RuntimeError(
                "The likelihood-flow branch needs rectified-flow-pytorch. "
                "Install it with: uv pip install --python .venv/bin/python rectified-flow-pytorch"
            ) from exc
        self.component_count = component_count
        self.null_index = component_count
        self.component_embed = nn.Embedding(component_count + 1, cond_dim)
        self.unet = Unet(
            dim=dim,
            channels=3,
            dim_mults=dim_mults,
            accept_time=True,
            accept_dest_time=accept_dest_time,
            accept_cond=True,
            dim_cond=cond_dim,
        )

    def hard_condition(self, components: torch.Tensor) -> torch.Tensor:
        return self.component_embed(components)

    def soft_condition(self, weights: torch.Tensor) -> torch.Tensor:
        return weights @ self.component_embed.weight[: self.component_count]

    def null_condition(self, n: int, device: torch.device) -> torch.Tensor:
        return self.component_embed(self.null_components(n, device))

    def forward_condition(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.unet(x, t, s=s, cond=cond)

    def forward(self, x: torch.Tensor, t: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        return self.forward_condition(x, t, self.hard_condition(components))

    def null_components(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.full((n,), self.null_index, device=device, dtype=torch.long)


class ComponentConditionedFlowObjectiveAdapter(nn.Module):
    """Adapts our component-conditioned UNet to lucidrains flow objectives."""

    def __init__(self, model: ComponentConditionalRectifiedFlowUNet):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if times is None:
            raise ValueError("Flow objective adapter requires a time tensor.")
        if cond is None:
            raise ValueError("Flow objective adapter requires a condition tensor.")
        return self.model.forward_condition(x, times, cond, s=s)


def build_likelihood_flow_objective(
    model: ComponentConditionalRectifiedFlowUNet,
    cfg: Config,
) -> nn.Module | None:
    if cfg.flow_objective == "manual":
        return None

    adapter = ComponentConditionedFlowObjectiveAdapter(model)
    if cfg.flow_objective == "nano":
        from rectified_flow_pytorch import NanoFlow

        return NanoFlow(adapter, times_cond_kwarg="times", data_shape=(3, 32, 32))
    if cfg.flow_objective == "mean":
        from rectified_flow_pytorch import MeanFlow

        return MeanFlow(adapter, data_shape=(3, 32, 32), accept_cond=True)
    if cfg.flow_objective == "split_mean":
        from rectified_flow_pytorch import SplitMeanFlow

        return SplitMeanFlow(adapter, data_shape=(3, 32, 32), accept_cond=True)
    if cfg.flow_objective == "lsd":
        from rectified_flow_pytorch import LsdFlow

        return LsdFlow(adapter, data_shape=(3, 32, 32), accept_cond=True)
    raise ValueError(f"unknown flow objective: {cfg.flow_objective}")


class LearnedImageMixturePrior(nn.Module):
    """Uniform mixture of learned synthetic image-space source anchors."""

    def __init__(self, component_count: int, std: float):
        super().__init__()
        self.std = std
        self.raw_means = nn.Parameter(torch.randn(component_count, 3, 32, 32) * 0.25)

    @property
    def component_count(self) -> int:
        return self.raw_means.shape[0]

    def means(self) -> torch.Tensor:
        return torch.tanh(self.raw_means)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.raw_means.device
        indices = torch.randint(self.component_count, (n,), device=device)
        means = self.means()[indices]
        if self.std <= 0:
            return means, indices
        return (means + self.std * torch.randn_like(means)).clamp(-3.0, 3.0), indices

    def repulsion_loss(self, feature_size: int = 4, sample_count: int = 256) -> torch.Tensor:
        means = self.means()
        if means.shape[0] > sample_count:
            indices = torch.randperm(means.shape[0], device=means.device)[:sample_count]
            means = means[indices]
        feat = particle_features(means, feature_size)
        dist = torch.cdist(feat, feat).pow(2)
        eye = torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(eye, float("inf"))
        return torch.exp(-dist / 4.0).mean()


class FeatureGaussianMixture(nn.Module):
    """A trainable finite Gaussian mixture in standardized image-feature space."""

    def __init__(self, initial_means: torch.Tensor, sigma: float):
        super().__init__()
        if sigma <= 0:
            raise ValueError("FeatureGaussianMixture sigma must be positive")
        self.means = nn.Parameter(initial_means.detach().clone())
        self.logits = nn.Parameter(torch.zeros(initial_means.shape[0]))
        self.sigma = float(sigma)

    @property
    def component_count(self) -> int:
        return self.means.shape[0]

    def component_logits(self, features: torch.Tensor) -> torch.Tensor:
        sq_dist = torch.cdist(features.float(), self.means.float()).pow(2)
        return self.logits.log_softmax(dim=0)[None, :] - 0.5 * sq_dist / (self.sigma * self.sigma)

    def nll_and_responsibilities(self, features: torch.Tensor, resp_temp: float) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.component_logits(features)
        feat_dim = features.shape[-1]
        log_norm = feat_dim * math.log(self.sigma) + 0.5 * feat_dim * math.log(2.0 * math.pi)
        nll = -(torch.logsumexp(logits, dim=1) - log_norm).mean()
        resp = (logits / max(resp_temp, 1e-6)).softmax(dim=1)
        return nll, resp

    def sample_components(self, n: int, device: torch.device) -> torch.Tensor:
        probs = self.logits.softmax(dim=0)
        return torch.multinomial(probs, n, replacement=True).to(device)

    def usage_entropy(self) -> torch.Tensor:
        probs = self.logits.softmax(dim=0)
        return -(probs * probs.clamp_min(1e-8).log()).sum()


@torch.inference_mode()
def sample_rectified_flow(model: RectifiedFlowUNet, n: int, steps: int, device: torch.device) -> torch.Tensor:
    model.eval()
    x = torch.randn(n, 3, 32, 32, device=device)
    dt = 1.0 / float(steps)
    for i in range(steps):
        t = torch.full((n,), i / float(steps), device=device)
        x = x + dt * model(x, t)
    return x.clamp(-1.0, 1.0)


@torch.inference_mode()
def sample_class_rectified_flow(
    model: ClassConditionalRectifiedFlowUNet,
    n: int,
    steps: int,
    device: torch.device,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(n, 3, 32, 32, device=device)
    if labels is None:
        labels = torch.arange(n, device=device) % 10
        labels = labels[torch.randperm(n, device=device)]
    dt = 1.0 / float(steps)
    for i in range(steps):
        t = torch.full((n,), i / float(steps), device=device)
        x = x + dt * model(x, t, labels)
    return x.clamp(-1.0, 1.0)


@torch.inference_mode()
def sample_likelihood_rectified_flow(
    model: ComponentConditionalRectifiedFlowUNet,
    mixture: FeatureGaussianMixture,
    n: int,
    steps: int,
    device: torch.device,
    guidance_scale: float = 1.0,
    conditioning: str = "hard",
    resp_temp: float = 0.7,
    flow_objective: str = "manual",
) -> torch.Tensor:
    model.eval()
    mixture.eval()
    x = torch.randn(n, 3, 32, 32, device=device)
    components = mixture.sample_components(n, device)
    if conditioning == "soft":
        feature_samples = mixture.means[components] + mixture.sigma * torch.randn(
            n,
            mixture.means.shape[1],
            device=device,
        )
        _, resp = mixture.nll_and_responsibilities(feature_samples, resp_temp)
        cond = model.soft_condition(resp)
    else:
        cond = model.hard_condition(components)
    null_cond = model.null_condition(n, device)

    def predict_with_guidance(
        x_in: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = model.forward_condition(x_in, t, cond, s=s)
        if guidance_scale != 1.0:
            pred_null = model.forward_condition(x_in, t, null_cond, s=s)
            pred = pred_null + guidance_scale * (pred - pred_null)
        return pred

    dt = 1.0 / float(steps)
    if flow_objective in {"manual", "nano"}:
        for i in range(steps):
            t = torch.full((n,), i / float(steps), device=device)
            x = x + dt * predict_with_guidance(x, t)
    elif flow_objective == "mean":
        if steps == 1:
            t = torch.ones(n, device=device)
            delta_time = torch.ones(n, device=device)
            x = x - predict_with_guidance(x, t, s=delta_time)
        else:
            delta_time = torch.zeros(n, device=device)
            for time in torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]:
                t = time.expand(n)
                x = x - dt * predict_with_guidance(x, t, s=delta_time)
    elif flow_objective == "split_mean":
        if steps == 1:
            t = torch.ones(n, device=device)
            x = x - predict_with_guidance(x, t, s=t)
        else:
            for time in torch.linspace(0.0, 1.0, steps + 1, device=device)[:-1]:
                t = time.expand(n)
                x = x + dt * predict_with_guidance(x, t, s=t)
    elif flow_objective == "lsd":
        for time in torch.linspace(0.0, 1.0, steps + 1, device=device)[:-1]:
            start = time.expand(n)
            end = (time + dt).expand(n)
            x = x + dt * predict_with_guidance(x, start, s=end)
    else:
        raise ValueError(f"unknown flow objective: {flow_objective}")
    return x.clamp(-1.0, 1.0)


@torch.inference_mode()
def sample_mixture_rectified_flow(
    model: RectifiedFlowUNet,
    prior: LearnedImageMixturePrior,
    n: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    prior.eval()
    x, _ = prior.sample(n)
    x = x.to(device)
    dt = 1.0 / float(steps)
    for i in range(steps):
        t = torch.full((n,), i / float(steps), device=device)
        x = x + dt * model(x, t)
    return x.clamp(-1.0, 1.0)


def evaluate_image_samples_against_cifar(
    samples: torch.Tensor,
    cfg: Config,
    device: torch.device,
    out_dir: Path,
    name: str,
) -> dict:
    from torchmetrics.image.fid import FrechetInceptionDistance

    samples = samples.to(device).clamp(-1.0, 1.0)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    test_loader = make_cifar_eval_loader(cfg, train=False, batch_size=cfg.eval_batch_size)
    seen = 0
    test_for_nn = []
    for x, _ in test_loader:
        x = x.to(device, non_blocking=True)
        take = min(x.shape[0], samples.shape[0] - seen)
        if take <= 0:
            break
        x = x[:take]
        fid.update(x, real=True)
        test_for_nn.append(x.mul(2.0).sub(1.0).cpu())
        seen += take

    for start in range(0, samples.shape[0], cfg.eval_batch_size):
        fid.update((samples[start : start + cfg.eval_batch_size] + 1.0) * 0.5, real=False)

    train_loader = make_cifar_eval_loader(cfg, train=True, batch_size=512)
    train_feats = []
    for x, _ in train_loader:
        train_feats.append(particle_features(x.to(device, non_blocking=True).mul(2.0).sub(1.0), 8).cpu())
    train_feats = torch.cat(train_feats, dim=0).to(device)

    def min_dists(query: torch.Tensor, bank: torch.Tensor, block: int = 8192) -> torch.Tensor:
        outs = []
        for start in range(0, query.shape[0], 256):
            chunk = query[start : start + 256]
            best = torch.full((chunk.shape[0],), float("inf"), device=device)
            for bank_start in range(0, bank.shape[0], block):
                dist = torch.cdist(chunk, bank[bank_start : bank_start + block]).min(dim=1).values
                best = torch.minimum(best, dist)
            outs.append(best)
        return torch.cat(outs, dim=0)

    gen_dist = min_dists(particle_features(samples, 8), train_feats)
    test_images = torch.cat(test_for_nn, dim=0)[: samples.shape[0]].to(device)
    test_dist = min_dists(particle_features(test_images, 8), train_feats)
    metrics = {
        "name": name,
        "samples": int(samples.shape[0]),
        "fid_cifar10_test": float(fid.compute().detach().cpu()),
        "gen_to_train_nn_mean": float(gen_dist.mean().detach().cpu()),
        "gen_to_train_nn_median": float(gen_dist.median().detach().cpu()),
        "test_to_train_nn_mean": float(test_dist.mean().detach().cpu()),
        "test_to_train_nn_median": float(test_dist.median().detach().cpu()),
        "gen_exact_train_rate": float((gen_dist < 1e-7).float().mean().detach().cpu()),
    }
    (out_dir / f"{name}_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@torch.inference_mode()
def compact_cifar_descriptors(images: torch.Tensor, encoder: FrozenFeatureEncoder) -> torch.Tensor:
    feats = encoder(images)
    parts = []
    for stage_id in sorted(feats):
        feat = feats[stage_id]
        parts.append(feat.mean(dim=(2, 3)))
        parts.append(feat.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6))
    parts.append(F.adaptive_avg_pool2d(images, output_size=(4, 4)).flatten(1))
    parts.append(images.mean(dim=(2, 3)))
    parts.append(images.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6))
    return torch.cat(parts, dim=1)


@torch.inference_mode()
def kmeans_feature_init(
    features: torch.Tensor,
    component_count: int,
    device: torch.device,
    iterations: int = 12,
) -> torch.Tensor:
    if component_count <= 0:
        raise ValueError("component_count must be positive")
    work = features.to(device)
    if component_count >= work.shape[0]:
        repeats = math.ceil(component_count / work.shape[0])
        return work.repeat((repeats, 1))[:component_count].contiguous()

    perm = torch.randperm(work.shape[0], device=device)
    means = work[perm[:component_count]].clone()
    for _ in tqdm(range(iterations), desc="feature kmeans"):
        labels = torch.cdist(work.float(), means.float()).argmin(dim=1)
        new_means = torch.zeros_like(means)
        counts = torch.bincount(labels, minlength=component_count).to(device=device, dtype=means.dtype).clamp_min(1.0)
        new_means.index_add_(0, labels, work)
        new_means = new_means / counts[:, None]
        empty = counts <= 1.0
        if empty.any():
            refill = torch.randperm(work.shape[0], device=device)[: int(empty.sum())]
            new_means[empty] = work[refill]
        means = new_means
    return means.contiguous()


def make_cifar_likelihood_dataset(
    cfg: Config,
    device: torch.device,
    out_dir: Path,
) -> tuple[TensorDataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset: Dataset = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=transform)
    if cfg.max_train_examples > 0:
        dataset = Subset(dataset, list(range(min(cfg.max_train_examples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    encoder = FrozenFeatureEncoder(stage_ids=(3, 4), backbone_name=cfg.feature_backbone).to(device).eval()
    images, labels, raw_features = [], [], []
    for x, y in tqdm(loader, desc="precompute mixture features"):
        x = x.to(device, non_blocking=True)
        raw_features.append(compact_cifar_descriptors(x, encoder).cpu())
        images.append(x.cpu())
        labels.append(y.cpu())

    image_tensor = torch.cat(images, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    raw_feature_tensor = torch.cat(raw_features, dim=0)
    feature_mean = raw_feature_tensor.mean(dim=0)
    feature_std = raw_feature_tensor.std(dim=0).clamp_min(1e-4)
    feature_tensor = (raw_feature_tensor - feature_mean) / feature_std
    initial_means = kmeans_feature_init(feature_tensor, cfg.mode_count, device).cpu()

    save_tanh_grid(image_tensor[: cfg.grid_size], out_dir / "real_grid.png", nrow=int(math.sqrt(cfg.grid_size)))
    torch.save(
        {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "initial_means": initial_means,
        },
        out_dir / "mixture_feature_stats.pt",
    )
    return TensorDataset(image_tensor, label_tensor, feature_tensor), feature_mean, feature_std, initial_means


def train_cifar_rectified_flow(cfg: Config, device: torch.device, out_dir: Path) -> None:
    loader = make_cifar_unet_loader(cfg)
    batches = cycle(loader)
    model = RectifiedFlowUNet(dim=cfg.unet_dim, dim_mults=parse_int_tuple(cfg.unet_dim_mults)).to(device)
    ema_model = build_ema_model(model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    metrics_path = out_dir / "metrics.jsonl"

    real_preview, _ = next(iter(loader))
    save_tanh_grid(real_preview[: cfg.grid_size], out_dir / "real_grid.png", nrow=int(math.sqrt(cfg.grid_size)))

    loss_ema = None
    pbar = tqdm(range(1, cfg.steps + 1), desc="rectified_flow")
    for step in pbar:
        x1, _ = next(batches)
        x1 = x1.to(device, non_blocking=True)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=device)
        view_t = t.view(-1, 1, 1, 1)
        xt = (1.0 - view_t) * x0 + view_t * x1
        target = x1 - x0
        pred = model(xt, t)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(ema_model, model, decay=effective_ema_decay(cfg.ema_decay, step))

        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        pbar.set_postfix(loss=f"{loss_ema:.4f}")
        metric = {
            "step": step,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)
        if do_sample:
            sample_model = ema_model if cfg.preview_source == "ema" else model
            samples = sample_rectified_flow(sample_model, cfg.grid_size, cfg.flow_sample_steps, device)
            save_tanh_grid(samples.cpu(), out_dir / f"samples_step{step:06d}.png", nrow=int(math.sqrt(cfg.grid_size)))

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "config": asdict(cfg),
                },
                out_dir / "cifar_rectified_flow.pt",
            )
            print(f"step {step}: loss_ema={loss_ema:.4f}; saved {out_dir / 'cifar_rectified_flow.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    sample_model = ema_model if cfg.preview_source == "ema" else model
    final = sample_rectified_flow(sample_model, cfg.grid_size, cfg.flow_sample_steps, device)
    save_tanh_grid(final.cpu(), out_dir / "samples_final.png", nrow=int(math.sqrt(cfg.grid_size)))
    if cfg.flow_eval_samples > 0:
        eval_samples = sample_rectified_flow(sample_model, cfg.flow_eval_samples, cfg.flow_sample_steps, device)
        metrics = evaluate_image_samples_against_cifar(eval_samples, cfg, device, out_dir, "flow")
        print(json.dumps(metrics, indent=2))


def train_cifar_mixture_flow(cfg: Config, device: torch.device, out_dir: Path) -> None:
    loader = make_cifar_unet_loader(cfg)
    batches = cycle(loader)
    model = RectifiedFlowUNet(dim=cfg.unet_dim, dim_mults=parse_int_tuple(cfg.unet_dim_mults)).to(device)
    prior = LearnedImageMixturePrior(component_count=cfg.mode_count, std=cfg.prior_std).to(device)
    ema_model = build_ema_model(model).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr},
            {"params": prior.parameters(), "lr": cfg.lr * cfg.prior_lr_scale},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    metrics_path = out_dir / "metrics.jsonl"

    real_preview, _ = next(iter(loader))
    save_tanh_grid(real_preview[: cfg.grid_size], out_dir / "real_grid.png", nrow=int(math.sqrt(cfg.grid_size)))
    save_tanh_grid(prior.means()[: cfg.grid_size].detach().cpu(), out_dir / "prior_means_step000000.png")

    loss_ema = None
    pbar = tqdm(range(1, cfg.steps + 1), desc="mixture_flow")
    for step in pbar:
        x1, _ = next(batches)
        x1 = x1.to(device, non_blocking=True)
        x0, indices = prior.sample(x1.shape[0])
        t = torch.rand(x1.shape[0], device=device)
        view_t = t.view(-1, 1, 1, 1)
        xt = (1.0 - view_t) * x0 + view_t * x1
        target = x1 - x0
        pred = model(xt, t)
        fm_loss = F.mse_loss(pred, target)
        prior_tv = total_variation_loss(prior.means())
        repulsion = prior.repulsion_loss()
        loss = fm_loss + cfg.prior_tv_weight * prior_tv + cfg.prior_repulsion_weight * repulsion

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *prior.parameters()], cfg.grad_clip)
        opt.step()
        ema_update(ema_model, model, decay=effective_ema_decay(cfg.ema_decay, step))

        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        pbar.set_postfix(loss=f"{loss_ema:.4f}", prior_std=f"{prior.means().std().item():.3f}")
        metric = {
            "step": step,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "fm_loss": float(fm_loss.detach().cpu()),
            "prior_tv": float(prior_tv.detach().cpu()),
            "repulsion": float(repulsion.detach().cpu()),
            "prior_mean_std": float(prior.means().std().detach().cpu()),
            "component_unique": int(indices.unique().numel()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)
        if do_sample:
            sample_model = ema_model if cfg.preview_source == "ema" else model
            samples = sample_mixture_rectified_flow(sample_model, prior, cfg.grid_size, cfg.flow_sample_steps, device)
            save_tanh_grid(samples.cpu(), out_dir / f"samples_step{step:06d}.png", nrow=int(math.sqrt(cfg.grid_size)))
            save_tanh_grid(
                prior.means()[: cfg.grid_size].detach().cpu(),
                out_dir / f"prior_means_step{step:06d}.png",
                nrow=int(math.sqrt(cfg.grid_size)),
            )

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "prior": prior.state_dict(),
                    "prior_means": prior.means().detach().cpu(),
                    "config": asdict(cfg),
                },
                out_dir / "cifar_mixture_flow.pt",
            )
            print(f"step {step}: loss_ema={loss_ema:.4f}; saved {out_dir / 'cifar_mixture_flow.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    sample_model = ema_model if cfg.preview_source == "ema" else model
    final = sample_mixture_rectified_flow(sample_model, prior, cfg.grid_size, cfg.flow_sample_steps, device)
    save_tanh_grid(final.cpu(), out_dir / "samples_final.png", nrow=int(math.sqrt(cfg.grid_size)))
    if cfg.flow_eval_samples > 0:
        eval_samples = sample_mixture_rectified_flow(
            sample_model,
            prior,
            cfg.flow_eval_samples,
            cfg.flow_sample_steps,
            device,
        )
        metrics = evaluate_image_samples_against_cifar(eval_samples, cfg, device, out_dir, "mixture_flow")
        print(json.dumps(metrics, indent=2))


def train_cifar_class_flow(cfg: Config, device: torch.device, out_dir: Path) -> None:
    loader = make_cifar_unet_loader(cfg)
    batches = cycle(loader)
    model = ClassConditionalRectifiedFlowUNet(
        dim=cfg.unet_dim,
        dim_mults=parse_int_tuple(cfg.unet_dim_mults),
        cond_dim=cfg.cond_dim,
    ).to(device)
    ema_model = build_ema_model(model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    metrics_path = out_dir / "metrics.jsonl"

    real_preview, _ = next(iter(loader))
    save_tanh_grid(real_preview[: cfg.grid_size], out_dir / "real_grid.png", nrow=int(math.sqrt(cfg.grid_size)))

    loss_ema = None
    pbar = tqdm(range(1, cfg.steps + 1), desc="class_flow")
    for step in pbar:
        x1, y = next(batches)
        x1 = x1.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=device)
        view_t = t.view(-1, 1, 1, 1)
        xt = (1.0 - view_t) * x0 + view_t * x1
        target = x1 - x0
        pred = model(xt, t, y)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(ema_model, model, decay=effective_ema_decay(cfg.ema_decay, step))

        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        pbar.set_postfix(loss=f"{loss_ema:.4f}")
        metric = {
            "step": step,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)
        if do_sample:
            sample_model = ema_model if cfg.preview_source == "ema" else model
            samples = sample_class_rectified_flow(sample_model, cfg.grid_size, cfg.flow_sample_steps, device)
            save_tanh_grid(samples.cpu(), out_dir / f"samples_step{step:06d}.png", nrow=int(math.sqrt(cfg.grid_size)))

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "config": asdict(cfg),
                },
                out_dir / "cifar_class_flow.pt",
            )
            print(f"step {step}: loss_ema={loss_ema:.4f}; saved {out_dir / 'cifar_class_flow.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    sample_model = ema_model if cfg.preview_source == "ema" else model
    final = sample_class_rectified_flow(sample_model, cfg.grid_size, cfg.flow_sample_steps, device)
    save_tanh_grid(final.cpu(), out_dir / "samples_final.png", nrow=int(math.sqrt(cfg.grid_size)))
    if cfg.flow_eval_samples > 0:
        eval_samples = sample_class_rectified_flow(sample_model, cfg.flow_eval_samples, cfg.flow_sample_steps, device)
        metrics = evaluate_image_samples_against_cifar(eval_samples, cfg, device, out_dir, "class_flow")
        print(json.dumps(metrics, indent=2))


def train_cifar_likelihood_flow(cfg: Config, device: torch.device, out_dir: Path) -> None:
    dataset, feature_mean, feature_std, initial_means = make_cifar_likelihood_dataset(cfg, device, out_dir)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    batches = cycle(loader)

    model = ComponentConditionalRectifiedFlowUNet(
        dim=cfg.unet_dim,
        dim_mults=parse_int_tuple(cfg.unet_dim_mults),
        cond_dim=cfg.cond_dim,
        component_count=cfg.mode_count,
        accept_dest_time=cfg.flow_objective in {"mean", "split_mean", "lsd"},
    ).to(device)
    flow_objective = build_likelihood_flow_objective(model, cfg)
    if flow_objective is not None:
        flow_objective = flow_objective.to(device)
    mixture = FeatureGaussianMixture(initial_means.to(device), sigma=cfg.feature_sigma).to(device)
    ema_model = build_ema_model(model).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr},
            {"params": mixture.parameters(), "lr": cfg.lr * cfg.prior_lr_scale},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    metrics_path = out_dir / "metrics.jsonl"

    start_step = 1
    if cfg.resume_ckpt:
        ckpt_path = Path(cfg.resume_ckpt)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        else:
            ema_model.load_state_dict(ckpt["model"])
        mixture.load_state_dict(ckpt["mixture"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"resumed {ckpt_path} at step {start_step - 1}")

    loss_ema = None
    nll_ema = None
    pbar = tqdm(range(start_step, cfg.steps + 1), desc="likelihood_flow")
    for step in pbar:
        x1, _, feat = next(batches)
        x1 = x1.to(device, non_blocking=True)
        feat = feat.to(device, non_blocking=True)
        nll, resp = mixture.nll_and_responsibilities(feat, cfg.mixture_resp_temp)
        nll_per_dim = nll / feat.shape[1]
        mean_resp = resp.mean(dim=0)
        balance = (mean_resp * (mean_resp * cfg.mode_count).clamp_min(1e-8).log()).sum()
        components = torch.multinomial(resp.detach(), num_samples=1).squeeze(1)
        if cfg.likelihood_conditioning == "soft":
            cond = model.soft_condition(resp.detach())
            if cfg.cond_dropout > 0:
                drop = torch.rand_like(components.float()) < cfg.cond_dropout
                cond = torch.where(drop[:, None], model.null_condition(components.shape[0], device), cond)
        else:
            if cfg.cond_dropout > 0:
                drop = torch.rand_like(components.float()) < cfg.cond_dropout
                components = torch.where(drop, model.null_components(components.shape[0], device), components)
            cond = model.hard_condition(components)

        if flow_objective is None:
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            view_t = t.view(-1, 1, 1, 1)
            xt = (1.0 - view_t) * x0 + view_t * x1
            target = x1 - x0
            pred = model.forward_condition(xt, t, cond)
            fm_loss = F.mse_loss(pred, target)
        else:
            fm_loss = flow_objective(x1, cond=cond)
        loss = fm_loss + cfg.mixture_nll_weight * nll_per_dim + cfg.mixture_balance_weight * balance

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *mixture.parameters()], cfg.grad_clip)
        opt.step()
        ema_update(ema_model, model, decay=effective_ema_decay(cfg.ema_decay, step))

        loss_val = float(loss.detach().cpu())
        nll_val = float(nll_per_dim.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        nll_ema = nll_val if nll_ema is None else 0.98 * nll_ema + 0.02 * nll_val
        hard_unique = int(components[components != model.null_index].unique().numel())
        pbar.set_postfix(loss=f"{loss_ema:.4f}", nll=f"{nll_ema:.3f}", uniq=hard_unique)
        metric = {
            "step": step,
            "loss": loss_val,
            "loss_ema": float(loss_ema),
            "fm_loss": float(fm_loss.detach().cpu()),
            "mixture_nll_per_dim": nll_val,
            "balance": float(balance.detach().cpu()),
            "usage_entropy": float(mixture.usage_entropy().detach().cpu()),
            "hard_unique": hard_unique,
            "resp_entropy": float((-(resp * resp.clamp_min(1e-8).log()).sum(dim=1).mean()).detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

        do_log = step == 1 or (cfg.log_every > 0 and step % cfg.log_every == 0)
        do_sample = cfg.sample_every > 0 and (step == 1 or step % cfg.sample_every == 0 or step == cfg.steps)
        do_save = cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps)
        if do_sample:
            sample_model = ema_model if cfg.preview_source == "ema" else model
            samples = sample_likelihood_rectified_flow(
                sample_model,
                mixture,
                cfg.grid_size,
                cfg.flow_sample_steps,
                device,
                guidance_scale=cfg.flow_guidance_scale,
                conditioning=cfg.likelihood_conditioning,
                resp_temp=cfg.mixture_resp_temp,
                flow_objective=cfg.flow_objective,
            )
            save_tanh_grid(samples.cpu(), out_dir / f"samples_step{step:06d}.png", nrow=int(math.sqrt(cfg.grid_size)))

        if do_save:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "mixture": mixture.state_dict(),
                    "optimizer": opt.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "config": asdict(cfg),
                },
                out_dir / "cifar_likelihood_flow.pt",
            )
            print(f"step {step}: loss_ema={loss_ema:.4f}; saved {out_dir / 'cifar_likelihood_flow.pt'}")

        if do_log or do_save:
            append_metric(metrics_path, metric)

    sample_model = ema_model if cfg.preview_source == "ema" else model
    final = sample_likelihood_rectified_flow(
        sample_model,
        mixture,
        cfg.grid_size,
        cfg.flow_sample_steps,
        device,
        guidance_scale=cfg.flow_guidance_scale,
        conditioning=cfg.likelihood_conditioning,
        resp_temp=cfg.mixture_resp_temp,
        flow_objective=cfg.flow_objective,
    )
    save_tanh_grid(final.cpu(), out_dir / "samples_final.png", nrow=int(math.sqrt(cfg.grid_size)))
    if cfg.flow_eval_samples > 0:
        eval_samples = sample_likelihood_rectified_flow(
            sample_model,
            mixture,
            cfg.flow_eval_samples,
            cfg.flow_sample_steps,
            device,
            guidance_scale=cfg.flow_guidance_scale,
            conditioning=cfg.likelihood_conditioning,
            resp_temp=cfg.mixture_resp_temp,
            flow_objective=cfg.flow_objective,
        )
        metrics = evaluate_image_samples_against_cifar(eval_samples, cfg, device, out_dir, "likelihood_flow")
        print(json.dumps(metrics, indent=2))


def make_cifar_eval_loader(cfg: Config, train: bool, batch_size: int) -> DataLoader:
    dataset: Dataset = datasets.CIFAR10(root=cfg.data_dir, train=train, download=True, transform=transforms.ToTensor())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_particle_refiner_checkpoint(path: Path, device: torch.device) -> tuple[Config, nn.Module, torch.Tensor]:
    ckpt = torch.load(path, map_location=device)
    ckpt_cfg = Config(**ckpt["config"])
    model = ParticleRefinerUNet(
        dim=ckpt_cfg.unet_dim,
        dim_mults=parse_int_tuple(ckpt_cfg.unet_dim_mults),
        residual_scale=ckpt_cfg.particle_residual_scale,
    ).to(device)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    particle_bank = ckpt["particle_bank"].to(device)
    return ckpt_cfg, model, particle_bank


def load_cifar_likelihood_flow_checkpoint(
    path: Path,
    device: torch.device,
    *,
    use_ema: bool = True,
) -> tuple[Config, ComponentConditionalRectifiedFlowUNet, FeatureGaussianMixture]:
    ckpt = torch.load(path, map_location=device)
    ckpt_cfg = Config(**ckpt["config"])
    model = ComponentConditionalRectifiedFlowUNet(
        dim=ckpt_cfg.unet_dim,
        dim_mults=parse_int_tuple(ckpt_cfg.unet_dim_mults),
        cond_dim=ckpt_cfg.cond_dim,
        component_count=ckpt_cfg.mode_count,
        accept_dest_time=ckpt_cfg.flow_objective in {"mean", "split_mean", "lsd"},
    ).to(device)
    state_key = "ema_model" if use_ema and "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    means = ckpt["mixture"]["means"]
    mixture = FeatureGaussianMixture(means.to(device), sigma=ckpt_cfg.feature_sigma).to(device)
    mixture.load_state_dict(ckpt["mixture"])
    mixture.eval()
    return ckpt_cfg, model, mixture


def evaluate_cifar_likelihood_flow(cfg: Config, device: torch.device, out_dir: Path) -> None:
    if not cfg.eval_ckpt:
        raise ValueError("--eval-ckpt is required for --pipeline eval_cifar_likelihood_flow")

    from torchmetrics.image.fid import FrechetInceptionDistance

    ckpt_cfg, model, mixture = load_cifar_likelihood_flow_checkpoint(Path(cfg.eval_ckpt), device)
    eval_cfg = ckpt_cfg
    eval_cfg.data_dir = cfg.data_dir
    eval_cfg.num_workers = cfg.num_workers
    eval_cfg.eval_batch_size = cfg.eval_batch_size
    eval_cfg.eval_nn_samples = cfg.eval_nn_samples
    eval_cfg.eval_fid_split = cfg.eval_fid_split
    eval_cfg.flow_eval_samples = cfg.flow_eval_samples
    eval_cfg.flow_sample_steps = cfg.flow_sample_steps
    eval_cfg.flow_guidance_scale = cfg.flow_guidance_scale
    if cfg.likelihood_conditioning != Config().likelihood_conditioning:
        eval_cfg.likelihood_conditioning = cfg.likelihood_conditioning
    if cfg.flow_objective != Config().flow_objective:
        eval_cfg.flow_objective = cfg.flow_objective
    eval_cfg.mixture_resp_temp = cfg.mixture_resp_temp

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real_loader = make_cifar_eval_loader(
        eval_cfg,
        train=eval_cfg.eval_fid_split == "train",
        batch_size=eval_cfg.eval_batch_size,
    )
    real_seen = 0
    for x, _ in tqdm(real_loader, desc=f"fid real {eval_cfg.eval_fid_split}"):
        x = x.to(device, non_blocking=True)
        fid.update(x, real=True)
        real_seen += x.shape[0]

    train_loader = make_cifar_eval_loader(eval_cfg, train=True, batch_size=eval_cfg.eval_batch_size)
    train_feats = []
    for x, _ in tqdm(train_loader, desc="train nn features"):
        train_feats.append(particle_features(x.to(device, non_blocking=True).mul(2.0).sub(1.0), 8).cpu())
    train_feats = torch.cat(train_feats, dim=0).to(device)

    test_loader = make_cifar_eval_loader(eval_cfg, train=False, batch_size=eval_cfg.eval_batch_size)
    test_feats = []
    wanted_test = eval_cfg.eval_nn_samples
    for x, _ in test_loader:
        if wanted_test <= 0:
            break
        take = min(x.shape[0], wanted_test)
        x = x[:take].to(device, non_blocking=True).mul(2.0).sub(1.0)
        test_feats.append(particle_features(x, 8).cpu())
        wanted_test -= take

    def min_dists(query: torch.Tensor, bank: torch.Tensor, block: int = 8192) -> torch.Tensor:
        outs = []
        for start in range(0, query.shape[0], 256):
            chunk = query[start : start + 256]
            best = torch.full((chunk.shape[0],), float("inf"), device=device)
            for bank_start in range(0, bank.shape[0], block):
                dist = torch.cdist(chunk, bank[bank_start : bank_start + block]).min(dim=1).values
                best = torch.minimum(best, dist)
            outs.append(best)
        return torch.cat(outs, dim=0)

    gen_feat_chunks = []
    grid_chunks = []
    fake_seen = 0
    pbar = tqdm(total=eval_cfg.flow_eval_samples, desc="fid fake")
    while fake_seen < eval_cfg.flow_eval_samples:
        take = min(eval_cfg.eval_batch_size, eval_cfg.flow_eval_samples - fake_seen)
        samples = sample_likelihood_rectified_flow(
            model,
            mixture,
            take,
            eval_cfg.flow_sample_steps,
            device,
            guidance_scale=eval_cfg.flow_guidance_scale,
            conditioning=eval_cfg.likelihood_conditioning,
            resp_temp=eval_cfg.mixture_resp_temp,
            flow_objective=eval_cfg.flow_objective,
        )
        fid.update((samples + 1.0) * 0.5, real=False)
        if sum(chunk.shape[0] for chunk in gen_feat_chunks) < eval_cfg.eval_nn_samples:
            need = eval_cfg.eval_nn_samples - sum(chunk.shape[0] for chunk in gen_feat_chunks)
            gen_feat_chunks.append(particle_features(samples[:need], 8).cpu())
        if sum(chunk.shape[0] for chunk in grid_chunks) < eval_cfg.grid_size:
            need = eval_cfg.grid_size - sum(chunk.shape[0] for chunk in grid_chunks)
            grid_chunks.append(samples[:need].cpu())
        fake_seen += take
        pbar.update(take)
    pbar.close()

    name = (
        f"likelihood_flow_{eval_cfg.eval_fid_split}_g{eval_cfg.flow_guidance_scale:g}"
        f"_s{eval_cfg.flow_sample_steps}_n{eval_cfg.flow_eval_samples}"
    )
    if grid_chunks:
        grid = torch.cat(grid_chunks, dim=0)[: eval_cfg.grid_size]
        save_tanh_grid(grid, out_dir / f"{name}.png", nrow=int(math.sqrt(eval_cfg.grid_size)))

    metrics = {
        "name": name,
        "samples": int(eval_cfg.flow_eval_samples),
        f"fid_cifar10_{eval_cfg.eval_fid_split}": float(fid.compute().detach().cpu()),
        "real_split": eval_cfg.eval_fid_split,
        "real_samples": int(real_seen),
    }
    if gen_feat_chunks:
        gen_feats = torch.cat(gen_feat_chunks, dim=0).to(device)
        gen_dist = min_dists(gen_feats, train_feats)
        metrics.update(
            {
                "gen_to_train_nn_mean": float(gen_dist.mean().detach().cpu()),
                "gen_to_train_nn_median": float(gen_dist.median().detach().cpu()),
                "gen_exact_train_rate": float((gen_dist < 1e-7).float().mean().detach().cpu()),
            }
        )
    if test_feats:
        test_dist = min_dists(torch.cat(test_feats, dim=0).to(device), train_feats)
        metrics.update(
            {
                "test_to_train_nn_mean": float(test_dist.mean().detach().cpu()),
                "test_to_train_nn_median": float(test_dist.median().detach().cpu()),
            }
        )
    metrics.update(
        {
            "eval_ckpt": str(Path(cfg.eval_ckpt)),
            "flow_objective": eval_cfg.flow_objective,
            "conditioning": eval_cfg.likelihood_conditioning,
            "guidance_scale": eval_cfg.flow_guidance_scale,
            "flow_sample_steps": eval_cfg.flow_sample_steps,
        }
    )
    (out_dir / f"{name}_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


@torch.inference_mode()
def sample_particle_refiner_batch(
    model: nn.Module,
    particle_bank: torch.Tensor,
    cfg: Config,
    batch_size: int,
) -> torch.Tensor:
    indices = torch.randint(particle_bank.shape[0], (batch_size,), device=particle_bank.device)
    coarse = particle_bank[indices]
    if cfg.particle_noise > 0:
        coarse = (coarse + cfg.particle_noise * torch.randn_like(coarse)).clamp(-1.0, 1.0)
    return model(coarse)


def nearest_particle_stats(
    generated: torch.Tensor,
    test_images: torch.Tensor,
    particle_bank: torch.Tensor,
    feature_size: int,
) -> dict[str, float]:
    particle_feat = particle_features(particle_bank, feature_size).float()
    gen_feat = particle_features(generated, feature_size).float()
    test_feat = particle_features(test_images, feature_size).float()
    gen_dist = torch.cdist(gen_feat, particle_feat).min(dim=1).values
    test_dist = torch.cdist(test_feat, particle_feat).min(dim=1).values
    eps = 1e-7
    return {
        "gen_to_particle_nn_mean": float(gen_dist.mean().cpu()),
        "gen_to_particle_nn_median": float(gen_dist.median().cpu()),
        "test_to_particle_nn_mean": float(test_dist.mean().cpu()),
        "test_to_particle_nn_median": float(test_dist.median().cpu()),
        "gen_exact_particle_rate": float((gen_dist < eps).float().mean().cpu()),
    }


def evaluate_cifar_particle_refiner(cfg: Config, device: torch.device, out_dir: Path) -> None:
    if not cfg.eval_ckpt:
        raise ValueError("--eval-ckpt is required for --pipeline eval_cifar_particle_refiner")
    ckpt_cfg, model, particle_bank = load_particle_refiner_checkpoint(Path(cfg.eval_ckpt), device)

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as exc:
        raise RuntimeError("FID evaluation requires torchmetrics[image].") from exc

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    test_loader = make_cifar_eval_loader(cfg, train=False, batch_size=cfg.eval_batch_size)

    real_seen = 0
    test_for_nn = []
    for x, _ in tqdm(test_loader, desc="fid real"):
        x = x.to(device, non_blocking=True)
        take = min(x.shape[0], cfg.eval_samples - real_seen)
        if take <= 0:
            break
        x = x[:take]
        fid.update(x, real=True)
        if sum(t.shape[0] for t in test_for_nn) < cfg.eval_nn_samples:
            test_for_nn.append(x.mul(2.0).sub(1.0).cpu())
        real_seen += take

    gen_seen = 0
    gen_for_nn = []
    while gen_seen < cfg.eval_samples:
        take = min(cfg.eval_batch_size, cfg.eval_samples - gen_seen)
        samples = sample_particle_refiner_batch(model, particle_bank, ckpt_cfg, take)
        fid.update((samples + 1.0) * 0.5, real=False)
        if sum(t.shape[0] for t in gen_for_nn) < cfg.eval_nn_samples:
            gen_for_nn.append(samples.cpu())
        gen_seen += take

    metrics = {
        "eval_ckpt": str(Path(cfg.eval_ckpt)),
        "eval_samples": cfg.eval_samples,
        "fid_cifar10_test": float(fid.compute().detach().cpu()),
    }
    if gen_for_nn and test_for_nn:
        gen_nn = torch.cat(gen_for_nn, dim=0)[: cfg.eval_nn_samples].to(device)
        test_nn = torch.cat(test_for_nn, dim=0)[: cfg.eval_nn_samples].to(device)
        metrics.update(
            nearest_particle_stats(
                generated=gen_nn,
                test_images=test_nn,
                particle_bank=particle_bank,
                feature_size=ckpt_cfg.particle_feature_size,
            )
        )

    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def make_timestamped_out_dir(raw_out_dir: str) -> Path:
    base = Path(raw_out_dir)
    stamp = time.strftime("%d%m_%H%M")
    parent = base.parent if str(base.parent) else Path(".")
    name = base.name or "run"
    out_dir = parent / f"{stamp}_{name}"
    if not out_dir.exists():
        return out_dir

    for suffix in range(1, 100):
        candidate = parent / f"{stamp}_{name}_{suffix:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free run directory for {out_dir}")


# %%
def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if cfg.resume_ckpt and cfg.out_dir == Config().out_dir:
        out_dir = Path(cfg.resume_ckpt).parent
    elif cfg.resume_ckpt:
        out_dir = Path(cfg.out_dir)
    else:
        out_dir = make_timestamped_out_dir(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"device={device}; out_dir={out_dir}")

    if cfg.pipeline == "cifar_unet":
        train_cifar_unet(cfg, device, out_dir)
        print(f"Done. Samples and checkpoints are in {out_dir}")
        return

    if cfg.pipeline == "cifar_particle_refiner":
        train_cifar_particle_refiner(cfg, device, out_dir)
        print(f"Done. Samples and checkpoints are in {out_dir}")
        return

    if cfg.pipeline == "cifar_drift_particles":
        train_cifar_drift_particles(cfg, device, out_dir)
        print(f"Done. Samples and checkpoints are in {out_dir}")
        return

    if cfg.pipeline == "cifar_rectified_flow":
        train_cifar_rectified_flow(cfg, device, out_dir)
        print(f"Done. Samples and checkpoints are in {out_dir}")
        return

    if cfg.pipeline == "cifar_mixture_flow":
        train_cifar_mixture_flow(cfg, device, out_dir)
        print(f"Done. Samples and checkpoints are in {out_dir}")
        return

    if cfg.pipeline == "cifar_class_flow":
        train_cifar_class_flow(cfg, device, out_dir)
        print(f"Done. Samples and checkpoints are in {out_dir}")
        return

    if cfg.pipeline == "cifar_likelihood_flow":
        train_cifar_likelihood_flow(cfg, device, out_dir)
        print(f"Done. Samples and checkpoints are in {out_dir}")
        return

    if cfg.pipeline == "eval_cifar_particle_refiner":
        evaluate_cifar_particle_refiner(cfg, device, out_dir)
        print(f"Done. Evaluation is in {out_dir}")
        return

    if cfg.pipeline == "eval_cifar_likelihood_flow":
        evaluate_cifar_likelihood_flow(cfg, device, out_dir)
        print(f"Done. Evaluation is in {out_dir}")
        return

    train_loader, test_loader = make_image_loaders(cfg)
    vae = train_or_load_vae(cfg, train_loader, device, out_dir)
    save_reconstruction_grid(vae, test_loader, device, out_dir / "vae_recon.png", cfg)

    train_latents = encode_dataset(vae, train_loader, device, f"encoding {cfg.dataset} train")
    test_latents = encode_dataset(vae, test_loader, device, f"encoding {cfg.dataset} test")
    phi = train_feature_model(cfg, train_latents, device, out_dir)
    feature_mean, feature_std = fit_feature_stats(phi, train_latents, device, cfg.batch_size, cfg.feature_std_floor)
    torch.save({"mean": feature_mean, "std": feature_std}, out_dir / "feature_stats.pt")

    model = train_mixture(
        cfg=cfg,
        vae=vae,
        phi=phi,
        latent_ds=train_latents,
        test_latent_ds=test_latents,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
        out_dir=out_dir,
    )
    save_sample_grid(model, vae, out_dir / "samples_final.png", cfg.grid_size)
    print(f"Done. Samples and checkpoints are in {out_dir}")


if __name__ == "__main__":
    main()
