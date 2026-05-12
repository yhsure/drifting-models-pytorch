# %%
"""Adversarial decoder fine-tune for CIFAR decoder-style mixture experiments.

The goal is intentionally narrow: keep the encoder/latent geometry fixed,
fine-tune only the CIFAR VAE decoder with an image discriminator, then evaluate
whether old decoder-style mixture latents look better through the sharper
decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixture


# %%
class PatchDiscriminator(nn.Module):
    def __init__(self, channels: int = 3, base: int = 64):
        super().__init__()

        def block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            block(channels, base, 2),
            block(base, base * 2, 2),
            block(base * 2, base * 4, 2),
            nn.utils.spectral_norm(nn.Conv2d(base * 4, base * 4, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base * 4, 1, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = []
        h = x
        for layer in self.net:
            h = layer(h)
            if return_features and h.ndim == 4 and h.shape[1] > 1:
                features.append(h)
        logits = h.flatten(1).mean(dim=1)
        if return_features:
            return logits, features
        return logits


def image_edge_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    recon_dx = recon[..., :, 1:] - recon[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    recon_dy = recon[..., 1:, :] - recon[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy)


def make_out_dir(root: str) -> Path:
    root_path = Path(root)
    out_dir = root_path / f"{time.strftime('%d%m_%H%M')}_decoder_gan"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_loader(data_dir: str, train: bool, batch_size: int, num_workers: int) -> DataLoader:
    ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transforms.ToTensor())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def cycle(loader: DataLoader):
    while True:
        yield from loader


def build_vae(device: torch.device, ckpt_path: Path, latent_dim: int, vae_arch: str) -> nn.Module:
    cfg = mixture.Config(
        dataset="cifar10",
        latent_dim=latent_dim,
        vae_arch=vae_arch,
        vae_epochs=30,
        vae_lr=0.002,
        vae_beta=0.0,
        vae_l1_weight=1.0,
        vae_perceptual_weight=0.5,
        vae_edge_weight=0.5,
        feature_std_floor=1.0,
    )
    vae = mixture.build_vae(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "vae" in ckpt:
        state = ckpt["vae"]
    else:
        state = ckpt
    vae.load_state_dict(state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for module in (vae.decoder_fc, vae.decoder):
        module.train()
        for p in module.parameters():
            p.requires_grad_(True)
    return vae


@torch.inference_mode()
def save_recon_grid(vae: nn.Module, loader: DataLoader, device: torch.device, path: Path, n: int = 50) -> None:
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    z = vae.encode(x, sample=False)
    recon = vae.decode(z)
    paired = torch.stack([x, recon], dim=1).reshape(-1, 3, 32, 32)
    save_image(paired.cpu(), path, nrow=10)


@torch.inference_mode()
def collect_recon_samples(
    vae: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n: int,
) -> torch.Tensor:
    outs = []
    seen = 0
    vae.eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        take = min(x.shape[0], n - seen)
        if take <= 0:
            break
        x = x[:take]
        recon = vae.decode(vae.encode(x, sample=False))
        outs.append(recon.cpu().mul(2.0).sub(1.0))
        seen += take
    return torch.cat(outs, dim=0)


def load_decoder_mixture(path: Path, device: torch.device) -> tuple[mixture.Config, mixture.StochasticModeMixture]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = mixture.Config(**{k: v for k, v in ckpt["config"].items() if k in mixture.Config.__dataclass_fields__})
    model = mixture.StochasticModeMixture(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model


@torch.inference_mode()
def collect_mixture_samples(
    model: mixture.StochasticModeMixture,
    vae: nn.Module,
    cfg: mixture.Config,
    n: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outs = []
    vae.eval()
    model.eval()
    for start in range(0, n, batch_size):
        take = min(batch_size, n - start)
        z, _, _ = model.sample_latents(take, 1)
        decoded = vae.decode(z, temperature=cfg.sample_temperature)
        outs.append(decoded.detach().cpu().clamp(0.0, 1.0).mul(2.0).sub(1.0))
    return torch.cat(outs, dim=0)


def evaluate_samples(samples: torch.Tensor, name: str, out_dir: Path, batch_size: int, num_workers: int, device: torch.device) -> dict:
    cfg = mixture.Config(dataset="cifar10", data_dir="data", eval_batch_size=batch_size, num_workers=num_workers)
    metrics = mixture.evaluate_image_samples_against_cifar(samples, cfg, device, out_dir, name)
    save_image((samples[:100] + 1.0) * 0.5, out_dir / f"{name}_grid.png", nrow=10)
    return metrics


def parse_temperatures(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = make_out_dir(args.out_dir)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"device={device}; out_dir={out_dir}")

    mixture.seed_everything(args.seed)
    train_loader = make_loader(args.data_dir, True, args.batch_size, args.num_workers)
    test_loader = make_loader(args.data_dir, False, args.eval_batch_size, args.num_workers)
    batches = cycle(train_loader)

    vae = build_vae(device, Path(args.vae_ckpt), args.latent_dim, args.vae_arch)
    disc = PatchDiscriminator(base=args.disc_dim).to(device)
    perceptual = None
    if args.perceptual_weight > 0:
        perceptual = mixture.CIFARPerceptualLoss().to(device).eval()
        for p in perceptual.parameters():
            p.requires_grad_(False)

    opt_g = torch.optim.AdamW(
        list(vae.decoder_fc.parameters()) + list(vae.decoder.parameters()),
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay,
    )
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)

    metrics_path = out_dir / "metrics.jsonl"
    save_recon_grid(vae, test_loader, device, out_dir / "recon_before.png")
    if args.mixture_ckpt:
        mix_cfg, mix_model = load_decoder_mixture(Path(args.mixture_ckpt), device)
        before = collect_mixture_samples(mix_model, vae, mix_cfg, min(args.eval_samples, 2048), args.eval_batch_size, device)
        evaluate_samples(before, "mixture_before", out_dir, args.eval_batch_size, args.num_workers, device)
    else:
        mix_cfg = mix_model = None

    pbar = tqdm(range(1, args.steps + 1), desc="decoder-gan")
    for step in pbar:
        x, _ = next(batches)
        x = x.to(device, non_blocking=True)
        real = x.mul(2.0).sub(1.0)

        with torch.no_grad():
            z = vae.encode(x, sample=False)

        recon = vae.decode(z)
        fake = recon.mul(2.0).sub(1.0)

        mix_fake = None
        if mix_model is not None and args.mixture_adv_weight > 0:
            with torch.no_grad():
                z_mix, _, _ = mix_model.sample_latents(x.shape[0], 1)
            mix_recon = vae.decode(z_mix, temperature=args.mixture_train_temperature)
            mix_fake = mix_recon.mul(2.0).sub(1.0)

        real_logits = disc(real)
        fake_logits = disc(fake.detach())
        d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
        if mix_fake is not None and args.disc_mix_weight > 0:
            mix_fake_logits = disc(mix_fake.detach())
            d_loss = d_loss + args.disc_mix_weight * F.relu(1.0 + mix_fake_logits).mean()
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        fake_logits_g, fake_feats = disc(fake, return_features=True)
        mix_adv = torch.zeros((), device=device)
        mix_fake_logit = torch.zeros((), device=device)
        if mix_fake is not None:
            mix_fake_logits_g = disc(mix_fake)
            mix_fake_logit = mix_fake_logits_g.mean()
            mix_adv = -mix_fake_logit
        with torch.no_grad():
            _, real_feats = disc(real, return_features=True)
        l1 = F.l1_loss(recon, x)
        mse = F.mse_loss(recon, x)
        edge = image_edge_loss(recon, x)
        perc = perceptual(recon, x) if perceptual is not None else torch.zeros((), device=device)
        fm = sum(
            F.l1_loss(fake_feat.mean(dim=0), real_feat.mean(dim=0))
            for fake_feat, real_feat in zip(fake_feats, real_feats)
        )
        adv = -fake_logits_g.mean()
        g_loss = (
            args.l1_weight * l1
            + args.mse_weight * mse
            + args.edge_weight * edge
            + args.perceptual_weight * perc
            + args.fm_weight * fm
            + args.adv_weight * adv
            + args.mixture_adv_weight * mix_adv
        )
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        metric = {
            "step": step,
            "g_loss": float(g_loss.detach().cpu()),
            "d_loss": float(d_loss.detach().cpu()),
            "l1": float(l1.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "edge": float(edge.detach().cpu()),
            "perceptual": float(perc.detach().cpu()),
            "feature_matching": float(fm.detach().cpu()),
            "adv": float(adv.detach().cpu()),
            "mixture_adv": float(mix_adv.detach().cpu()),
            "real_logit": float(real_logits.mean().detach().cpu()),
            "fake_logit": float(fake_logits.mean().detach().cpu()),
            "mixture_fake_logit": float(mix_fake_logit.detach().cpu()),
        }
        pbar.set_postfix(l1=metric["l1"], adv=metric["adv"], d=metric["d_loss"])
        if step == 1 or step % args.log_every == 0:
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metric) + "\n")
        if args.sample_every > 0 and (step == 1 or step % args.sample_every == 0):
            save_recon_grid(vae, test_loader, device, out_dir / f"recon_step{step:06d}.png")

    save_recon_grid(vae, test_loader, device, out_dir / "recon_final.png")
    torch.save(
        {
            "vae": vae.state_dict(),
            "disc": disc.state_dict(),
            "args": vars(args),
        },
        out_dir / "decoder_gan.pt",
    )

    recon_samples = collect_recon_samples(vae, test_loader, device, args.eval_samples)
    recon_metrics = evaluate_samples(recon_samples, "recon_final", out_dir, args.eval_batch_size, args.num_workers, device)
    print("recon_final", json.dumps(recon_metrics, indent=2))

    if args.mixture_ckpt and mix_model is not None and mix_cfg is not None:
        mix_samples = collect_mixture_samples(mix_model, vae, mix_cfg, args.eval_samples, args.eval_batch_size, device)
        mix_metrics = evaluate_samples(mix_samples, "mixture_final", out_dir, args.eval_batch_size, args.num_workers, device)
        print("mixture_final", json.dumps(mix_metrics, indent=2))

        temperatures = parse_temperatures(args.eval_temperatures)
        if temperatures:
            temp_dir = out_dir / f"fid{args.eval_temperature_samples}_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            summary = []
            for temperature in temperatures:
                mix_cfg.sample_temperature = temperature
                name = f"mixture_t{temperature:g}_{args.eval_temperature_samples // 1000}k"
                samples = collect_mixture_samples(
                    mix_model,
                    vae,
                    mix_cfg,
                    args.eval_temperature_samples,
                    args.eval_batch_size,
                    device,
                )
                metrics = evaluate_samples(samples, name, temp_dir, args.eval_batch_size, args.num_workers, device)
                (temp_dir / f"{name}_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                summary.append([temperature, metrics["fid_cifar10_test"], metrics["gen_to_train_nn_median"]])
                print(name, json.dumps(metrics, indent=2))
            (temp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="runs/cifar_decoder_gan")
    parser.add_argument("--vae-ckpt", default="runs/mixture_cache/cifar10_vae_residual8_full_z2048_e30_lr0.002_wd0_b0_rauto_l11_p0.5_edge0.5_det.pt")
    parser.add_argument("--mixture-ckpt", default="runs/cifar10_mixture_full_v1/0504_143333/cifar10_mixture.pt")
    parser.add_argument("--latent-dim", type=int, default=2048)
    parser.add_argument("--vae-arch", choices=["auto", "conv", "residual", "residual8"], default="residual8")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-temperatures", default="")
    parser.add_argument("--eval-temperature-samples", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--disc-dim", type=int, default=64)
    parser.add_argument("--adv-weight", type=float, default=0.05)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--fm-weight", type=float, default=0.0)
    parser.add_argument("--mixture-adv-weight", type=float, default=0.0)
    parser.add_argument("--mixture-train-temperature", type=float, default=0.9)
    parser.add_argument("--disc-mix-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
