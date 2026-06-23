"""FID-only inference entrypoint.

Usage:
    python inference.py --init-from "hf://latent_L_sota" --workdir runs/fid
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from dataset.dataset import get_postprocess_fn
from utils.env import HF_ROOT, IMAGENET_PATH
from utils.fid_util import evaluate_fid
from utils.init_util import load_generator_model_and_params
from utils.logging import WandbLogger
from utils.misc import run_init, stamp_workdir

run_init()


def _is_latent(metadata: dict) -> bool:
    model_cfg = metadata.get("model_config", {})
    return model_cfg.get("in_channels", 3) == 4


class _LabelOnlyImageFolder(Dataset):
    """ImageFolder-compatible labels without validation JPEG decode."""

    def __init__(self, root: str | Path):
        folder = ImageFolder(root=str(root))
        self.targets = [int(target) for target in folder.targets]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.empty(0), self.targets[index]


def _build_label_eval_loader(*, batch_size: int, split: str = "val") -> DataLoader:
    ds = _LabelOnlyImageFolder(Path(IMAGENET_PATH) / split)
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )
        batch_size = max(1, batch_size // dist.get_world_size())
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)


def _set_eval_seed(seed: int | None) -> None:
    if seed is None or int(seed) < 0:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(init_from: str, *, compile_model: bool = True):
    model, params, metadata = load_generator_model_and_params(init_from, hf_cache_dir=HF_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(params, strict=False)
    model.eval()
    if compile_model:
        model = torch.compile(model, dynamic=False, fullgraph=True)

    latent = _is_latent(metadata)
    postprocess_fn = get_postprocess_fn(use_aug=False, use_latent=False, use_cache=latent)
    return model, postprocess_fn, metadata, device


def generate_step(batch, params, apply_fn, postprocess_fn, cfg_scale=1.0):
    _, labels = batch
    model = params if isinstance(params, torch.nn.Module) else params["model"]
    device = next(model.parameters()).device
    labels = labels.to(device=device, dtype=torch.long)
    if apply_fn is None:
        apply_fn = lambda m, y, cfg: m(c=y, cfg_scale=cfg)["samples"]  # noqa: E731
    model.eval()
    with torch.inference_mode():
        latent_samples = apply_fn(model, labels, cfg_scale)
        return postprocess_fn(latent_samples).cpu()


def _infer_eval_step(init_from: str) -> int:
    run_path = Path(init_from).expanduser()
    if not run_path.exists():
        return 0
    if run_path.is_file():
        try:
            return int(run_path.stem.removeprefix("step_"))
        except ValueError:
            return 0
    checkpoint_dir = run_path / "checkpoints"
    if not checkpoint_dir.exists():
        return 0
    steps = []
    for ckpt in checkpoint_dir.glob("step_*.pt"):
        try:
            steps.append(int(ckpt.stem.removeprefix("step_")))
        except ValueError:
            continue
    return max(steps, default=0)


def _infer_run_dir(init_from: str) -> Path:
    init_path = Path(init_from).expanduser()
    if init_path.is_file() and init_path.parent.name == "checkpoints":
        return init_path.parent.parent
    if init_path.name == "checkpoints" and init_path.is_dir():
        return init_path.parent
    return init_path


def run_eval_fid(
    gen_step_jit,
    params,
    metadata,
    init_from: str,
    workdir: str,
    *,
    num_samples: int,
    cfg_scale: float,
    eval_batch_size: int,
    eval_isc: bool,
    use_wandb: bool,
    wandb_entity: str | None,
    wandb_project: str,
    wandb_name: str | None,
    wandb_run_id: str | None = None,
    wandb_mode: str | None = None,
) -> dict:
    postprocess_fn = gen_step_jit["postprocess_fn"]
    apply_fn = gen_step_jit["apply_fn"]
    eval_loader = _build_label_eval_loader(batch_size=eval_batch_size)

    work_path = Path(workdir).resolve()
    init_path = _infer_run_dir(init_from)
    train_metadata = WandbLogger.read_run_metadata(init_path) if init_path.exists() else {}
    if train_metadata:
        use_wandb = use_wandb or bool(train_metadata.get("use_wandb", False))
        wandb_run_id = wandb_run_id or train_metadata.get("run_id")
        wandb_project = wandb_project or train_metadata.get("project") or "drift"
        wandb_entity = wandb_entity or train_metadata.get("entity")
        wandb_name = wandb_name or train_metadata.get("name")
        # Keep standalone eval metrics and images in the training run directory
        # when updating the corresponding WandB experiment.
        if wandb_run_id:
            work_path = init_path.resolve()

    eval_step = _infer_eval_step(init_from)
    logger = WandbLogger()
    logger.set_logging(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"{Path(init_from).name}_fid",
        use_wandb=use_wandb,
        workdir=str(work_path),
        log_every_k=1,
        run_id=wandb_run_id,
        mode=wandb_mode,
    )
    logger.set_step(eval_step)

    metrics = evaluate_fid(
        dataset_name="imagenet256",
        gen_func=generate_step,
        gen_params={
            "params": params,
            "apply_fn": apply_fn,
            "cfg_scale": cfg_scale,
            "postprocess_fn": postprocess_fn,
        },
        eval_loader=eval_loader,
        logger=logger,
        num_samples=num_samples,
        log_folder="eval",
        log_prefix=f"cfg_{cfg_scale:g}",
        eval_prc_recall=(num_samples >= 50000),
        eval_isc=eval_isc,
        eval_fid=True,
    )
    log_payload = {
        "eval/cfg_scale": cfg_scale,
        **{f"eval/{k}": v for k, v in metrics.items()},
    }
    if "fid" in metrics:
        log_payload["samples/final_fid"] = metrics["fid"]
    if "isc_mean" in metrics:
        log_payload["samples/final_isc_mean"] = metrics["isc_mean"]
    if "isc_std" in metrics:
        log_payload["samples/final_isc_std"] = metrics["isc_std"]
    log_payload["samples/final_cfg_scale"] = cfg_scale
    log_payload["samples/step"] = eval_step
    logger.log_dict(log_payload)
    logger.finish()
    return {
        "init_from": init_from,
        "cfg_scale": cfg_scale,
        "eval_step": eval_step,
        "wandb_run_id": wandb_run_id,
        "metadata": metadata,
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference: FID evaluation.")
    parser.add_argument("--init-from", required=True, help="hf://<name> or local checkpoint path.")
    parser.add_argument("--workdir", default="runs/infer", help="Output directory.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--hsdp-dim", type=int, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-isc", action="store_true")
    parser.add_argument("--seed", type=int, default=-1, help="Set RNG seed for reproducible sampling; <0 disables.")
    return parser


def run_inference_from_args(args: argparse.Namespace) -> dict:
    _ = args.hsdp_dim
    _set_eval_seed(args.seed)
    model, postprocess_fn, metadata, device = _load_model(args.init_from, compile_model=not args.no_compile)
    _ = device
    gen_step_jit = {
        "apply_fn": lambda m, y, cfg: m(c=y, cfg_scale=cfg)["samples"],
        "postprocess_fn": postprocess_fn,
    }
    result = run_eval_fid(
        gen_step_jit,
        model,
        metadata,
        args.init_from,
        args.workdir,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        eval_batch_size=args.eval_batch_size,
        eval_isc=not args.no_isc,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_run_id=args.wandb_run_id,
        wandb_mode=args.wandb_mode,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    args.workdir = stamp_workdir(args.workdir)
    result = run_inference_from_args(args)
    print(json.dumps(result, indent=2))
    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
