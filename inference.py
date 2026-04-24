"""FID-only inference entrypoint.

Usage:
    python inference.py --init-from "hf://latent_L_sota" --workdir runs/fid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset.dataset import create_imagenet_split, get_postprocess_fn
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.init_util import load_generator_model_and_params
from utils.logging import WandbLogger
from utils.misc import run_init, stamp_workdir

run_init()


def _is_latent(metadata: dict) -> bool:
    model_cfg = metadata.get("model_config", {})
    return model_cfg.get("in_channels", 3) == 4


def _load_model(init_from: str):
    model, params, metadata = load_generator_model_and_params(init_from, hf_cache_dir=HF_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(params, strict=False)
    model.eval()
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
    use_wandb: bool,
    wandb_entity: str | None,
    wandb_project: str,
    wandb_name: str | None,
) -> dict:
    postprocess_fn = gen_step_jit["postprocess_fn"]
    apply_fn = gen_step_jit["apply_fn"]
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()

    eval_loader, _, _ = create_imagenet_split(
        resolution=256,
        split="val",
        batch_size=eval_batch_size // world_size,
        num_workers=0,
    )

    work_path = Path(workdir).resolve()
    logger = WandbLogger()
    logger.set_logging(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"{Path(init_from).name}_fid",
        use_wandb=use_wandb,
        workdir=str(work_path),
        log_every_k=1,
    )

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
        log_folder="fid_eval",
        log_prefix=f"cfg_{cfg_scale:g}",
        eval_prc_recall=(num_samples >= 50000),
        eval_isc=True,
        eval_fid=True,
    )
    logger.finish()
    return {"init_from": init_from, "cfg_scale": cfg_scale, "metadata": metadata, **metrics}


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
    parser.add_argument("--wandb-project", type=str, default="release-fid")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser


def run_inference_from_args(args: argparse.Namespace) -> dict:
    _ = args.hsdp_dim
    model, postprocess_fn, metadata, device = _load_model(args.init_from)
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
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
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
