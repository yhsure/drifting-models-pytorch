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


def _compile_kwargs_for_metadata(metadata: dict) -> dict:
    model_cfg = dict(metadata.get("model_config", {}) or {})
    if int(model_cfg.get("likelihood_component_count", 0) or 0) > 0:
        return {"dynamic": False}
    return {"dynamic": False, "fullgraph": True}


def _load_model(init_from: str):
    model, params, metadata = load_generator_model_and_params(init_from, hf_cache_dir=HF_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if hasattr(model, "resize_likelihood_prior_from_state_dict"):
        model.resize_likelihood_prior_from_state_dict(params)
    model.load_state_dict(params, strict=False)
    model.eval()
    model = torch.compile(model, **_compile_kwargs_for_metadata(metadata))

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


def _unwrap_model(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


class LikelihoodSamplingTracker:
    def __init__(self, component_count: int):
        self.component_count = int(component_count)
        self.weighted_counts = torch.zeros(self.component_count, dtype=torch.float64)
        self.class_prior_entropy_sum = 0.0
        self.selected_mass_sum = 0.0
        self.samples = 0
        self.component_slots = 0

    @torch.no_grad()
    def update(
        self,
        *,
        logits: torch.Tensor,
        components: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> None:
        if components is None or self.component_count <= 0:
            return
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        components = components.to(device=probs.device, dtype=torch.long)
        if components.ndim == 1:
            components = components[:, None]
            component_weights = torch.ones(components.shape, device=probs.device, dtype=torch.float32)
        elif weights is None:
            component_weights = torch.full(
                components.shape,
                1.0 / max(1, int(components.shape[1])),
                device=probs.device,
                dtype=torch.float32,
            )
        else:
            component_weights = weights.to(device=probs.device, dtype=torch.float32)
        components = components.clamp(0, self.component_count - 1)
        selected_mass = probs.gather(1, components).sum(dim=1)

        counts = torch.zeros(self.component_count, device=probs.device, dtype=torch.float64)
        counts.scatter_add_(0, components.reshape(-1), component_weights.double().reshape(-1))
        self.weighted_counts += counts.cpu()
        self.class_prior_entropy_sum += float(entropy.sum().detach().cpu().item())
        self.selected_mass_sum += float(selected_mass.sum().detach().cpu().item())
        self.samples += int(components.shape[0])
        self.component_slots += int(components.numel())

    def summary(self) -> dict[str, float]:
        if self.samples <= 0 or self.weighted_counts.sum() <= 0:
            return {}
        usage = self.weighted_counts / self.weighted_counts.sum().clamp_min(1e-12)
        usage_entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        return {
            "likelihood_component_usage_entropy": float(usage_entropy.item()),
            "likelihood_component_usage_perplexity": float(torch.exp(usage_entropy).item()),
            "likelihood_component_unique": float((self.weighted_counts > 0).sum().item()),
            "likelihood_class_prior_entropy": self.class_prior_entropy_sum / float(self.samples),
            "likelihood_selected_mass_mean": self.selected_mass_sum / float(self.samples),
            "likelihood_samples_tracked": float(self.samples),
            "likelihood_component_slots_tracked": float(self.component_slots),
        }


@torch.no_grad()
def _sample_likelihood_condition(
    model,
    labels: torch.Tensor,
    *,
    conditioning: str,
    topk: int,
    temperature: float,
    tracker: LikelihoodSamplingTracker | None = None,
):
    base = _unwrap_model(model)
    if not hasattr(base, "sample_likelihood_components") or not base.has_likelihood_prior():
        return None, None
    components, weights = base.sample_likelihood_components(
        labels,
        conditioning=conditioning,
        topk=topk,
        temperature=temperature,
    )
    if tracker is not None and components is not None:
        labels = labels.to(device=base.likelihood_class_logits.device, dtype=torch.long).clamp(0, base.num_classes - 1)
        logits = base.likelihood_class_logits[labels] / max(float(temperature), 1e-6)
        tracker.update(logits=logits, components=components, weights=weights)
    return components, weights


def _make_likelihood_tracker(model) -> LikelihoodSamplingTracker | None:
    base = _unwrap_model(model)
    if not hasattr(base, "has_likelihood_prior") or not base.has_likelihood_prior():
        return None
    return LikelihoodSamplingTracker(int(base.likelihood_component_count))


def _infer_eval_step(init_from: str) -> int:
    run_path = Path(init_from).expanduser()
    if not run_path.exists():
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
    wandb_run_id: str | None = None,
    wandb_mode: str | None = None,
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
    init_path = Path(init_from).expanduser()
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
        eval_isc=True,
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
    parser.add_argument(
        "--likelihood-conditioning",
        default="",
        choices=("", "none", "hard", "top1", "sparse_soft", "soft"),
        help="Override explicit-likelihood sampling mode for likelihood-prior checkpoints.",
    )
    parser.add_argument("--likelihood-topk", type=int, default=0, help="Top-k components for sparse likelihood sampling.")
    parser.add_argument("--likelihood-temperature", type=float, default=1.0, help="Temperature for class-conditional prior sampling.")
    return parser


def run_inference_from_args(args: argparse.Namespace) -> dict:
    _ = args.hsdp_dim
    model, postprocess_fn, metadata, device = _load_model(args.init_from)
    _ = device
    model_cfg = dict(metadata.get("model_config", {}) or {})
    likelihood_conditioning = args.likelihood_conditioning or str(model_cfg.get("likelihood_conditioning", "none"))
    likelihood_topk = int(args.likelihood_topk or model_cfg.get("likelihood_topk", 1))
    likelihood_temperature = float(args.likelihood_temperature)
    likelihood_tracker = _make_likelihood_tracker(model)

    def apply_fn(m, y, cfg):
        components, weights = _sample_likelihood_condition(
            m,
            y,
            conditioning=likelihood_conditioning,
            topk=likelihood_topk,
            temperature=likelihood_temperature,
            tracker=likelihood_tracker,
        )
        return m(
            c=y,
            cfg_scale=cfg,
            likelihood_components=components,
            likelihood_weights=weights,
            likelihood_conditioning=likelihood_conditioning,
            likelihood_topk=likelihood_topk,
            likelihood_temperature=likelihood_temperature,
        )["samples"]

    gen_step_jit = {
        "apply_fn": apply_fn,
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
        wandb_run_id=args.wandb_run_id,
        wandb_mode=args.wandb_mode,
    )
    result["likelihood_conditioning"] = likelihood_conditioning
    result["likelihood_topk"] = likelihood_topk
    result["likelihood_temperature"] = likelihood_temperature
    result["num_samples"] = int(args.num_samples)
    result["eval_batch_size"] = int(args.eval_batch_size)
    if likelihood_tracker is not None:
        result.update(likelihood_tracker.summary())
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
