from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import main_gen
from utils.misc import load_config, stamp_workdir


def _ensure_dict(parent, key: str):
    if key not in parent or parent[key] is None:
        parent[key] = {}
    return parent[key]


def _env_rank() -> int:
    for key in ("RANK", "SLURM_PROCID"):
        value = str(os.environ.get(key, "")).strip()
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def _env_int(key: str) -> int | None:
    value = str(os.environ.get(key, "")).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _detected_world_size() -> int | None:
    world = _env_int("WORLD_SIZE")
    if world is not None:
        return world
    nnodes = _env_int("SLURM_NNODES")
    local_world = _env_int("LOCAL_WORLD_SIZE")
    if nnodes is not None and local_world is not None:
        return nnodes * local_world
    nproc_per_node = _env_int("DRIFT_NPROC_PER_NODE")
    if nnodes is not None and nproc_per_node is not None:
        return nnodes * nproc_per_node
    if nnodes is not None:
        try:
            import torch

            cuda_devices = int(torch.cuda.device_count())
        except Exception:
            cuda_devices = 0
        if cuda_devices > 0:
            return nnodes * cuda_devices
    return None


def validate_expected_world_size(expected_world_size: int) -> None:
    expected = int(expected_world_size)
    if expected <= 0:
        return
    actual = _detected_world_size()
    if actual is None:
        return
    if int(actual) != expected:
        raise RuntimeError(
            f"Explicit-likelihood ImageNet run expected world_size={expected} for comparable references, "
            f"but detected world_size={actual}. Set --expected-world-size 0 to disable this guard."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ImageNet explicit-likelihood prior prototype.")
    parser.add_argument("--base-config", default="configs/gen/latent_sota_B.yaml")
    parser.add_argument("--run-root", default="runs/imagenet_likelihood_sparse_soft_5k")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--component-count", type=int, default=8192)
    parser.add_argument("--conditioning", choices=("sparse_soft", "hard", "top1", "soft"), default="sparse_soft")
    parser.add_argument("--topk-start", type=int, default=16)
    parser.add_argument("--topk-final", type=int, default=8)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-final", type=float, default=0.45)
    parser.add_argument("--temp-anneal-steps", type=int, default=3500)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--dataset-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--gen-per-label", type=int, default=8)
    parser.add_argument("--save-per-step", type=int, default=2000)
    parser.add_argument("--eval-per-step", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--fixed-cfg", type=float, default=1.0)
    parser.add_argument("--feature-sigma", type=float, default=0.0)
    parser.add_argument("--nll-weight", type=float, default=0.03)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--component-dropout", type=float, default=0.10)
    parser.add_argument("--uniform-same-class-positive-fraction", type=float, default=0.20)
    parser.add_argument("--fit-samples", type=int, default=0, help="Prior-fit sample count; <=0 uses the full train set.")
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--fit-num-workers", type=int, default=0)
    parser.add_argument("--kmeans-iters", type=int, default=40)
    parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    parser.add_argument("--responsibility-batch-size", type=int, default=512)
    parser.add_argument("--prior-cache-path", default="", help="Optional fitted likelihood prior cache to load/reuse.")
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=32,
        help="Fail fast when torchrun world size differs from comparable reference runs; <=0 disables.",
    )
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument(
        "--comparison-family",
        choices=("auto", "imagenet_5k", "imagenet_20k_1node"),
        default="auto",
        help="Reference/eval family for the comparison manifest.",
    )
    parser.add_argument("--profile", action="store_true")
    return parser


def apply_overrides(config, args: argparse.Namespace) -> None:
    config.setdefault("model", {})
    config.model["likelihood_component_count"] = int(args.component_count)
    config.model["likelihood_descriptor_dim"] = int(config.model.get("likelihood_descriptor_dim", 0))
    config.model["likelihood_sigma"] = float(args.feature_sigma)
    config.model["likelihood_conditioning"] = str(args.conditioning)
    config.model["likelihood_topk"] = int(args.topk_final)
    config.model["likelihood_embedding_scale"] = float(config.model.get("likelihood_embedding_scale", 1.0))

    if args.compile is not None:
        config["compile"] = int(args.compile)

    dataset = _ensure_dict(config, "dataset")
    dataset["batch_size"] = int(args.dataset_batch_size)
    dataset["eval_batch_size"] = int(args.eval_batch_size)

    optimizer = _ensure_dict(config, "optimizer")
    lr_schedule = _ensure_dict(optimizer, "lr_schedule")
    lr_schedule["total_steps"] = int(args.total_steps)

    train = _ensure_dict(config, "train")
    train["total_steps"] = int(args.total_steps)
    train["train_batch_size"] = int(args.train_batch_size)
    train["save_per_step"] = int(args.save_per_step)
    train["eval_per_step"] = int(args.eval_per_step)
    train["eval_samples"] = int(args.eval_samples)
    train["run_eval"] = bool(args.eval_per_step > 0 and args.eval_samples > 0)
    train["eval_on_step_one"] = False
    train.setdefault("keep_last", 2)
    train.setdefault("push_at_resume", 1)
    forward = _ensure_dict(train, "forward_dict")
    forward["gen_per_label"] = int(args.gen_per_label)
    forward["cfg_min"] = float(args.fixed_cfg)
    forward["cfg_max"] = float(args.fixed_cfg)
    forward.setdefault("neg_cfg_pw", 3.0)
    train["cfg_list"] = [float(args.fixed_cfg)]

    likelihood_prior = {
        "enabled": True,
        "component_count": int(args.component_count),
        "conditioning": str(args.conditioning),
        "topk_start": int(args.topk_start),
        "topk_final": int(args.topk_final),
        "temp_start": float(args.temp_start),
        "temp_final": float(args.temp_final),
        "temp_anneal_steps": int(args.temp_anneal_steps),
        "feature_sigma": float(args.feature_sigma),
        "nll_weight": float(args.nll_weight),
        "balance_weight": float(args.balance_weight),
        "component_dropout": float(args.component_dropout),
        "uniform_same_class_positive_fraction": float(args.uniform_same_class_positive_fraction),
        "fit_samples": int(args.fit_samples),
        "fit_batch_size": int(args.fit_batch_size),
        "fit_num_workers": int(args.fit_num_workers),
        "kmeans_iters": int(args.kmeans_iters),
        "kmeans_batch_size": int(args.kmeans_batch_size),
        "responsibility_batch_size": int(args.responsibility_batch_size),
        "activation_kwargs": {
            "patch_mean_size": [],
            "patch_std_size": [],
            "use_std": True,
            "use_mean": True,
            "every_k_block": 0.0,
        },
    }
    if bool(config.get("feature", {}).get("use_mae", True)):
        likelihood_prior["descriptor_keys"] = ["layer4_mean", "layer4_std"]
    prior_cache_path = str(getattr(args, "prior_cache_path", "")).strip()
    if prior_cache_path:
        likelihood_prior["cache_path"] = prior_cache_path
    config["likelihood_prior"] = likelihood_prior


def _comparison_family(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "comparison_family", "auto") or "auto")
    if explicit != "auto":
        return explicit
    if (
        int(getattr(args, "total_steps", 0)) >= 20000
        and int(getattr(args, "train_batch_size", 0)) <= 16
        and int(getattr(args, "gen_per_label", 0)) <= 8
        and abs(float(getattr(args, "fixed_cfg", 1.0)) - 1.0) < 1e-6
    ):
        return "imagenet_20k_1node"
    return "imagenet_5k"


def write_comparison_manifest(workdir: str, args: argparse.Namespace) -> None:
    run_path = Path(workdir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    if _env_rank() != 0:
        return

    comparison_family = _comparison_family(args)
    primary_cfg = 1.0 if comparison_family == "imagenet_20k_1node" else 2.5
    secondary_cfg = 2.5 if comparison_family == "imagenet_20k_1node" else 1.0
    panel_cfg = primary_cfg
    eval_root_base = f"{str(getattr(args, 'run_root', 'runs/imagenet_likelihood_sparse_soft_5k')).rstrip('/')}_eval"
    alt_topk = 4 if int(args.topk_final) != 4 else 16
    training_settings = {
        "total_steps": int(args.total_steps),
        "train_batch_size": int(args.train_batch_size),
        "dataset_batch_size": int(args.dataset_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "gen_per_label": int(args.gen_per_label),
        "fixed_cfg": float(args.fixed_cfg),
        "component_count": int(args.component_count),
        "conditioning": str(args.conditioning),
        "topk_start": int(args.topk_start),
        "topk_final": int(args.topk_final),
        "temp_start": float(args.temp_start),
        "temp_final": float(args.temp_final),
        "temp_anneal_steps": int(args.temp_anneal_steps),
        "fit_samples": int(args.fit_samples),
        "nll_weight": float(args.nll_weight),
        "balance_weight": float(args.balance_weight),
        "component_dropout": float(args.component_dropout),
        "uniform_same_class_positive_fraction": float(args.uniform_same_class_positive_fraction),
        "compile": int(args.compile) if args.compile is not None else None,
    }
    if str(args.prior_cache_path).strip():
        training_settings["prior_cache_path"] = str(args.prior_cache_path).strip()

    def eval_command(*, cfg_scale: float, conditioning: str, topk: int, suffix: str) -> str:
        eval_root = f"{eval_root_base}/{run_path.name}_{suffix}"
        return (
            f".venv/bin/python inference.py --init-from {run_path} "
            f"--workdir {eval_root} "
            f"--cfg-scale {cfg_scale:g} --num-samples 50000 --eval-batch-size 512 "
            f"--likelihood-conditioning {conditioning} --likelihood-topk {topk} "
            f"--likelihood-temperature 1.0 --json-out {eval_root}/result.json"
        )

    panel_root = f"{eval_root_base}/{run_path.name}_clip_panel"
    panel_output = f"{panel_root}/nearest_train_clip_panel.png"
    panel_json = f"{panel_root}/nearest_train_clip_panel.json"
    panel_command = (
        f".venv/bin/python scripts/make_imagenet_likelihood_clip_panel.py --init-from {run_path} "
        f"--output {panel_output} --json-out {panel_json} "
        f"--cfg-scale {panel_cfg:g} --num-samples 8 --candidates-per-sample 128 --topk 3 "
        f"--local-files-only "
        f"--likelihood-conditioning {args.conditioning} --likelihood-topk {int(args.topk_final)} "
        f"--likelihood-temperature 1.0"
    )

    manifest = {
        "purpose": "Comparable ImageNet explicit-likelihood prototype.",
        "comparison_family": comparison_family,
        "base_config": args.base_config,
        "train_step": int(args.total_steps),
        "expected_world_size": int(getattr(args, "expected_world_size", 32)),
        "actual_world_size": _detected_world_size(),
        "training_settings": training_settings,
        "primary_eval": {
            "cfg_scale": primary_cfg,
            "num_samples": 50000,
            "eval_batch_size": 512,
            "command": eval_command(
                cfg_scale=primary_cfg,
                conditioning=args.conditioning,
                topk=int(args.topk_final),
                suffix=f"cfg{primary_cfg:g}_{args.conditioning}_top{args.topk_final}",
            ),
        },
        "secondary_eval": {
            "cfg_scale": secondary_cfg,
            "num_samples": 50000,
            "eval_batch_size": 512,
            "command": eval_command(
                cfg_scale=secondary_cfg,
                conditioning=args.conditioning,
                topk=int(args.topk_final),
                suffix=f"cfg{secondary_cfg:g}_{args.conditioning}_top{args.topk_final}",
            ),
        },
        "sampling_variant_evals": [
            {
                "name": f"{args.conditioning}_top{args.topk_final}",
                "cfg_scale": primary_cfg,
                "num_samples": 50000,
                "command": eval_command(
                    cfg_scale=primary_cfg,
                    conditioning=args.conditioning,
                    topk=int(args.topk_final),
                    suffix=f"cfg{primary_cfg:g}_{args.conditioning}_top{args.topk_final}",
                ),
            },
            {
                "name": "hard_top1",
                "cfg_scale": primary_cfg,
                "num_samples": 50000,
                "command": eval_command(
                    cfg_scale=primary_cfg,
                    conditioning="hard",
                    topk=1,
                    suffix=f"cfg{primary_cfg:g}_hard_top1",
                ),
            },
            {
                "name": f"sparse_soft_top{alt_topk}",
                "cfg_scale": primary_cfg,
                "num_samples": 50000,
                "command": eval_command(
                    cfg_scale=primary_cfg,
                    conditioning="sparse_soft",
                    topk=alt_topk,
                    suffix=f"cfg{primary_cfg:g}_sparse_soft_top{alt_topk}",
                ),
            },
        ],
        "nearest_train_clip_panel": {
            "cfg_scale": panel_cfg,
            "num_samples": 8,
            "candidates_per_sample": 128,
            "topk": 3,
            "output": panel_output,
            "json": panel_json,
            "command": panel_command,
        },
        "reference_5k": [
            {"run": "0501_0019_latent_ablation_30k_mae640_407971", "cfg": 2.5, "fid": 29.50, "is": 51.76},
            {
                "run": "0501_0023_latent_ablation_30k_mae640_pos_enhanced_feat_407972",
                "cfg": 2.5,
                "fid": 24.59,
                "is": 68.04,
            },
            {
                "run": "0501_0937_latent_ablation_5k_mae640_pos_enhanced_feat_408529",
                "cfg": 2.5,
                "fid": 23.37,
                "is": 75.80,
            },
        ],
        "reference_cfg1_20k": [
            {"run": "baseline drifting", "step": 20000, "cfg": 1.0, "samples": 50000, "fid": 32.09, "is": 39.67},
            {"run": "pos-enhanced", "step": 20000, "cfg": 1.0, "samples": 50000, "fid": 29.45, "is": 43.34},
            {
                "run": "separate-weight baseline",
                "step": 20000,
                "cfg": 1.0,
                "samples": 25000,
                "fid": 30.55,
                "is": 41.73,
            },
            {
                "run": "pos-enhanced feature anchors",
                "step": 20000,
                "cfg": 1.0,
                "samples": 50000,
                "fid": 24.68,
                "is": 58.03,
            },
        ],
    }
    (run_path / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def resolve_workdir(args: argparse.Namespace) -> str:
    tag = f"k{args.component_count}_top{args.topk_final}_tau{args.temp_final:g}"
    job_id = str(os.environ.get("SLURM_JOB_ID", "")).strip()
    if job_id:
        return str(Path(args.run_root).expanduser() / f"{job_id}_{tag}")
    return stamp_workdir(str(Path(args.run_root) / tag))


def main() -> None:
    args = build_parser().parse_args()
    validate_expected_world_size(int(args.expected_world_size))
    config = load_config(args.base_config)
    apply_overrides(config, args)
    workdir = resolve_workdir(args)
    write_comparison_manifest(workdir, args)
    main_gen(config, output_dir=workdir, profile=args.profile)


if __name__ == "__main__":
    main()
