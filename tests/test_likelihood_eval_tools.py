from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import torch

from utils.misc import EasyDict


ROOT = Path(__file__).resolve().parents[1]


def _load_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    return _load_path(ROOT / "scripts" / name)


def test_eval_manifest_extracts_json_out_and_selected_entries():
    tool = _load_script("eval_imagenet_likelihood_manifest.py")
    manifest = {
        "primary_eval": {"command": "python inference.py --json-out runs/primary/result.json"},
        "secondary_eval": {"command": "python inference.py --json-out=runs/secondary/result.json"},
        "sampling_variant_evals": [
            {"name": "hard_top1", "command": "python inference.py --json-out runs/hard/result.json"},
        ],
    }

    entries = tool._entries(manifest, {"primary", "variants"})
    assert [name for name, _ in entries] == ["primary", "hard_top1"]
    assert str(tool._json_out_from_command(entries[0][1])).endswith("runs/primary/result.json")
    assert str(tool._json_out_from_command(entries[1][1])).endswith("runs/hard/result.json")


def test_eval_manifest_default_includes_secondary_and_variants():
    tool = _load_script("eval_imagenet_likelihood_manifest.py")
    args = tool.build_parser().parse_args(["runs/example"])
    assert args.include == "primary,secondary,variants,panel"


def test_run_script_defaults_to_compiled_entrypoint():
    tool = _load_script("run_imagenet_likelihood_prior.py")
    args = tool.build_parser().parse_args([])
    assert args.compile == 1


def test_eval_manifest_can_select_nearest_panel():
    tool = _load_script("eval_imagenet_likelihood_manifest.py")
    manifest = {
        "nearest_train_clip_panel": {
            "command": "python scripts/make_imagenet_likelihood_clip_panel.py --json-out panel.json",
        }
    }

    entries = tool._entries(manifest, {"panel"})
    assert entries == [
        ("nearest_train_clip_panel", "python scripts/make_imagenet_likelihood_clip_panel.py --json-out panel.json")
    ]


def test_summarize_manifest_reports_reference_delta(tmp_path: Path):
    tool = _load_script("summarize_imagenet_likelihood_results.py")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "cfg_scale": 2.5,
                "fid": 22.5,
                "isc_mean": 80.0,
                "precision": 0.61,
                "recall": 0.37,
                "likelihood_conditioning": "sparse_soft",
                "likelihood_topk": 8,
                "likelihood_component_usage_entropy": 1.25,
                "likelihood_class_prior_entropy": 1.75,
                "likelihood_selected_mass_mean": 0.42,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "primary_eval": {"command": f"python inference.py --json-out {result_path}"},
        "reference_5k": [
            {"run": "baseline", "cfg": 2.5, "fid": 29.5, "is": 51.76},
            {"run": "best", "cfg": 2.5, "fid": 23.37, "is": 75.8},
        ],
    }

    rows = tool.summarize(manifest)
    assert rows[0]["status"] == "ok"
    assert rows[0]["ref"] == "best"
    assert rows[0]["fid_delta"] == 22.5 - 23.37
    assert rows[0]["is_delta"] == 80.0 - 75.8
    assert rows[0]["beats_ref"] is True
    assert rows[0]["precision"] == 0.61
    assert rows[0]["recall"] == 0.37
    assert rows[0]["usage_entropy"] == 1.25
    assert rows[0]["prior_entropy"] == 1.75
    assert rows[0]["selected_mass"] == 0.42


def test_run_script_writes_full_comparison_manifest(tmp_path: Path, monkeypatch):
    tool = _load_script("run_imagenet_likelihood_prior.py")
    run_dir = tmp_path / "run"
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "32")
    args = SimpleNamespace(
        base_config="configs/gen/latent_sota_B.yaml",
        total_steps=5000,
        train_batch_size=64,
        dataset_batch_size=1024,
        eval_batch_size=2048,
        gen_per_label=64,
        fixed_cfg=1.0,
        component_count=8192,
        conditioning="sparse_soft",
        topk_start=16,
        topk_final=8,
        temp_start=1.5,
        temp_final=0.45,
        temp_anneal_steps=3500,
        fit_samples=0,
        nll_weight=0.03,
        balance_weight=0.02,
        component_dropout=0.1,
        uniform_same_class_positive_fraction=0.2,
        compile=0,
        prior_cache_path="",
        expected_world_size=32,
    )

    tool.write_comparison_manifest(str(run_dir), args)
    manifest = json.loads((run_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_world_size"] == 32
    assert manifest["actual_world_size"] == 32
    assert manifest["training_settings"]["component_count"] == 8192
    assert manifest["training_settings"]["topk_final"] == 8
    assert manifest["training_settings"]["compile"] == 0
    assert manifest["primary_eval"]["cfg_scale"] == 2.5
    assert manifest["secondary_eval"]["cfg_scale"] == 1.0
    assert len(manifest["sampling_variant_evals"]) == 3
    assert manifest["nearest_train_clip_panel"]["output"].endswith("nearest_train_clip_panel.png")
    assert "make_imagenet_likelihood_clip_panel.py" in manifest["nearest_train_clip_panel"]["command"]
    assert "--local-files-only" in manifest["nearest_train_clip_panel"]["command"]
    assert {entry["name"] for entry in manifest["sampling_variant_evals"]} == {
        "sparse_soft_top8",
        "hard_top1",
        "sparse_soft_top4",
    }
    assert any(ref["fid"] == 23.37 for ref in manifest["reference_5k"])
    assert any(ref["fid"] == 24.68 and ref["cfg"] == 1.0 for ref in manifest["reference_cfg1_20k"])


def test_run_script_writes_20k_1node_manifest_against_cfg1_references(tmp_path: Path, monkeypatch):
    tool = _load_script("run_imagenet_likelihood_prior.py")
    run_dir = tmp_path / "run20k"
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "4")
    args = SimpleNamespace(
        base_config="configs/gen/latent_sota_B.yaml",
        run_root="runs/imagenet_likelihood_sparse_soft_20k_1node",
        total_steps=20000,
        train_batch_size=16,
        dataset_batch_size=256,
        eval_batch_size=256,
        gen_per_label=8,
        fixed_cfg=1.0,
        component_count=8192,
        conditioning="sparse_soft",
        topk_start=16,
        topk_final=8,
        temp_start=1.5,
        temp_final=0.45,
        temp_anneal_steps=3500,
        fit_samples=0,
        nll_weight=0.03,
        balance_weight=0.02,
        component_dropout=0.1,
        uniform_same_class_positive_fraction=0.2,
        compile=1,
        prior_cache_path="runs/old/likelihood_prior.pt",
        expected_world_size=4,
        comparison_family="imagenet_20k_1node",
    )

    tool.write_comparison_manifest(str(run_dir), args)
    manifest = json.loads((run_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["comparison_family"] == "imagenet_20k_1node"
    assert manifest["expected_world_size"] == 4
    assert manifest["primary_eval"]["cfg_scale"] == 1.0
    assert manifest["secondary_eval"]["cfg_scale"] == 2.5
    assert manifest["nearest_train_clip_panel"]["cfg_scale"] == 1.0
    assert manifest["training_settings"]["compile"] == 1
    assert manifest["training_settings"]["prior_cache_path"] == "runs/old/likelihood_prior.pt"
    assert "imagenet_likelihood_sparse_soft_20k_1node_eval" in manifest["primary_eval"]["command"]
    assert any(ref["run"] == "pos-enhanced feature anchors" for ref in manifest["reference_cfg1_20k"])


def test_run_script_only_rank_zero_writes_manifest(tmp_path: Path, monkeypatch):
    tool = _load_script("run_imagenet_likelihood_prior.py")
    run_dir = tmp_path / "run"
    monkeypatch.setenv("RANK", "1")
    args = SimpleNamespace(
        base_config="configs/gen/latent_sota_B.yaml",
        total_steps=5000,
        conditioning="sparse_soft",
        topk_final=8,
    )

    tool.write_comparison_manifest(str(run_dir), args)
    assert run_dir.exists()
    assert not (run_dir / "comparison_manifest.json").exists()


def test_run_script_likelihood_activation_config_is_finite():
    tool = _load_script("run_imagenet_likelihood_prior.py")
    config = EasyDict(
        {
            "model": {},
            "dataset": {},
            "optimizer": {"lr_schedule": {}},
            "train": {},
            "feature": {"use_mae": True},
        }
    )
    args = SimpleNamespace(
        component_count=4,
        conditioning="sparse_soft",
        topk_start=2,
        topk_final=2,
        temp_start=1.0,
        temp_final=1.0,
        temp_anneal_steps=0,
        feature_sigma=1.0,
        nll_weight=0.0,
        balance_weight=0.0,
        component_dropout=0.0,
        uniform_same_class_positive_fraction=0.2,
        fit_samples=8,
        fit_batch_size=2,
        fit_num_workers=0,
        kmeans_iters=1,
        kmeans_batch_size=2,
        responsibility_batch_size=2,
        compile=None,
        dataset_batch_size=8,
        eval_batch_size=8,
        total_steps=1,
        train_batch_size=1,
        save_per_step=1,
        eval_per_step=0,
        eval_samples=0,
        gen_per_label=1,
        fixed_cfg=1.0,
    )

    tool.apply_overrides(config, args)
    assert config["likelihood_prior"]["activation_kwargs"]["every_k_block"] == 0.0


def test_run_script_uses_stable_slurm_job_workdir(monkeypatch):
    tool = _load_script("run_imagenet_likelihood_prior.py")
    args = SimpleNamespace(
        run_root="runs/imagenet_likelihood_sparse_soft_5k",
        component_count=8192,
        topk_final=8,
        temp_final=0.45,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "427582")

    assert tool.resolve_workdir(args).endswith("runs/imagenet_likelihood_sparse_soft_5k/427582_k8192_top8_tau0.45")


def test_run_script_world_size_guard_accepts_matching_world(monkeypatch):
    tool = _load_script("run_imagenet_likelihood_prior.py")
    monkeypatch.setenv("WORLD_SIZE", "32")

    tool.validate_expected_world_size(32)


def test_run_script_world_size_guard_rejects_mismatch(monkeypatch):
    tool = _load_script("run_imagenet_likelihood_prior.py")
    monkeypatch.setenv("WORLD_SIZE", "16")

    try:
        tool.validate_expected_world_size(32)
    except RuntimeError as exc:
        assert "expected world_size=32" in str(exc)
        assert "detected world_size=16" in str(exc)
    else:
        raise AssertionError("Expected world-size mismatch to raise RuntimeError")


def test_run_script_world_size_detects_slurm_nproc_fallback(monkeypatch):
    tool = _load_script("run_imagenet_likelihood_prior.py")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("SLURM_NNODES", "8")
    monkeypatch.setenv("DRIFT_NPROC_PER_NODE", "4")

    assert tool._detected_world_size() == 32
    tool.validate_expected_world_size(32)


def test_summarize_manifest_uses_secondary_cfg1_reference(tmp_path: Path):
    tool = _load_script("summarize_imagenet_likelihood_results.py")
    result_path = tmp_path / "cfg1.json"
    result_path.write_text(
        json.dumps({"cfg_scale": 1.0, "fid": 24.0, "isc_mean": 60.0}),
        encoding="utf-8",
    )
    manifest = {
        "secondary_eval": {"command": f"python inference.py --json-out {result_path}"},
        "reference_5k": [{"run": "best-5k", "cfg": 2.5, "fid": 23.37, "is": 75.8}],
        "reference_cfg1_20k": [
            {"run": "weak", "step": 20000, "cfg": 1.0, "fid": 32.09, "is": 39.67},
            {"run": "best-cfg1", "step": 20000, "cfg": 1.0, "fid": 24.68, "is": 58.03},
        ],
    }

    rows = tool.summarize(manifest)
    assert rows[0]["status"] == "ok"
    assert rows[0]["ref"] == "best-cfg1"
    assert rows[0]["ref_step"] == 20000
    assert rows[0]["beats_ref"] is True


def test_likelihood_sampling_tracker_reports_actual_component_usage():
    inference = _load_path(ROOT / "inference.py")
    tracker = inference.LikelihoodSamplingTracker(component_count=4)
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0]]).clamp_min(1e-6))
    components = torch.tensor([[0, 1], [1, 2]])
    weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]])

    tracker.update(logits=logits, components=components, weights=weights)
    summary = tracker.summary()
    assert summary["likelihood_samples_tracked"] == 2.0
    assert summary["likelihood_component_slots_tracked"] == 4.0
    assert summary["likelihood_component_unique"] == 3.0
    assert 0.0 < summary["likelihood_component_usage_entropy"] < 1.3864
    assert abs(summary["likelihood_selected_mass_mean"] - 0.9) < 1e-5


def test_inference_keeps_likelihood_checkpoints_compiled_without_fullgraph():
    inference = _load_path(ROOT / "inference.py")

    assert inference._compile_kwargs_for_metadata({"model_config": {}}) == {"dynamic": False, "fullgraph": True}
    assert inference._compile_kwargs_for_metadata({"model_config": {"likelihood_component_count": 8192}}) == {
        "dynamic": False
    }


def test_completion_audit_requires_checkpoint_evals_and_primary_improvement(tmp_path: Path):
    audit_tool = _load_script("audit_imagenet_likelihood_completion.py")
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    (run_dir / "params_ema").mkdir(parents=True)
    (run_dir / "log").mkdir()
    eval_dir.mkdir()
    (run_dir / "params_ema" / "ema_params.pt").write_bytes(b"params")
    (run_dir / "params_ema" / "metadata.json").write_text(
        json.dumps({"step": 5000, "model_config": {"likelihood_component_count": 8}}),
        encoding="utf-8",
    )
    primary = eval_dir / "primary.json"
    secondary = eval_dir / "secondary.json"
    variant = eval_dir / "variant.json"
    panel_png = eval_dir / "nearest_train_clip_panel.png"
    panel_json = eval_dir / "nearest_train_clip_panel.json"
    panel_png.write_bytes(b"png")
    panel_json.write_text(json.dumps({"num_samples": 8}), encoding="utf-8")
    for path, cfg, fid in [(primary, 2.5, 22.0), (secondary, 1.0, 25.0), (variant, 2.5, 23.0)]:
        path.write_text(
            json.dumps({"cfg_scale": cfg, "num_samples": 50000, "eval_step": 5000, "fid": fid, "isc_mean": 80.0}),
            encoding="utf-8",
        )
    manifest = {
        "train_step": 5000,
        "expected_world_size": 32,
        "actual_world_size": 32,
        "training_settings": {
            "total_steps": 5000,
            "train_batch_size": 64,
            "dataset_batch_size": 1024,
            "eval_batch_size": 2048,
            "gen_per_label": 64,
            "fixed_cfg": 1.0,
            "component_count": 8,
            "conditioning": "sparse_soft",
            "topk_start": 16,
            "topk_final": 8,
            "temp_start": 1.5,
            "temp_final": 0.45,
            "temp_anneal_steps": 3500,
            "fit_samples": 0,
            "nll_weight": 0.03,
            "balance_weight": 0.02,
            "component_dropout": 0.1,
            "uniform_same_class_positive_fraction": 0.2,
        },
        "primary_eval": {"cfg_scale": 2.5, "num_samples": 50000, "command": f"python inference.py --json-out {primary}"},
        "secondary_eval": {"cfg_scale": 1.0, "num_samples": 50000, "command": f"python inference.py --json-out {secondary}"},
        "sampling_variant_evals": [
            {"name": "hard_top1", "cfg_scale": 2.5, "num_samples": 50000, "command": f"python inference.py --json-out {variant}"},
        ],
        "nearest_train_clip_panel": {
            "num_samples": 8,
            "output": str(panel_png),
            "json": str(panel_json),
            "command": f"python scripts/make_imagenet_likelihood_clip_panel.py --output {panel_png} --json-out {panel_json}",
        },
        "reference_5k": [{"run": "best", "cfg": 2.5, "fid": 23.37, "is": 75.8}],
    }
    (run_dir / "comparison_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "log" / "config.yaml").write_text(
        """
dataset:
  batch_size: 1024
  eval_batch_size: 2048
model:
  likelihood_component_count: 8
train:
  total_steps: 5000
  train_batch_size: 64
  forward_dict:
    gen_per_label: 64
likelihood_prior:
  component_count: 8
  conditioning: sparse_soft
  topk_start: 16
  topk_final: 8
  temp_start: 1.5
  temp_final: 0.45
  temp_anneal_steps: 3500
  fit_samples: 0
  nll_weight: 0.03
  balance_weight: 0.02
  component_dropout: 0.1
  uniform_same_class_positive_fraction: 0.2
""".lstrip(),
        encoding="utf-8",
    )

    ok, rows = audit_tool.audit(str(run_dir))
    assert ok is True
    assert all(row["ok"] for row in rows)
    assert any(row["name"] == "world_size" and row["ok"] for row in rows)
    assert any(row["name"] == "training_config" and row["ok"] for row in rows)

    primary.write_text(
        json.dumps({"cfg_scale": 2.5, "num_samples": 50000, "eval_step": 5000, "fid": 24.0, "isc_mean": 80.0}),
        encoding="utf-8",
    )
    ok, rows = audit_tool.audit(str(run_dir))
    assert ok is False
    assert any(row["name"] == "primary_beats_5k_ref" and not row["ok"] for row in rows)


def test_postrun_script_finds_latest_manifest_run(tmp_path: Path):
    tool = _load_script("run_imagenet_likelihood_postrun.py")
    older = tmp_path / "0507_1000_k8192"
    newer = tmp_path / "0507_1100_k8192"
    older.mkdir()
    newer.mkdir()
    (older / "comparison_manifest.json").write_text("{}", encoding="utf-8")
    (newer / "comparison_manifest.json").write_text("{}", encoding="utf-8")

    # Ensure mtime ordering is deterministic on filesystems with coarse timestamps.
    older.touch()
    newer.touch()
    latest = tool._latest_run(str(tmp_path))
    assert latest == newer.resolve()
