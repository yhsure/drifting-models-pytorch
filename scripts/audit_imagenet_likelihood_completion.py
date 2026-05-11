from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml


def _json_out_from_command(command: str) -> Path | None:
    return _path_arg_from_command(command, "--json-out")


def _path_arg_from_command(command: str, flag: str) -> Path | None:
    parts = shlex.split(command)
    for idx, part in enumerate(parts):
        if part == flag and idx + 1 < len(parts):
            return Path(parts[idx + 1]).expanduser().resolve()
        if part.startswith(f"{flag}="):
            return Path(part.split("=", 1)[1]).expanduser().resolve()
    return None


def _manifest_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        p = p / "comparison_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"comparison manifest not found: {p}")
    return p


def _eval_entries(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for key, name in (("primary_eval", "primary"), ("secondary_eval", "secondary")):
        entry = manifest.get(key, {})
        if entry.get("command"):
            entries.append((name, entry))
    for item in manifest.get("sampling_variant_evals", []):
        if item.get("command"):
            entries.append((str(item.get("name", "variant")), item))
    return entries


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _best_reference(manifest: dict[str, Any], cfg_scale: float | None) -> dict[str, Any] | None:
    if cfg_scale is not None and abs(float(cfg_scale) - 1.0) < 1e-6:
        refs = manifest.get("reference_cfg1_20k", [])
    else:
        refs = manifest.get("reference_5k", [])
    if cfg_scale is not None:
        refs = [ref for ref in refs if float(ref.get("cfg", -1)) == float(cfg_scale)]
    refs = [ref for ref in refs if "fid" in ref]
    if not refs:
        return None
    return min(refs, key=lambda ref: float(ref["fid"]))


def audit(run_dir: str) -> tuple[bool, list[dict[str, Any]]]:
    run_path = Path(run_dir).expanduser().resolve()
    manifest_path = _manifest_path(str(run_path))
    manifest = _load_json(manifest_path)
    train_step = int(manifest.get("train_step", 0))
    rows: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"name": name, "ok": bool(ok), "detail": detail})

    metadata_path = run_path / "params_ema" / "metadata.json"
    params_path = run_path / "params_ema" / "ema_params.pt"
    add("manifest", manifest_path.is_file(), str(manifest_path))
    expected_world_size = int(manifest.get("expected_world_size", 0) or 0)
    actual_world_size = manifest.get("actual_world_size")
    if expected_world_size > 0:
        try:
            actual_world_size_int = int(actual_world_size)
        except (TypeError, ValueError):
            actual_world_size_int = -1
        add(
            "world_size",
            actual_world_size_int == expected_world_size,
            f"actual={actual_world_size} expected={expected_world_size}",
        )

    config_path = run_path / "log" / "config.yaml"
    settings = dict(manifest.get("training_settings", {}) or {})
    if settings:
        if config_path.is_file():
            cfg = _load_yaml(config_path)
            train_cfg = dict(cfg.get("train", {}) or {})
            dataset_cfg = dict(cfg.get("dataset", {}) or {})
            model_cfg_logged = dict(cfg.get("model", {}) or {})
            forward_cfg = dict(train_cfg.get("forward_dict", {}) or {})
            likelihood_cfg = dict(cfg.get("likelihood_prior", {}) or {})
            checks = {
                "compile": cfg.get("compile"),
                "total_steps": train_cfg.get("total_steps"),
                "train_batch_size": train_cfg.get("train_batch_size"),
                "dataset_batch_size": dataset_cfg.get("batch_size"),
                "eval_batch_size": dataset_cfg.get("eval_batch_size"),
                "gen_per_label": forward_cfg.get("gen_per_label"),
                "component_count": likelihood_cfg.get("component_count"),
                "conditioning": likelihood_cfg.get("conditioning"),
                "topk_start": likelihood_cfg.get("topk_start"),
                "topk_final": likelihood_cfg.get("topk_final"),
                "temp_start": likelihood_cfg.get("temp_start"),
                "temp_final": likelihood_cfg.get("temp_final"),
                "temp_anneal_steps": likelihood_cfg.get("temp_anneal_steps"),
                "fit_samples": likelihood_cfg.get("fit_samples"),
                "nll_weight": likelihood_cfg.get("nll_weight"),
                "balance_weight": likelihood_cfg.get("balance_weight"),
                "component_dropout": likelihood_cfg.get("component_dropout"),
                "uniform_same_class_positive_fraction": likelihood_cfg.get("uniform_same_class_positive_fraction"),
                "prior_cache_path": likelihood_cfg.get("cache_path"),
                "model_likelihood_component_count": model_cfg_logged.get("likelihood_component_count"),
            }
            mismatches = []
            for key, expected in settings.items():
                if key == "fixed_cfg":
                    continue
                actual = checks.get(key)
                if isinstance(expected, float):
                    ok = actual is not None and abs(float(actual) - expected) < 1e-6
                else:
                    ok = actual == expected
                if not ok:
                    mismatches.append(f"{key}: actual={actual} expected={expected}")
            if checks["model_likelihood_component_count"] != settings.get("component_count"):
                mismatches.append(
                    "model_likelihood_component_count: "
                    f"actual={checks['model_likelihood_component_count']} expected={settings.get('component_count')}"
                )
            add("training_config", not mismatches, "; ".join(mismatches) if mismatches else str(config_path))
        else:
            add("training_config", False, f"missing {config_path}")
    add("ema_params", params_path.is_file(), str(params_path))
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        step_ok = train_step <= 0 or int(metadata.get("step", -1)) == train_step
        add("metadata_step", step_ok, f"step={metadata.get('step')} expected={train_step}")
        model_cfg = dict(metadata.get("model_config", {}) or {})
        add(
            "likelihood_model_config",
            int(model_cfg.get("likelihood_component_count", 0)) > 0,
            f"likelihood_component_count={model_cfg.get('likelihood_component_count')}",
        )
    else:
        add("metadata_step", False, f"missing {metadata_path}")
        add("likelihood_model_config", False, f"missing {metadata_path}")

    primary_result: dict[str, Any] | None = None
    for name, entry in _eval_entries(manifest):
        json_path = _json_out_from_command(str(entry["command"]))
        if json_path is None:
            add(f"eval_{name}", False, "missing --json-out")
            continue
        if not json_path.is_file():
            add(f"eval_{name}", False, f"missing {json_path}")
            continue
        result = _load_json(json_path)
        expected_cfg = entry.get("cfg_scale")
        expected_samples = entry.get("num_samples")
        cfg_ok = expected_cfg is None or abs(float(result.get("cfg_scale", -1)) - float(expected_cfg)) < 1e-6
        samples_ok = expected_samples is None or int(result.get("num_samples", -1)) == int(expected_samples)
        metrics_ok = result.get("fid") is not None and result.get("isc_mean") is not None
        step_ok = train_step <= 0 or int(result.get("eval_step", -1)) == train_step
        add(
            f"eval_{name}",
            cfg_ok and samples_ok and metrics_ok and step_ok,
            (
                f"json={json_path} cfg={result.get('cfg_scale')} samples={result.get('num_samples')} "
                f"step={result.get('eval_step')} fid={result.get('fid')} is={result.get('isc_mean')}"
            ),
        )
        if name == "primary":
            primary_result = result

    primary_cfg = None
    if primary_result is not None and primary_result.get("cfg_scale") is not None:
        primary_cfg = float(primary_result["cfg_scale"])
    best_ref = _best_reference(manifest, primary_cfg)
    if primary_result is None or best_ref is None or primary_result.get("fid") is None:
        add("primary_beats_reference", False, "missing primary result or matching reference")
    else:
        fid = float(primary_result["fid"])
        ref_fid = float(best_ref["fid"])
        check_name = "primary_beats_cfg1_20k_ref" if primary_cfg is not None and abs(primary_cfg - 1.0) < 1e-6 else "primary_beats_5k_ref"
        detail = f"fid={fid:.4f} ref={best_ref.get('run')} ref_fid={ref_fid:.4f}"
        add(check_name, fid < ref_fid, detail)
        add("primary_beats_reference", fid < ref_fid, detail)

    panel = manifest.get("nearest_train_clip_panel", {})
    if panel.get("command") or panel.get("output") or panel.get("json"):
        output_path = Path(panel["output"]).expanduser().resolve() if panel.get("output") else _path_arg_from_command(panel["command"], "--output")
        json_path = Path(panel["json"]).expanduser().resolve() if panel.get("json") else _json_out_from_command(panel["command"])
        output_ok = output_path is not None and output_path.is_file()
        json_ok = json_path is not None and json_path.is_file()
        if json_ok and json_path is not None:
            panel_result = _load_json(json_path)
            expected_n = int(panel.get("num_samples", 0))
            json_ok = expected_n <= 0 or int(panel_result.get("num_samples", -1)) == expected_n
        add(
            "nearest_train_clip_panel",
            output_ok and json_ok,
            f"output={output_path} json={json_path}",
        )

    return all(row["ok"] for row in rows), rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit whether an ImageNet likelihood run satisfies the comparison goal.")
    parser.add_argument("run_dir", help="Run directory containing comparison_manifest.json and params_ema.")
    parser.add_argument("--json-out", default="", help="Optional JSON path for audit rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ok, rows = audit(args.run_dir)
    print("check\tok\tdetail")
    for row in rows:
        print(f"{row['name']}\t{row['ok']}\t{row['detail']}")
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ok": ok, "checks": rows}, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
