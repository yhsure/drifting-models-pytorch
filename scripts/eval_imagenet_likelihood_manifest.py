from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _manifest_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        p = p / "comparison_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"comparison manifest not found: {p}")
    return p


def _json_out_from_command(command: str) -> Path | None:
    parts = shlex.split(command)
    for idx, part in enumerate(parts):
        if part == "--json-out" and idx + 1 < len(parts):
            return Path(parts[idx + 1]).expanduser().resolve()
        if part.startswith("--json-out="):
            return Path(part.split("=", 1)[1]).expanduser().resolve()
    return None


def _entries(manifest: dict[str, Any], include: set[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if "primary" in include and manifest.get("primary_eval", {}).get("command"):
        out.append(("primary", manifest["primary_eval"]["command"]))
    if "secondary" in include and manifest.get("secondary_eval", {}).get("command"):
        out.append(("secondary", manifest["secondary_eval"]["command"]))
    if "variants" in include:
        for item in manifest.get("sampling_variant_evals", []):
            name = str(item.get("name", "variant"))
            command = item.get("command")
            if command:
                out.append((name, command))
    panel = manifest.get("nearest_train_clip_panel", {})
    if "panel" in include and panel.get("command"):
        out.append(("nearest_train_clip_panel", panel["command"]))
    return out


def _load_result(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _print_summary(rows: list[tuple[str, Path | None, dict[str, Any] | None]]) -> None:
    print("name\tjson\tfid\tis\tprecision\trecall\tcfg\tconditioning\ttopk\tusage-H\tmass")
    for name, path, result in rows:
        if not result:
            print(f"{name}\t{path or ''}\t\t\t\t\t\t\t\t\t")
            continue
        fid = result.get("fid", "")
        isc = result.get("isc_mean", "")
        precision = result.get("precision", "")
        recall = result.get("recall", "")
        cfg = result.get("cfg_scale", "")
        conditioning = result.get("likelihood_conditioning", "")
        topk = result.get("likelihood_topk", "")
        usage_entropy = result.get("likelihood_component_usage_entropy", "")
        selected_mass = result.get("likelihood_selected_mass_mean", "")
        print(
            f"{name}\t{path or ''}\t{fid}\t{isc}\t{precision}\t{recall}\t"
            f"{cfg}\t{conditioning}\t{topk}\t{usage_entropy}\t{selected_mass}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run eval commands from an ImageNet likelihood comparison manifest.")
    parser.add_argument("manifest_or_run_dir", help="comparison_manifest.json path or run directory containing it.")
    parser.add_argument(
        "--include",
        default="primary,secondary,variants,panel",
        help="Comma-separated subset: primary,secondary,variants,panel.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force", action="store_true", help="Run even when the command's --json-out already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = _manifest_path(args.manifest_or_run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    include = {x.strip() for x in args.include.split(",") if x.strip()}
    entries = _entries(manifest, include)
    if not entries:
        raise SystemExit(f"No eval commands selected from {manifest_path}")

    rows: list[tuple[str, Path | None, dict[str, Any] | None]] = []
    for name, command in entries:
        json_out = _json_out_from_command(command)
        existing = _load_result(json_out)
        if existing is not None and not args.force:
            print(f"[skip] {name}: {json_out}")
            rows.append((name, json_out, existing))
            continue
        print(f"[run] {name}: {command}")
        if args.dry_run:
            rows.append((name, json_out, existing))
            continue
        subprocess.run(shlex.split(command), check=True)
        rows.append((name, json_out, _load_result(json_out)))

    _print_summary(rows)


if __name__ == "__main__":
    main()
