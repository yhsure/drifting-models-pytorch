from __future__ import annotations

import argparse
import json
import shlex
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


def _eval_entries(manifest: dict[str, Any]) -> list[tuple[str, Path | None]]:
    entries: list[tuple[str, Path | None]] = []
    primary = manifest.get("primary_eval", {})
    if primary.get("command"):
        entries.append(("primary", _json_out_from_command(primary["command"])))
    secondary = manifest.get("secondary_eval", {})
    if secondary.get("command"):
        entries.append(("secondary", _json_out_from_command(secondary["command"])))
    for item in manifest.get("sampling_variant_evals", []):
        if item.get("command"):
            entries.append((str(item.get("name", "variant")), _json_out_from_command(item["command"])))
    return entries


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, result_path in _eval_entries(manifest):
        result = _load_json(result_path)
        if result is None:
            rows.append({"name": name, "json": str(result_path or ""), "status": "missing"})
            continue
        cfg_scale = result.get("cfg_scale")
        ref = _best_reference(manifest, float(cfg_scale) if cfg_scale is not None else None)
        fid = result.get("fid")
        isc = result.get("isc_mean")
        ref_fid = ref.get("fid") if ref else None
        ref_is = ref.get("is") if ref else None
        rows.append(
            {
                "name": name,
                "json": str(result_path or ""),
                "status": "ok",
                "cfg": cfg_scale,
                "conditioning": result.get("likelihood_conditioning", ""),
                "topk": result.get("likelihood_topk", ""),
                "fid": fid,
                "is": isc,
                "ref": ref.get("run") if ref else "",
                "ref_step": ref.get("step") if ref else "",
                "ref_fid": ref_fid,
                "ref_is": ref_is,
                "fid_delta": (float(fid) - float(ref_fid)) if fid is not None and ref_fid is not None else None,
                "is_delta": (float(isc) - float(ref_is)) if isc is not None and ref_is is not None else None,
                "beats_ref": bool(fid is not None and ref_fid is not None and float(fid) < float(ref_fid)),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "usage_entropy": result.get("likelihood_component_usage_entropy"),
                "prior_entropy": result.get("likelihood_class_prior_entropy"),
                "selected_mass": result.get("likelihood_selected_mass_mean"),
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print("name\tstatus\tcfg\tconditioning\ttopk\tfid\tFID-ref\tIS\tIS-ref\tref\tref-step\tprecision\trecall\tusage-H\tprior-H\tmass\tbeats\tjson")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row.get("name", "")),
                    str(row.get("status", "")),
                    _fmt(row.get("cfg")),
                    str(row.get("conditioning", "")),
                    _fmt(row.get("topk"), digits=0),
                    _fmt(row.get("fid")),
                    _fmt(row.get("fid_delta")),
                    _fmt(row.get("is")),
                    _fmt(row.get("is_delta")),
                    str(row.get("ref", "")),
                    _fmt(row.get("ref_step"), digits=0),
                    _fmt(row.get("precision")),
                    _fmt(row.get("recall")),
                    _fmt(row.get("usage_entropy")),
                    _fmt(row.get("prior_entropy")),
                    _fmt(row.get("selected_mass")),
                    str(row.get("beats_ref", "")),
                    str(row.get("json", "")),
                ]
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize ImageNet likelihood evals against manifest references.")
    parser.add_argument("manifest_or_run_dir", help="comparison_manifest.json path or run directory containing it.")
    parser.add_argument("--json-out", default="", help="Optional path to write the summary rows as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = json.loads(_manifest_path(args.manifest_or_run_dir).read_text(encoding="utf-8"))
    rows = summarize(manifest)
    print_table(rows)
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
