from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def _latest_run(root: str) -> Path:
    root_path = Path(root).expanduser().resolve()
    candidates = [
        p
        for p in root_path.glob("*")
        if p.is_dir() and (p / "comparison_manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories with comparison_manifest.json under {root_path}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(x) for x in command))
    if not dry_run:
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ImageNet likelihood post-training eval, summary, and completion audit.")
    parser.add_argument("run_dir", nargs="?", default="", help="Run directory. Defaults to latest under --run-root.")
    parser.add_argument("--run-root", default="runs/imagenet_likelihood_sparse_soft_5k")
    parser.add_argument("--include", default="primary,secondary,variants,panel")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path. Defaults under the run directory.")
    parser.add_argument("--audit-json", default="", help="Optional audit JSON path. Defaults under the run directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _latest_run(args.run_root)
    if not (run_dir / "comparison_manifest.json").is_file():
        raise FileNotFoundError(f"comparison_manifest.json missing from {run_dir}")

    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else run_dir / "comparison_summary.json"
    audit_json = Path(args.audit_json).expanduser().resolve() if args.audit_json else run_dir / "completion_audit.json"

    eval_cmd = [
        ".venv/bin/python",
        "scripts/eval_imagenet_likelihood_manifest.py",
        str(run_dir),
        "--include",
        str(args.include),
    ]
    if args.force_eval:
        eval_cmd.append("--force")
    _run(eval_cmd, dry_run=bool(args.dry_run))

    _run(
        [
            ".venv/bin/python",
            "scripts/summarize_imagenet_likelihood_results.py",
            str(run_dir),
            "--json-out",
            str(summary_json),
        ],
        dry_run=bool(args.dry_run),
    )
    _run(
        [
            ".venv/bin/python",
            "scripts/audit_imagenet_likelihood_completion.py",
            str(run_dir),
            "--json-out",
            str(audit_json),
        ],
        dry_run=bool(args.dry_run),
    )

    if not args.dry_run and audit_json.is_file():
        audit = json.loads(audit_json.read_text(encoding="utf-8"))
        if not audit.get("ok", False):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
