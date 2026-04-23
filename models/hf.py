"""Minimal Hugging Face helpers for Drift artifacts (PyTorch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from utils.env import HF_ROOT


def read_metadata(artifact_dir: Path) -> Dict[str, Any]:
    return json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))


def load_torch_ema_params(artifact_dir: Path) -> Any:
    return torch.load((artifact_dir / "ema_params.pt"), map_location="cpu", weights_only=False)


def load_jax_ema_params(artifact_dir: Path) -> Any:
    """Compatibility alias for old import paths."""

    return load_torch_ema_params(artifact_dir)


def _download_artifact(
    *,
    repo_id: str,
    kind: str,
    backend: str,
    model_id: str,
    output_root: str,
    prefix: Optional[str],
) -> Path:
    from huggingface_hub import snapshot_download

    local_root = Path(output_root).resolve() / "models" / kind / backend / model_id
    local_root.mkdir(parents=True, exist_ok=True)
    root = f"models/{kind}/{backend}/{model_id}"
    path_in_repo = f"{prefix.strip('/')}/{root}" if prefix else root

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=str(local_root),
    )
    nested = local_root / path_in_repo
    return nested if nested.exists() else local_root


def load_mae_torch(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    artifact_dir = _download_artifact(
        repo_id=repo_id,
        kind="mae",
        backend="pytorch",
        model_id=name,
        output_root=output_root,
        prefix=prefix,
    )
    metadata = read_metadata(artifact_dir)

    from models.mae_model import _mae_from_metadata

    module = _mae_from_metadata(metadata)
    params = load_torch_ema_params(artifact_dir)
    return module, params, metadata


def load_mae_jax(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Compatibility alias for old import paths."""

    return load_mae_torch(name=name, repo_id=repo_id, prefix=prefix, output_root=output_root)


def load_generator_torch(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    artifact_dir = _download_artifact(
        repo_id=repo_id,
        kind="gen",
        backend="pytorch",
        model_id=name,
        output_root=output_root,
        prefix=prefix,
    )
    metadata = read_metadata(artifact_dir)

    model_cfg = dict(metadata.get("model_config", {}) or {})
    if not model_cfg:
        raise ValueError(
            f"Generator artifact is missing metadata.model_config and cannot be restored: {name}"
        )

    from models.generator import build_generator_from_config

    module = build_generator_from_config(model_cfg)
    params = load_torch_ema_params(artifact_dir)
    return module, params, metadata


def load_generator_jax(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Compatibility alias for old import paths."""

    return load_generator_torch(name=name, repo_id=repo_id, prefix=prefix, output_root=output_root)
