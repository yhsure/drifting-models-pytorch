from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from models.hf import load_torch_ema_params, read_metadata
from utils.env import HF_ROOT


def resolve_artifact_dir(path: str) -> Path:
    base = Path(path).resolve()
    if base.is_file():
        return base
    params_ema_dir = base / "params_ema"
    ckpt_dir = base / "checkpoints"
    if params_ema_dir.is_dir():
        return params_ema_dir
    if ckpt_dir.is_dir():
        return ckpt_dir
    return base


def _strip_state_dict_wrappers(params: Any) -> Any:
    if not isinstance(params, dict):
        return params
    prefixes = ("module._orig_mod.", "_orig_mod.", "module.")
    for prefix in prefixes:
        if params and all(isinstance(k, str) and k.startswith(prefix) for k in params):
            return {k.removeprefix(prefix): v for k, v in params.items()}
    return params


def _read_checkpoint_metadata(checkpoint_path: Path, restored: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if isinstance(restored, dict) and isinstance(restored.get("metadata"), dict):
        metadata.update(restored["metadata"])

    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    for candidate in (
        run_dir / "params_ema" / "metadata.json",
        run_dir / "metadata.json",
        checkpoint_path.with_suffix(".metadata.json"),
    ):
        if candidate.is_file():
            metadata.update(json.loads(candidate.read_text(encoding="utf-8")))
            break

    if isinstance(restored, dict) and "step" in restored:
        metadata["step"] = int(restored["step"])
    metadata.setdefault("format", "torch.checkpoint")
    return metadata


def _load_checkpoint_file(checkpoint_path: Path) -> Tuple[Any, Dict[str, Any]]:
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(restored, dict):
        params = restored.get("ema_model", restored.get("ema_params", restored.get("model", restored)))
    else:
        params = restored
    return _strip_state_dict_wrappers(params), _read_checkpoint_metadata(checkpoint_path, restored)


def _load_local_init_entry(path: str) -> Tuple[Any, Dict[str, Any]]:
    artifact_dir = resolve_artifact_dir(path)
    if artifact_dir.is_file():
        return _load_checkpoint_file(artifact_dir)

    metadata_path = artifact_dir / "metadata.json"
    params_path = artifact_dir / "ema_params.pt"
    legacy_meta_path = artifact_dir / "ema_model.metadata.json"
    legacy_params_path = artifact_dir / "ema_model.pt"

    if metadata_path.is_file() and params_path.is_file():
        return load_torch_ema_params(artifact_dir), read_metadata(artifact_dir)
    if params_path.is_file():
        return load_torch_ema_params(artifact_dir), {}
    if legacy_meta_path.is_file() and legacy_params_path.is_file():
        metadata = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
        params = torch.load(legacy_params_path, map_location="cpu", weights_only=False)
        return params, metadata
    if legacy_params_path.is_file():
        params = torch.load(legacy_params_path, map_location="cpu", weights_only=False)
        return params, {}

    ckpts = sorted(artifact_dir.glob("step_*.pt"))
    if ckpts:
        return _load_checkpoint_file(ckpts[-1])

    raise ValueError(
        "Local init_from must be an artifact or checkpoint dir with params: "
        f"{artifact_dir}"
    )


def load_init_entry(
    model_type: str,
    init_from: str,
    *,
    hf_cache_dir: str = HF_ROOT,
) -> Tuple[Any, Dict[str, Any]]:
    if not init_from:
        raise ValueError("`init_from` is empty.")

    if not init_from.startswith("hf://"):
        return _load_local_init_entry(init_from)

    model_name = init_from[len("hf://") :].strip()
    if not model_name:
        raise ValueError("Invalid HF init_from path, expected `hf://<name>`.")

    if model_type == "mae":
        from models.mae_model import load_mae_hf

        _, params, metadata = load_mae_hf(model_name, dir=hf_cache_dir)
        return params, metadata

    if model_type == "generator":
        from models.generator import load_hf

        _, params, metadata = load_hf(model_name, dir=hf_cache_dir)
        return params, metadata

    raise ValueError(f"Unsupported model_type={model_type!r}, expected 'mae' or 'generator'.")


def maybe_init_state_params(
    state: Any,
    *,
    model_type: str,
    init_from: str,
    hf_cache_dir: str = HF_ROOT,
) -> Any:
    if not init_from:
        return state

    loaded_params, _ = load_init_entry(model_type, init_from, hf_cache_dir=hf_cache_dir)
    state.model.load_state_dict(loaded_params, strict=False)
    if hasattr(state, "ema_model") and state.ema_model is not None:
        state.ema_model.load_state_dict(loaded_params, strict=False)
    if hasattr(state, "ema_params"):
        state.ema_params = {k: v.detach().clone() for k, v in state.model.state_dict().items()}
    return state


def load_generator_model_and_params(
    init_from: str,
    *,
    hf_cache_dir: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if not init_from:
        raise ValueError("`init_from` is empty.")

    if init_from.startswith("hf://"):
        from models.generator import load_hf

        model_name = init_from[len("hf://") :].strip()
        model, params, metadata = load_hf(model_name, dir=hf_cache_dir)
        return model, params, metadata

    params, metadata = _load_local_init_entry(init_from)
    model_cfg = dict(metadata.get("model_config", {}) or {})
    if not model_cfg:
        raise ValueError(
            f"missing metadata.model_config: local artifact at {Path(init_from).resolve()} "
            "cannot be restored without model_config in metadata.json"
        )
    from models.generator import build_generator_from_config

    model = build_generator_from_config(model_cfg)
    return model, params, metadata
