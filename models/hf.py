"""Minimal Hugging Face helpers for Drift artifacts (PyTorch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
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
    nested = local_root / path_in_repo
    if nested.exists() and any(nested.iterdir()):
        return nested
    if (local_root / "metadata.json").is_file():
        return local_root

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=str(local_root),
    )
    return nested if nested.exists() else local_root


def _decode_flax_ndarray(payload: bytes) -> np.ndarray:
    import msgpack

    shape, dtype_name, data = msgpack.unpackb(payload, raw=False)
    arr = np.frombuffer(data, dtype=np.dtype(dtype_name))
    return arr.reshape(tuple(shape))


def _load_flax_msgpack(path: Path) -> Dict[str, Any]:
    import msgpack

    def ext_hook(code: int, payload: bytes):
        if code == 1:
            return _decode_flax_ndarray(payload)
        return msgpack.ExtType(code, payload)

    return msgpack.unpackb(path.read_bytes(), raw=False, ext_hook=ext_hook, strict_map_key=False)


def _torch_tensor(arr: np.ndarray, *, conv: bool = False, linear: bool = False) -> torch.Tensor:
    if conv:
        arr = np.transpose(arr, (3, 2, 0, 1))
    elif linear:
        arr = np.transpose(arr, (1, 0))
    return torch.from_numpy(np.array(arr, copy=True, order="C"))


def _copy_norm(out: Dict[str, torch.Tensor], prefix: str, src: Dict[str, np.ndarray]) -> None:
    out[f"{prefix}.weight"] = _torch_tensor(src["scale"])
    out[f"{prefix}.bias"] = _torch_tensor(src["bias"])


def _copy_conv(out: Dict[str, torch.Tensor], name: str, src: Dict[str, np.ndarray]) -> None:
    out[f"{name}.weight"] = _torch_tensor(src["kernel"], conv=True)


def _convert_mae_jax_params_to_torch(params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    enc = params["encoder"]
    dec = params["decoder"]

    _copy_conv(out, "encoder.conv1", enc["conv1"])
    _copy_norm(out, "encoder.gn1", enc["gn1"])

    for stage_idx in range(4):
        stage = enc[f"stages_{stage_idx}"]
        for block_name, block in stage.items():
            block_idx = int(block_name.split("_")[-1])
            prefix = f"encoder.stages.{stage_idx}.{block_idx}"
            _copy_conv(out, f"{prefix}.conv1", block["conv1"])
            _copy_norm(out, f"{prefix}.gn1", block["gn1"])
            _copy_conv(out, f"{prefix}.conv2", block["conv2"])
            _copy_norm(out, f"{prefix}.gn2", block["gn2"])
            if "proj_conv" in block:
                _copy_conv(out, f"{prefix}.proj_conv", block["proj_conv"])
                _copy_norm(out, f"{prefix}.proj_gn", block["proj_gn"])
        _copy_norm(out, f"encoder.stage_norms.{stage_idx}", enc[f"layer{stage_idx + 1}_norm"])

    _copy_conv(out, "decoder.bridge.conv", dec["bridge"]["conv"])
    _copy_norm(out, "decoder.bridge.gn", dec["bridge"]["gn"])

    for name in ("up43", "up32", "up21", "up10"):
        block = dec[name]
        prefix = f"decoder.{name}"
        _copy_norm(out, f"{prefix}.concat_norm_fn", block["concat_norm_fn"])
        _copy_conv(out, f"{prefix}.proj.0", block["proj"]["conv"])
        _copy_norm(out, f"{prefix}.proj.1", block["proj"]["gn"])
        _copy_conv(out, f"{prefix}.refine.conv", block["refine"]["conv"])
        _copy_norm(out, f"{prefix}.refine.gn", block["refine"]["gn"])

    _copy_conv(out, "decoder.head", dec["head"])
    out["decoder.head.bias"] = _torch_tensor(dec["head"]["bias"])
    out["fc.weight"] = _torch_tensor(params["fc"]["kernel"], linear=True)
    out["fc.bias"] = _torch_tensor(params["fc"]["bias"])
    return out


def _materialize_mae_torch_from_jax(
    *,
    name: str,
    repo_id: str,
    prefix: Optional[str],
    output_root: str,
) -> Path:
    artifact_dir = _download_artifact(
        repo_id=repo_id,
        kind="mae",
        backend="jax",
        model_id=name,
        output_root=output_root,
        prefix=prefix,
    )
    metadata = read_metadata(artifact_dir)
    params = _load_flax_msgpack(artifact_dir / "ema_params.msgpack")
    torch_params = _convert_mae_jax_params_to_torch(params)

    out_dir = Path(output_root).resolve() / "models" / "mae" / "pytorch" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata["backend"] = "pytorch"
    metadata["converted_from_backend"] = "jax"
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    torch.save(torch_params, out_dir / "ema_params.pt")
    return out_dir


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
    if not (artifact_dir / "metadata.json").is_file() or not (artifact_dir / "ema_params.pt").is_file():
        artifact_dir = _materialize_mae_torch_from_jax(
            name=name,
            repo_id=repo_id,
            prefix=prefix,
            output_root=output_root,
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
