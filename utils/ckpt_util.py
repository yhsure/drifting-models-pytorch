from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from utils.logging import log_for_0


def _is_rank_zero() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _to_python_int(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().reshape(-1)[0].item())
    return int(x)


def _output_root(workdir: Optional[str] = None) -> Path:
    if workdir:
        return Path(workdir).resolve()
    return Path("runs").resolve()


def _job_ckpt_dir(workdir: Optional[str] = None) -> Path:
    return _output_root(workdir) / "checkpoints"


def _ckpt_name(step: int) -> str:
    return f"step_{int(step):09d}.pt"


def _latest_checkpoint_path(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        return None
    return ckpts[-1]


def restore_checkpoint(step=None, state=None, workdir: Optional[str] = None):
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    if not ckpt_dir.exists():
        log_for_0("No local checkpoint dir at %s", str(ckpt_dir))
        return state

    ckpt_path = ckpt_dir / _ckpt_name(int(step)) if step is not None else _latest_checkpoint_path(ckpt_dir)
    if ckpt_path is None or not ckpt_path.exists():
        return state

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if state is None:
        return payload

    state.model.load_state_dict(payload["model"])
    ema_payload = payload.get("ema_model", payload.get("ema_params"))
    if hasattr(state, "ema_model") and state.ema_model is not None and ema_payload is not None:
        state.ema_model.load_state_dict(ema_payload)
    if hasattr(state, "ema_params") and ema_payload is not None:
        state.ema_params = {k: v.clone() for k, v in ema_payload.items()}
    if state.optimizer is not None and "optimizer" in payload:
        state.optimizer.load_state_dict(payload["optimizer"])
    state.step = int(payload.get("step", 0))
    state.ema_decay = float(payload.get("ema_decay", state.ema_decay))
    return state


def save_checkpoint(state, keep=2, keep_every=None, workdir: Optional[str] = None):
    del keep_every
    if not _is_rank_zero():
        return
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(state, "ema_model") and state.ema_model is not None:
        ema_payload = state.ema_model.state_dict()
    elif hasattr(state, "ema_params"):
        ema_payload = state.ema_params
    else:
        ema_payload = None

    payload = {
        "step": _to_python_int(state.step),
        "model": _unwrap_model(state.model).state_dict(),
        "ema_model": ema_payload,
        "ema_params": ema_payload,
        "optimizer": state.optimizer.state_dict() if state.optimizer is not None else None,
        "ema_decay": float(state.ema_decay),
    }
    out = ckpt_dir / _ckpt_name(_to_python_int(state.step))
    torch.save(payload, out)
    log_for_0("Saving checkpoint step %d to %s", _to_python_int(state.step), str(out))

    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if keep is not None and keep > 0 and len(ckpts) > keep:
        for p in ckpts[: len(ckpts) - keep]:
            p.unlink(missing_ok=True)


def save_params_ema_artifact(
    state: Any,
    *,
    workdir: Optional[str] = None,
    kind: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> Path:
    if not _is_rank_zero():
        return _output_root(workdir) / "params_ema"
    step = _to_python_int(state.step)
    ema_decay = float(getattr(state, "ema_decay"))

    out_dir = _output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "ema_params.pt"
    if hasattr(state, "ema_model") and state.ema_model is not None:
        ema_payload = state.ema_model.state_dict()
    elif hasattr(state, "ema_params"):
        ema_payload = state.ema_params
    else:
        ema_payload = {}
    torch.save(ema_payload, params_path)

    metadata = {
        "format": "torch.state_dict",
        "kind": kind,
        "backend": "torch",
        "ema_decay": ema_decay,
        "step": step,
        "path": "params_ema/ema_params.pt",
        "model_config": dict(model_config or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log_for_0("Saved EMA params artifact step %d to %s", step, str(out_dir))
    return out_dir
