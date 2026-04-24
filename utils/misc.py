from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml


class EasyDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value):
        self[name] = value


def _dict_to_easydict(d):
    if not isinstance(d, dict):
        return d
    out = EasyDict()
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _dict_to_easydict(v)
        elif isinstance(v, list):
            out[k] = [_dict_to_easydict(i) for i in v]
        else:
            out[k] = v
    return out


_IGNORED_LEGACY_MODEL_KEYS = frozenset({"use_bf16", "attn_fp32"})
_IGNORED_LEGACY_TRAIN_KEYS = frozenset({"keep_every"})


def _sanitize_legacy_section(section: Mapping[str, Any] | None, ignored_keys: frozenset[str]) -> EasyDict:
    clean = dict(section or {})
    for key in ignored_keys:
        clean.pop(key, None)
    return _dict_to_easydict(clean)


def sanitize_model_config(model_config: Mapping[str, Any] | None) -> EasyDict:
    return _sanitize_legacy_section(model_config, _IGNORED_LEGACY_MODEL_KEYS)


def sanitize_train_config(train_config: Mapping[str, Any] | None) -> EasyDict:
    return _sanitize_legacy_section(train_config, _IGNORED_LEGACY_TRAIN_KEYS)


def sanitize_runtime_config(config):
    if not isinstance(config, dict):
        return config
    if isinstance(config.get("model"), dict):
        config["model"] = sanitize_model_config(config["model"])
    if isinstance(config.get("train"), dict):
        config["train"] = sanitize_train_config(config["train"])
    return config


def maybe_compile(obj, level: int):
    """Wrap obj with torch.compile at the requested depth.

    0 — no compilation
    1 — torch.compile with default settings
    2 — torch.compile(dynamic=False, fullgraph=True)
    """
    if level == 0:
        return obj
    if level == 1:
        return torch.compile(obj)
    return torch.compile(obj, dynamic=False, fullgraph=True)


def stamp_workdir(workdir: str) -> str:
    p = Path(workdir).expanduser()
    ts = time.strftime("%m%d_%H%M", time.localtime())
    if len(p.parts) == 1 and p.name == "runs":
        return str(p / ts)
    return str(p.parent / f"{ts}_{p.name}")


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = _dict_to_easydict(yaml.safe_load(f))
    return sanitize_runtime_config(config)


def prepare_rng(rng_key: int | torch.Generator, tags=("params", "dropout")):
    if isinstance(rng_key, torch.Generator):
        base_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng_key).item())
    else:
        base_seed = int(rng_key)
    out = {}
    for idx, tag in enumerate(tags):
        g = torch.Generator()
        g.manual_seed(base_seed + idx)
        out[tag] = g
    return out


_did_run_init = False


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def run_init() -> None:
    global _did_run_init
    if _did_run_init:
        return

    default_cache_root = Path.cwd() / ".torch-cache"
    default_cache_root.mkdir(parents=True, exist_ok=True)
    desired_inductor = default_cache_root / "torchinductor"
    desired_home = default_cache_root / "torch"
    desired_inductor.mkdir(parents=True, exist_ok=True)
    desired_home.mkdir(parents=True, exist_ok=True)

    current_inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    if not current_inductor or not os.access(Path(current_inductor).parent if current_inductor else default_cache_root, os.W_OK):
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(desired_inductor)
    current_torch_home = os.environ.get("TORCH_HOME", "")
    if not current_torch_home or not os.access(Path(current_torch_home).parent if current_torch_home else default_cache_root, os.W_OK):
        os.environ["TORCH_HOME"] = str(desired_home)

    seed = int(os.environ.get("GLOBAL_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.cuda.manual_seed_all(seed)

    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1 and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    _did_run_init = True


_jitted_rand = {}


def ddp_rand_func(rand_type="normal", shard="ddp"):
    del shard
    key = (rand_type,)
    if key in _jitted_rand:
        return _jitted_rand[key]

    def _normal(generator: torch.Generator, shape, *, device=None, dtype=torch.float32):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def _uniform(generator: torch.Generator, shape, *, device=None, dtype=torch.float32):
        return torch.rand(shape, generator=generator, device=device, dtype=dtype)

    if rand_type == "normal":
        _jitted_rand[key] = _normal
    elif rand_type == "uniform":
        _jitted_rand[key] = _uniform
    else:
        raise ValueError(rand_type)
    return _jitted_rand[key]


def _profile_log(report: list[str], msg: str, *, console_print: bool) -> None:
    report.append(msg)
    if console_print and _rank() == 0:
        print(msg, flush=True)


def _format_metric_value(value: float, suffix: str = "") -> str:
    for unit in ("", "K", "M", "G", "T", "P"):
        if abs(value) < 1000.0:
            return f"{value:3.2f} {unit}{suffix}".rstrip()
        value /= 1000.0
    return f"{value:.2f} E{suffix}".rstrip()


def _normalize_cost_analysis(cost_analysis):
    if isinstance(cost_analysis, list):
        return dict(cost_analysis[0] or {})
    return dict(cost_analysis or {})


def _extract_memory_metrics(compiled):
    del compiled
    return {
        "profile/Memory_GB": 0.0,
        "profile/Weights_MB": 0.0,
        "profile/Activations_MB": 0.0,
        "profile/Output_MB": 0.0,
    }


def profile_func(
    target_fn: Callable,
    args: tuple,
    kwargs: Optional[Dict] = None,
    name: str = "Model",
    console_print: bool = True,
    hardware_peak_bw: float = 1600.0,
    actual_run: bool = False,
    n_loops: int = 10,
    print_hlo: bool = False,
):
    del hardware_peak_bw, print_hlo
    kwargs = kwargs or {}
    report: list[str] = []
    metrics: Dict[str, float] = {}
    _profile_log(report, f"[Profile] Inspecting '{name}'", console_print=console_print)

    if not actual_run:
        return metrics

    with torch.inference_mode():
        _ = target_fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_loops):
            _ = target_fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / max(1, n_loops)
    metrics["profile/Time_ms"] = float(dt * 1000.0)
    _profile_log(report, f"[Profile] Runtime: time={metrics['profile/Time_ms']:.2f} ms", console_print=console_print)
    return metrics
