from __future__ import annotations

import copy
import os
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.distributed as dist


global_mesh = None
axis_to_dim = {}


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def set_global_mesh(hsdp_dim: int = 8):
    global global_mesh
    ws = _world_size()
    hsdp_dim = min(max(1, hsdp_dim), ws)
    axis_to_dim["fsdp"] = hsdp_dim
    axis_to_dim["data"] = max(1, ws // hsdp_dim)
    global_mesh = {"world_size": ws, "hsdp_dim": hsdp_dim}


def get_global_mesh():
    if global_mesh is None:
        set_global_mesh(1)
    return global_mesh


def axis_dim(axis: str):
    return axis_to_dim.get(axis, 1)


def get_spec(path_name, tensor_leaf, axis_tuple=("fsdp",)):
    del path_name, tensor_leaf, axis_tuple
    return None


_STATIC_TYPES = (bool, int, float, str)


def split_static_dynamic(dummy_input: dict):
    dynamic_input = {}
    static_input = {}
    for k, v in dummy_input.items():
        if isinstance(v, _STATIC_TYPES):
            static_input[k] = v
        else:
            dynamic_input[k] = v
    return dynamic_input, static_input


def prepare_rngs(rng: int, all_keys: Sequence[str]):
    return {k: int(rng) + i for i, k in enumerate(all_keys)}


def ddp_shard():
    return None


def data_shard():
    return None


def enforce_ddp(x):
    return x


def init_state_from_dummy_input(
    model,
    optimizer,
    TrainState,
    rng,
    dummy_input: dict,
    rng_keys_extra: Sequence[str] = (),
    ema_decay=0.999,
    **_unused_kwargs,
):
    del rng, dummy_input, rng_keys_extra
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ema_model = copy.deepcopy(model).to(device)
    return TrainState(
        step=0,
        model=model,
        optimizer=optimizer,
        ema_model=ema_model,
        ema_decay=float(ema_decay),
        device=device,
    )


def map_to_sharding(params):
    del params

    def identity(x):
        return x

    return identity


def init_model_distributed(
    model,
    dummy_input: Dict[str, Any],
    rng: int | None = None,
    rng_keys_extra: Sequence[str] = (),
) -> Tuple[Any, Dict[str, int]]:
    del dummy_input, rng, rng_keys_extra
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model.state_dict(), get_global_mesh()


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def merge_data(data, use_ddp=False):
    del use_ddp
    device = _device()
    return _to_device(data, device)


def _pad_leaf(x: torch.Tensor, pad_len: int):
    if pad_len <= 0:
        return x
    pad_shape = (pad_len,) + tuple(x.shape[1:])
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


def pad_and_merge(data, local_bsz, use_ddp=False):
    del use_ddp
    if isinstance(data, dict):
        leaf = next(iter(data.values()))
    else:
        leaf = data[0]
    current_len = int(leaf.shape[0])
    pad_len = int(local_bsz) - current_len
    if pad_len < 0:
        raise ValueError(f"local_bsz={local_bsz} < current_len={current_len}")

    mask = torch.cat(
        [
            torch.ones(current_len, dtype=torch.float32, device=leaf.device),
            torch.zeros(pad_len, dtype=torch.float32, device=leaf.device),
        ],
        dim=0,
    )

    if isinstance(data, dict):
        padded = {k: _pad_leaf(v, pad_len) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        padded = type(data)(_pad_leaf(v, pad_len) if isinstance(v, torch.Tensor) else v for v in data)
    else:
        padded = _pad_leaf(data, pad_len)

    return padded, mask
