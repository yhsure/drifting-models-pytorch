from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from absl import logging as absl_logging
from PIL import Image


def is_rank_zero() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log_for_0(msg, *args, **kwargs):
    if is_rank_zero():
        absl_logging.info(msg, *args, **kwargs)


def log_for_all(msg):
    if dist.is_available() and dist.is_initialized():
        absl_logging.info("[Rank %s] %s", dist.get_rank(), msg)
    else:
        absl_logging.info("[Rank 0] %s", msg)


class WandbLogger:
    _PLACEHOLDER_VALUES = {"", "YOUR_WANDB_PROJECT", "YOUR_WANDB_ENTITY", "none", "null"}

    def __init__(self) -> None:
        self.step = 0
        self.use_wandb = True
        self.log_every_k = 1
        self._buffer: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self.offline_dir = Path("log")
        self.workdir: Path | None = None
        self._wandb = None
        self._run = None
        self._run_id = None
        self._run_name = None

    @staticmethod
    def _to_plain(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): WandbLogger._to_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [WandbLogger._to_plain(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    @staticmethod
    def _infer_wandb_mode(requested_mode: str | None) -> str:
        if requested_mode:
            return str(requested_mode).lower()
        return os.environ.get("WANDB_MODE", "offline").lower()

    @classmethod
    def _resolve_wandb_name(cls, value: str | None, env_key: str, default: str | None = None) -> str | None:
        if value is not None and str(value).strip() not in cls._PLACEHOLDER_VALUES:
            return str(value).strip()
        env_value = os.environ.get(env_key)
        if env_value and env_value.strip() not in cls._PLACEHOLDER_VALUES:
            return env_value.strip()
        return default

    @staticmethod
    def _default_wandb_parent(workdir_path: Path | None) -> Path:
        env_dir = os.environ.get("WANDB_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        if workdir_path is not None:
            return workdir_path.resolve()
        return Path.cwd().resolve()

    @staticmethod
    def _env_snapshot() -> dict[str, str]:
        keys = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_NNODES",
            "SLURM_NODELIST",
            "MASTER_ADDR",
            "MASTER_PORT",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "RANK",
            "WANDB_MODE",
            "WANDB_DIR",
            "WANDB_CACHE_DIR",
            "WANDB_CONFIG_DIR",
        ]
        return {k: os.environ[k] for k in keys if os.environ.get(k)}

    def _write_run_metadata(
        self,
        *,
        project: str | None,
        entity: str | None,
        name: str | None,
        config: Any | None,
        use_wandb: bool,
        mode: str,
        wandb_parent: Path,
    ) -> None:
        if not is_rank_zero():
            return
        plain_config = self._to_plain(config or {})
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": self._run_id,
            "name": name,
            "project": project,
            "entity": entity,
            "use_wandb": use_wandb,
            "wandb_mode": mode,
            "wandb_parent": str(wandb_parent),
            "wandb_watch_dir": str(wandb_parent / "wandb"),
            "workdir": str(self.workdir) if self.workdir is not None else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "env": self._env_snapshot(),
        }
        with (self.offline_dir / "run.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        with (self.offline_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(plain_config, f, sort_keys=False)

    @staticmethod
    def read_run_metadata(workdir: str | Path) -> dict[str, Any]:
        path = Path(workdir).expanduser().resolve() / "log" / "run.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def set_logging(
        self,
        project: str | None = None,
        config: Any | None = None,
        entity: str | None = None,
        name: str | None = None,
        use_wandb: bool = True,
        offline_dir: str = "log",
        workdir: str | None = None,
        log_every_k: int = 1,
        allow_resume: bool = True,
        mode: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        **kwargs,
    ) -> None:
        self.use_wandb = bool(use_wandb)
        self.log_every_k = int(log_every_k)
        project = self._resolve_wandb_name(project, "WANDB_PROJECT", default="drift")
        entity = self._resolve_wandb_name(entity, "WANDB_ENTITY", default="ucph-dk")
        workdir_path = Path(workdir).resolve() if workdir else None
        self.workdir = workdir_path
        resolved_offline_dir = workdir_path / "log" if workdir_path is not None else Path(offline_dir).resolve()
        self.offline_dir = resolved_offline_dir
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        wandb_mode = self._infer_wandb_mode(mode)
        wandb_parent = self._default_wandb_parent(workdir_path)
        wandb_parent.mkdir(parents=True, exist_ok=True)

        if not is_rank_zero():
            return

        default_run_id = ""
        if workdir_path is not None:
            default_run_id = hashlib.sha1(str(workdir_path).encode("utf-8")).hexdigest()[:16]
        self._run_id = kwargs.pop("run_id", None) or kwargs.pop("id", None) or default_run_id
        self._run_name = name
        self._write_run_metadata(
            project=project,
            entity=entity,
            name=name,
            config=config,
            use_wandb=self.use_wandb,
            mode=wandb_mode,
            wandb_parent=wandb_parent,
        )

        if self.use_wandb:
            import wandb

            self._wandb = wandb
            init_kwargs = dict(
                project=project,
                entity=entity,
                name=name,
                config=self._to_plain(config),
                mode=wandb_mode,
                dir=str(wandb_parent),
                reinit=True,
                tags=tags,
                notes=notes,
            )
            if allow_resume:
                init_kwargs["resume"] = "allow"
                if self._run_id:
                    init_kwargs["id"] = self._run_id
            init_kwargs.update(kwargs)
            self._run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
            wandb.define_metric("samples/*", step_metric="samples/step")
            wandb.define_metric("eval/*", step_metric="eval/step")

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        reduced = {k: (self._buffer[k] / max(1, self._count.get(k, 1))) for k in self._buffer.keys()}
        if self._wandb is not None:
            self._wandb.log(reduced, step=self.step)
        p = self.offline_dir / "metrics.jsonl"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": self.step, **reduced}, ensure_ascii=False) + "\n")
        self._buffer.clear()
        self._count.clear()

    def log_dict(self, d: dict[str, Any]) -> None:
        if not is_rank_zero():
            return
        reduced = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                v = float(np.asarray(v).mean())
            if isinstance(v, (int, float, np.floating, np.integer)):
                reduced[k] = float(v)
        for k, v in reduced.items():
            self._buffer[k] = self._buffer.get(k, 0.0) + float(v)
            self._count[k] = self._count.get(k, 0) + 1
        if self.log_every_k <= 1 or (self.step % self.log_every_k == 0):
            self._flush_buffer()

    def log_dict_dir(self, prefix: str, d: dict[str, Any]) -> None:
        self.log_dict({f"{prefix}/{k}": v for k, v in d.items()})

    @staticmethod
    def _normalize_images(images) -> np.ndarray:
        if torch.is_tensor(images):
            images = images.detach().cpu()
            if images.dtype in (torch.bfloat16, torch.float16):
                images = images.float()
        arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Expected image batch with 3 or 4 dims, got shape {arr.shape}")
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected channel-last image batch with 3 channels, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        return arr

    @staticmethod
    def _make_grid_image(images: np.ndarray, rows: int | None = None, cols: int | None = None) -> Image.Image:
        if len(images) == 0:
            raise ValueError("Expected at least one image for grid logging")
        if cols is None:
            if rows is None:
                cols = int(math.ceil(math.sqrt(len(images))))
            else:
                cols = int(math.ceil(len(images) / max(1, int(rows))))
        cols = max(1, int(cols))
        rows = max(1, int(math.ceil(len(images) / cols)))
        pil_imgs = [Image.fromarray(img) for img in images]
        w, h = pil_imgs[0].size
        grid = Image.new("RGB", (cols * w, rows * h))
        for idx, img in enumerate(pil_imgs):
            row = idx // cols
            col = idx % cols
            grid.paste(img, (col * w, row * h))
        return grid

    def log_image(
        self,
        name: str,
        images,
        *,
        max_images: int = 36,
        grid_rows: int = 6,
        grid_cols: int | None = None,
        log_individual: bool = False,
    ) -> None:
        if not is_rank_zero():
            return
        arr = self._normalize_images(images)[:max(1, int(max_images))]
        grid_img = self._make_grid_image(arr, rows=grid_rows, cols=grid_cols)
        out = self.offline_dir / "images"
        out.mkdir(parents=True, exist_ok=True)
        grid_img.save(out / f"{name.replace('/', '_')}_step{self.step}.jpg", format="JPEG")
        if self._wandb is not None:
            payload = {f"{name}_grid": self._wandb.Image(grid_img)}
            if log_individual:
                payload[name] = [self._wandb.Image(img) for img in arr]
            self._wandb.log(payload, step=self.step)

    def finish(self) -> None:
        self._flush_buffer()
        if self._wandb is not None and is_rank_zero():
            self._wandb.finish()


class NullLogger:
    @staticmethod
    def log_dict(*args, **kwargs):
        return None

    @staticmethod
    def log_image(*args, **kwargs):
        return None

    @staticmethod
    def finish(*args, **kwargs):
        return None
