import json

import numpy as np
import torch

from inference import _infer_eval_step
from utils.logging import WandbLogger


def test_local_logger_writes_metadata_and_buffered_metrics(tmp_path):
    workdir = tmp_path / "run"
    logger = WandbLogger()
    logger.set_logging(
        project="proj",
        entity="ent",
        name="unit-test",
        config={"train": {"steps": 2}},
        use_wandb=False,
        workdir=str(workdir),
        log_every_k=2,
    )

    logger.set_step(1)
    logger.log_dict({"loss": 3.0})
    assert not (workdir / "log" / "metrics.jsonl").exists()

    logger.set_step(2)
    logger.log_dict({"loss": 1.0, "ignored": "not-a-number"})
    logger.finish()

    metadata = json.loads((workdir / "log" / "run.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "unit-test"
    assert metadata["project"] == "proj"
    assert metadata["use_wandb"] is False
    assert metadata["wandb_mode"] == "offline"
    assert metadata["wandb_parent"] == str(workdir.resolve())
    assert metadata["wandb_watch_dir"] == str(workdir.resolve() / "wandb")
    assert (workdir / "log" / "config.yaml").read_text(encoding="utf-8").strip() == "train:\n  steps: 2"

    lines = (workdir / "log" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"step": 2, "loss": 2.0}


def test_read_run_metadata_and_infer_eval_step(tmp_path):
    workdir = tmp_path / "run"
    logger = WandbLogger()
    logger.set_logging(name="train", use_wandb=False, workdir=str(workdir), run_id="abc123")
    (workdir / "checkpoints").mkdir()
    (workdir / "checkpoints" / "step_000000400.pt").touch()
    (workdir / "checkpoints" / "step_000000800.pt").touch()

    assert WandbLogger.read_run_metadata(workdir)["run_id"] == "abc123"
    assert _infer_eval_step(str(workdir)) == 800


def test_logger_normalizes_bfloat16_torch_images():
    images = torch.rand(2, 3, 8, 8, dtype=torch.bfloat16)

    arr = WandbLogger._normalize_images(images)

    assert arr.shape == (2, 8, 8, 3)
    assert arr.dtype == np.uint8


def test_grid_image_uses_columns_without_padding():
    images = np.ones((16, 8, 8, 3), dtype=np.uint8)

    grid = WandbLogger._make_grid_image(images, cols=4)

    assert grid.size == (32, 32)
