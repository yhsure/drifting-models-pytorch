import json

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
