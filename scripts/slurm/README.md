# Slurm Scripts

`common.sh` contains shared setup used by all jobs (module load, `uv sync`, workspace-root caches: `uv`, torch inductor, `TORCH_HOME`, `HF_ROOT`, offline WandB, and JSC network env: `MASTER_ADDR`, `MASTER_PORT`, `NCCL_SOCKET_IFNAME=ib0`, `GLOO_SOCKET_IFNAME=ib0`).

Submit from the repository root: Slurm copies the batch script to spool, so the job uses `SLURM_SUBMIT_DIR` as the repo path. Scripts use `--exclusive`, `--cpus-per-task=288`, and no `--gres`.

Training runs under `srun env -u CUDA_VISIBLE_DEVICES .venv/bin/python -u -m torchrun_jsc` (`--standalone` on one node; multi-node uses c10d on the first host). `drift_nproc_per_node` follows `torch.cuda.device_count()` and falls back to 4 if the batch shell does not see CUDA.

Examples:

```bash
sbatch scripts/slurm/mae_1node_smoke.sbatch
sbatch scripts/slurm/gen_1node_smoke.sbatch
sbatch scripts/slurm/mae_2node_smoke.sbatch
sbatch scripts/slurm/gen_2node_smoke.sbatch
```

If `--exclusive` is rejected by your partition, add the site-specific GPU line your center documents.

## Offline WandB sync

Compute nodes do not have internet access, so configs that set `logging.use_wandb: true` should keep `logging.mode: offline` or rely on `common.sh`'s default `WANDB_MODE=offline`. Offline WandB folders are written inside each stamped run directory, next to checkpoints and local `log/` files.

Run sync from a login node:

```bash
cd /e/project1/e-dev-2026d02-064/_abj/torch-port/drifting-models-pytorch
uv sync --group dev
.venv/bin/wandb login
tmux new -s drift_wandb_sync './scripts/wandb_sync_offline.sh'
```

Stop the sync loop with `tmux kill-session -t drift_wandb_sync` on the same login node where it was started.

The helper recursively finds `runs/**/wandb/offline-run-*`. If using the JSC AI recipe daemon directly, monitor a specific run's `wandb/` folder or adapt the daemon to scan recursively.
