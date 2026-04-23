# Slurm Scripts

`common.sh` contains shared setup used by all jobs (module load, `uv` cache, torch cache, and `uv sync`).

Submit from the repository root: Slurm copies the batch script to spool, so the job uses `SLURM_SUBMIT_DIR` as the repo path. Scripts use `--exclusive`, `--cpus-per-task=288`, and no `--gres`.

Training runs under `torchrun` (`--standalone` on one node; multi-node uses c10d on the first host). `drift_nproc_per_node` follows `torch.cuda.device_count()` and falls back to 4 if the batch shell does not see CUDA.

Examples:

```bash
sbatch scripts/slurm/mae_login_smoke.sbatch
sbatch scripts/slurm/gen_login_smoke.sbatch
sbatch scripts/slurm/mae_2node_smoke.sbatch
sbatch scripts/slurm/gen_2node_smoke.sbatch
```

If `--exclusive` is rejected by your partition, add the site-specific GPU line your center documents.
