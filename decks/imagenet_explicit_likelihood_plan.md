# ImageNet Explicit Likelihood Prototype

This is a short-run plan for putting the explicit likelihood mixture prior
inside the ImageNet latent drifting setup. The run should be useful as a
5k/10k-step prototype. Since we already have 5k ImageNet results, the primary
comparison should be a matched 5k evaluation; 10k is an optional continuation
for checking slope.

The base should stay as close as possible to the previous 4-GPU ImageNet pilot
recipe: cached SDVAE latents, `32 x 32 x 4` latent generation, ImageNet class
conditioning, CFG training, latent MAE features, and the same ImageNet
validation FID stats. The new ingredient is only the explicit feature-space
likelihood prior used to condition the generator and choose local drift
supports.

## Model

For a cached latent `z` and class label `y`, compute a frozen MAE descriptor

$$
v(z)=\frac{\operatorname{pool}(\phi(z))-\mu_\phi}{\sigma_\phi}.
$$

Use a compact pooled descriptor for the likelihood prior. The full multiscale
MAE features should still be used for the drifting loss.

Fit a shared Gaussian mixture in descriptor space:

$$
p_\alpha(v)=\sum_{k=1}^K \pi_k \mathcal N(v;c_k,\sigma^2I).
$$

For the first prototype, use `K=8192` centers initialized by mini-batch k-means.
Use `K=4096` only if memory is tight. Since ImageNet generation is
class-conditional, keep the centers global but estimate class-conditional
mixture weights:

$$
\pi_{k\mid y}
\propto
\epsilon+
\frac{1}{|\mathcal D_y|}
\sum_{z_i\in\mathcal D_y}p(k\mid v(z_i)).
$$

For a training example, compute responsibilities with an annealed temperature:

$$
r_{ik}(\tau)
=
\operatorname{softmax}_k
\left(
\frac{\log\pi_{k\mid y_i}
-\lVert v_i-c_k\rVert^2/(2\sigma^2)}{\tau}
\right).
$$

Then keep the top `k_sparse` responsibilities and renormalize:

$$
a_i=\sum_{k\in \operatorname{TopK}(r_i,k_\text{sparse})}
\tilde r_{ik} e_k.
$$

The ImageNet generator remains one shared conditional model:

$$
\hat z=G_\theta(\epsilon,y,a_i,\gamma),
$$

where `gamma` is the sampled CFG scale already used in the drifting code. The
lowest-risk implementation is to project `a_i` to `cond_dim=768` and add it to
the existing conditioning stream, or to insert a few component prefix tokens.
This is not an ensemble of independent generators; it is a shared generator
given a likelihood-derived sparse semantic address.

## Training

Keep the original drifting objective as the image-quality signal. The mixture
does not reconstruct the frozen descriptors. It assigns each real latent to
feature-space components, supplies the generator condition, and defines the
sampling prior.

For each anchor latent, positives should come from the same ImageNet class and
from examples assigned to one of its active top-k mixture components, sampled in
proportion to `tilde r_i`. Keep a small uniform same-class fraction, around
`0.20`, so early imperfect components do not become brittle. Negatives should
use the existing unconditional/global queue used for CFG.

The training loss is

$$
\mathcal L
=
\mathcal L_\text{drift}
+\lambda_\text{nll}\mathcal L_\text{mix}/d
+\lambda_\text{bal}\mathcal L_\text{bal}
+\lambda_\text{anchor}\lVert C-C_0\rVert_2^2.
$$

`L_mix` is the Gaussian-mixture NLL of real descriptors. `L_bal` discourages
dead components globally and within class. The anchor term keeps centers near
their k-means initialization during the short run. Either freeze centers for
the first 2k steps or train them from the start with a small center LR scale,
around `0.1`.

## Short Run

The default prototype is sparse-soft:

```text
steps: 5000 first, optionally continue to 10000
train_batch_size: 16
dataset_batch_size: 256
eval_batch_size: 256
gen_per_label: 8
component_count: 8192
topk: 16 -> 8
responsibility_temperature: 1.5 -> 0.45 over 3500 steps for 5k
component_dropout: 0.10
uniform_same_class_positive_fraction: 0.20
nll_weight: 0.02 to 0.05 after per-dim normalization
balance_weight: 0.01 to 0.03
```

Do not sweep many variants. The useful comparison from one checkpoint is
sparse-soft top-8, hard top-1, and one alternate sparse-soft setting, probably
top-4 or top-16.

Training should run on Slurm. Evaluation should not be submitted as a Slurm job
for this prototype; run it from this login node after the checkpoint is written,
using the same FID hyperparameters as the reference run.

The intended Slurm-training command shape, after wiring config support, is:

```bash
python scripts/run_imagenet_likelihood_prior.py \
  --base-config configs/gen/latent_sota_B.yaml \
  --run-root runs/imagenet_likelihood_sparse_soft_5k \
  --total-steps 5000 \
  --component-count 8192 \
  --conditioning sparse_soft \
  --topk-start 16 --topk-final 8 \
  --temp-start 1.5 --temp-final 0.45 --temp-anneal-steps 3500 \
  --train-batch-size 16 --dataset-batch-size 256 \
  --eval-batch-size 256 --gen-per-label 8 \
  --save-per-step 2000 --eval-per-step 0 --eval-samples 0 \
  --fixed-cfg 1.0
```

Use timestamp-prepended run directories, for example
`runs/imagenet_likelihood_sparse_soft_5k/<ddmm_hhmm>_k8192_top8_tau045`.

## Comparability

A 5k run is directly useful because we already have 5k ImageNet references. A
10k continuation is not directly comparable to the 20k rows in `EXPERIMENTS.md`
unless we also train or evaluate a matched drifting baseline at 10k. For a
direct table comparison, use the same base config, same training step, same
4-GPU pilot overrides, same FID stats, same checkpoint averaging, same CFG
scale, and same number of generated samples.

The primary evaluation track is the existing Torch-port 5k latent ablation
setting: 50k samples, `cfg=2.5`, and the ImageNet validation stats. These are
the numbers to beat:

| reference | step | cfg | FID | IS |
|---|---:|---:|---:|---:|
| `0501_0019_latent_ablation_30k_mae640_407971` | 5k | 2.5 | 29.50 | 51.76 |
| `0501_0023_latent_ablation_30k_mae640_pos_enhanced_feat_407972` | 5k | 2.5 | 24.59 | 68.04 |
| `0501_0937_latent_ablation_5k_mae640_pos_enhanced_feat_408529` | 5k | 2.5 | 23.37 | 75.80 |

As a secondary check, evaluate exactly like `EXPERIMENTS.md`:
`cfg=1.0`, ImageNet validation FID stats, and 50k generated samples where
available. The 20k targets are:

| reference | step | cfg | samples | FID | IS |
|---|---:|---:|---:|---:|---:|
| baseline drifting | 20k | 1.0 | 50k | 32.09 | 39.67 |
| pos-enhanced | 20k | 1.0 | 50k | 29.45 | 43.34 |
| separate-weight baseline | 20k | 1.0 | 25k | 30.55 | 41.73 |
| pos-enhanced feature anchors | 20k | 1.0 | 50k | 24.68 | 58.03 |

The bar for the first prototype is therefore the 5k table, especially the
`23.37` FID pos-enhanced feature-anchor run. The 20k `24.68` FID result is a
secondary reference because it uses `cfg=1.0` and belongs to a longer training
regime.

## Evaluation Checklist

Run evaluation on this login node, not through Slurm. The primary eval should
match the 5k Torch-port ablations:

```bash
.venv/bin/python inference.py \
  --init-from runs/imagenet_likelihood_sparse_soft_5k/<run_name> \
  --workdir runs/imagenet_likelihood_sparse_soft_5k_eval/<run_name>_cfg2.5 \
  --cfg-scale 2.5 \
  --num-samples 50000 \
  --eval-batch-size 512 \
  --json-out runs/imagenet_likelihood_sparse_soft_5k_eval/<run_name>_cfg2.5/result.json
```

Then run the secondary `EXPERIMENTS.md`-style eval with `cfg=1.0` and 50k
samples. Report FID, IS, precision/recall when available, component usage
entropy, posterior entropy, top-k mass before truncation, and a small
nearest-train CLIP panel.

The checkpoint should be sampled three ways: sparse-soft top-8, hard top-1, and
one nearby sparse-soft top-k. If hard top-1 wins, anneal lower or train with a
smaller top-k next. If sparse-soft wins, the model is using likelihood
uncertainty productively.
