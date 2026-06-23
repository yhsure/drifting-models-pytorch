# Current Line Of Thought: Drift Training Mechanisms

This note captures the working theory and the controlled ImageNet-latent results before Jupiter downtime. The aim is to make the next server work comparable to the current runs, not to freeze the idea too early.

## Reference Setup

Unless otherwise noted, the controlled comparisons below resume from:

```text
runs/0616_1805_measure_stratified_10k_slurm8/checkpoints/step_000010000.pt
```

and evaluate at:

```text
cfg = 2.5
num_samples = 50000
seed = 20260617
PR samples = 10000 ImageNet val images
```

The current best 15k recipe is:

```text
configs/gen/latent_ablation_15k_mae640_balanced_pos_control_continue.yaml
runs/0617_balanced_pos_control_from_10k
```

## Main Training Hypothesis

The support sampler should be treated as part of the estimator, not as a heuristic data loader. If we bias positive support selection toward locally useful supports, the loss must include the corresponding likelihood correction. Otherwise the model is trained on a changed target measure and can overfit the local neighborhood distribution.

The best working recipe is therefore:

1. Sample many close same-class positives because they carry stronger local drift information.
2. Keep some mid/far same-class positives so the drift target remains globally anchored.
3. Apply exact `p/q` correction in the loss once the sampler is trusted.
4. Anneal the sampling bias back toward a broader distribution instead of keeping a permanently sharp neighborhood-only objective.

In code this is the `weight_mode: "measure"` path plus `pos_sampling.strategy: "importance_feature_stratified"`.

## What The Current Best Recipe Does

The balanced 15k continuation uses rank strata over same-class feature distance:

```text
strata_edges: [0.125, 0.5, 1.0]
```

The bins are close, mid, and far. The allocation begins more close-heavy and moves toward a balanced/global allocation:

```text
allocation:       [0.34375, 0.40625, 0.25]
allocation_final: [0.125,   0.4375,  0.4375]
```

Importance correction is annealed in:

```text
weight_power: 0.0 -> 1.0 over steps 2500..10000
```

Local bias is also annealed:

```text
local_alpha: 1.0 -> 0.0 over steps 5000..25000
```

So the answer to "do we still sample close positives late?" is: yes during the tested 15k window, but less aggressively, and with full correction by 10k. If training continues to 25k under this schedule, the recipe returns close to uniform rank sampling.

## Controlled Results So Far

| 15k recipe | FID down | IS up | Precision up | Recall up |
|---|---:|---:|---:|---:|
| balanced stratified positives | **8.996** | 281.55 | **0.8512** | **0.3174** |
| original positive control | 9.132 | **281.88** | 0.8453 | 0.3151 |
| force-continuous data anchor | 9.057 | 279.28 | 0.8505 | 0.3108 |
| force-continuous generation anchor | 9.101 | 278.41 | 0.8461 | 0.3085 |
| rank-continuous gentle | 9.174 | 279.75 | 0.8468 | 0.3075 |
| signed hard negatives | 9.225 | 279.84 | 0.8494 | 0.3113 |
| rank-continuous sharp | 9.249 | 279.26 | 0.8475 | 0.3117 |

The best controlled result is still the balanced stratified sampler. The continuous alternatives were useful tests, but they did not beat the binned recipe.

## Interpretation

The surprising result is not that "bins are objectively optimal." The current evidence says this particular coarse stratification is a good regularizer. It may be winning because it enforces explicit coverage of local, mid, and far supports while keeping the importance weights numerically controlled.

The continuous force-leverage proposal was too close to uniform in practice. Its diagnostics showed almost no meaningful proposal distortion, so its underperformance is not decisive against all continuous methods.

The rank-continuous gentle and sharp runs were a more meaningful test. Sharp had real bias and nontrivial weights, but still underperformed. That makes me less excited about monotone continuous kernels over rank alone. They spend probability smoothly, but the training objective seems to benefit from hard coverage constraints.

## Negatives

Generation-anchored interclass hard negatives are plausible in principle: if a generated sample is close to another class, negatives from that other class should teach the model where not to drift.

The tested signed hard-negative version did not help. It reduced FID/recall relative to the balanced positive control. My current read is that the negative signal is easy to make too sharp or semantically brittle. It may push away from plausible ImageNet boundary cases rather than only removing bad off-manifold directions.

Hard negatives should not be the next primary axis unless they are made softer and gated by high-confidence semantic disagreement.

## Why The Binned Sampler May Be Working

The drift target is a finite-support estimator. Close positives reduce variance and improve local geometric alignment, but far positives prevent the estimator from collapsing into a narrow support shell. The bins make this bias explicit:

```text
close positives: local tangent/denoising signal
mid positives: manifold continuity signal
far positives: class-level anchoring and mode coverage
```

The `p/q` correction means this is still an estimator of the intended support measure, subject to the current weight clipping and ESS controls. That is the core principle worth preserving.

## What I Would Try Next

The next experiments should keep controls tight: same 10k checkpoint, same eval seed, same 50k eval, one mechanism changed at a time.

1. Extend the best balanced recipe from 15k to 25k or 30k.
   This tests whether returning toward uniform late actually helps or whether the recipe peaks around 15k.

2. Sweep only the final allocation.
   Compare the current `[0.125, 0.4375, 0.4375]` against a more local final allocation like `[0.20, 0.45, 0.35]` and a more global one like `[0.05, 0.45, 0.50]`.

3. Sweep only the local-alpha floor.
   The current schedule goes to `0.0`. Test a floor such as `0.25` so the sampler never fully forgets close supports.

4. Test more bins, not smoother kernels.
   A 5-bin sampler with minimum mass per bin may preserve the winning coverage property while reducing the arbitrariness of the 3-bin boundaries.

5. Try soft hard negatives only after the positive sampler is settled.
   If revisited, use small weight, confidence gating, and an ablation with identical positive sampling.

## Practical Recommendation

For the new server, treat this as the default recipe:

```text
configs/gen/latent_ablation_15k_mae640_balanced_pos_control_continue.yaml
```

Use the 10k checkpoint as the common restart point for new controlled experiments. Use the 15k balanced checkpoint only when the experiment is explicitly "continue the current best run."

Do not compare new recipes against old metrics unless the eval uses the same seed, same ImageNet stats, same precision/recall reference, and same `cfg=2.5` 50k protocol.
