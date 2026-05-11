# CIFAR Explicit Likelihood Mixture Flow

This note documents the current mixture-based CIFAR model in `mixture.py`.
The goal is to stay closer to the original Gaussian-mixture likelihood idea
than a plain rectified-flow baseline, while keeping a strong non-memorizing
sampler.

The working CIFAR run is:

```bash
.venv/bin/python mixture.py --pipeline cifar_likelihood_flow --dataset cifar10 \
  --device cuda --steps 10000 --mode-count 128 --batch-size 256 \
  --lr 0.0002 --unet-dim 64 --unet-dim-mults 1,2,4 \
  --sample-every 2000 --save-every 2000 --log-every 50 \
  --flow-sample-steps 32 --flow-eval-samples 2048 \
  --feature-sigma 5.0 --mixture-nll-weight 0.04 \
  --mixture-balance-weight 0.03 --mixture-resp-temp 0.6 \
  --cond-dropout 0.1 --flow-guidance-scale 1.3 \
  --out-dir runs/cifar_likelihood_flow_v1
```

The corresponding run directory is
`runs/cifar_likelihood_flow_v1/0504_165326`.

The same code path also supports the soft-prior variant by adding
`--likelihood-conditioning soft`. In that case the model does not sample a
single component embedding during training; it conditions the flow on the
responsibility-weighted embedding described below.

## Bare Explicit Likelihood

The most direct version of the idea is just a Gaussian mixture over data
representations. If `z_i` are image latents or frozen image features, and
`m_j` are generated particles or learned modes, the model is

$$
\hat p(z)
=
\frac1M\sum_{j=1}^{M}\mathcal N(z;m_j,\sigma^2I),
$$

trained by

$$
\mathcal L_{\mathrm{NLL}}
=
-\frac1B\sum_i
\log
\frac1M\sum_j
\mathcal N(z_i;m_j,\sigma^2I).
$$

That objective is clean and explicit, and it was the original target. On its
own, though, it gives weak image samples unless the representation is very
decodable and the particles do not collapse into training-example replay. Our
CIFAR experiments showed both failure modes: latent decoders made samples soft,
while data-like particles could look strong but failed nearest-train diagnostics.

The current model keeps this likelihood machinery, but changes what the mixture
is responsible for. The mixture no longer tries to be the whole generator. It
learns semantic regions and posterior responsibilities in feature space; a
conditional sampler then renders images from those regions.

## Model Definition

The model separates two roles that are easy to confuse. The explicit likelihood
lives in a frozen descriptor space, while the sampler lives in pixel space. The
descriptor space decides which mixture component an image belongs to; the pixel
model learns how to generate images conditioned on that component.

First, each image is mapped to a fixed descriptor. Let `phi(x)` be the frozen
feature encoder and let

$$
v(x) = \frac{\phi(x)-\mu_\phi}{\sigma_\phi}
$$

be the standardized descriptor. In the current CIFAR implementation, `phi` is a
frozen ResNet-18 descriptor, intended to be ImageNet-pretrained, using stages 3
and 4 plus a few low-level image statistics. This descriptor is not learned in
this experiment.

On top of these descriptors, we fit a trainable Gaussian mixture:

$$
p_\alpha(v)
=
\sum_{k=1}^{K}
\pi_k \mathcal N(v; c_k, \sigma^2 I),
$$

where

$$
\alpha = \{c_1,\ldots,c_K,\pi_1,\ldots,\pi_K\}.
$$

Here `alpha` denotes the learned mixture parameters: the component centers
`c_k` and mixture weights `pi_k`. The centers are initialized by k-means in
standardized descriptor space; after initialization, both centers and mixture
logits are optimized. These parameters are learned tensors, not a neural
network.

The mixture gives an explicit likelihood for each training image and an
explicit posterior over components:

$$
r_{ik}
=
p(k \mid v_i)
=
\frac{
\pi_k \exp\left(-\lVert v_i-c_k\rVert^2 / 2\sigma^2\right)
}{
\sum_\ell
\pi_\ell \exp\left(-\lVert v_i-c_\ell\rVert^2 / 2\sigma^2\right)
}.
$$

This is the only place where the frozen descriptors enter the training signal.
They define the likelihood and the responsibilities. We do not decode
`v(x)`, and we do not train the model to reconstruct `v(x)`.

The responsibilities then condition a pixel-space generator. There are two
conditioning variants. The hard-prior variant samples one component

$$
k_i \sim \mathrm{Cat}(r_i)
$$

and conditions on the corresponding learned embedding `e_{k_i}`. The soft-prior
variant uses the whole posterior distribution and conditions on the
responsibility-weighted embedding

$$
e(r_i)=\sum_k r_{ik}e_k.
$$

Both variants use the same explicit mixture likelihood. The difference is only
how the likelihood-derived posterior is passed into the pixel-space flow:
hard conditioning makes a single expert choice, while soft conditioning gives
the shared UNet a local mixture average.

Given a Gaussian source `x0`, real image `x1`, and time `t`, the rectified-flow
interpolant is

$$
x_t = (1-t)x_0 + t x_1,\qquad u_t = x_1-x_0.
$$

The UNet predicts

$$
\hat u_t = f_\theta(x_t,t,a_i),
$$

where `a_i=e_{k_i}` in the hard-prior variant and `a_i=e(r_i)` in the
soft-prior variant. The neural parts of the model are the shared UNet
`f_theta` and the component embeddings. There is one UNet for all components;
changing the component posterior changes the conditioning vector, not the
network weights.

So the model is mixture-of-experts-like in its probability structure:
components select different conditional image distributions. But the component
choices come from likelihood-derived responsibilities in descriptor space, and
the image generator itself is a single shared conditional network.

At generation time, the posterior is no longer needed because there is no real
image to condition on. The hard-prior sampler draws

$$
k \sim \mathrm{Cat}(\pi),\qquad x_0\sim\mathcal N(0,I),
$$

then integrates the component-conditioned velocity field. The generated image
therefore comes from the learned mixture prior and the shared conditional flow.
The soft-prior sampler starts the same way, but treats the sampled component as
a way to draw a synthetic descriptor

$$
\tilde v \sim \mathcal N(c_k,\sigma^2I),
$$

then recomputes responsibilities under the mixture and conditions on

$$
e(\tilde r)=\sum_j p(j\mid \tilde v)e_j.
$$

This is why "soft prior" is still a prior-sampling procedure rather than a
reconstruction step. It samples from the learned mixture in descriptor space,
uses that sample to form a soft component posterior, and renders pixels through
the same conditional flow.

A null component is trained with component dropout, so sampling can use
classifier-free-style guidance:

$$
f_{\mathrm{cfg}} = f_{\varnothing} + s(f_k-f_{\varnothing}).
$$

## Objective

The training loss is a hybrid objective:

$$
\mathcal L
=
\mathcal L_{\mathrm{flow}}
+ \lambda_{\mathrm{nll}}\frac{\mathcal L_{\mathrm{mix}}}{d}
+ \lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}}.
$$

The flow term is standard rectified-flow MSE:

$$
\mathcal L_{\mathrm{flow}}
=
\lVert f_\theta(x_t,t,a_i) - (x_1-x_0)\rVert_2^2.
$$

The mixture term is the explicit negative log likelihood:

$$
\mathcal L_{\mathrm{mix}}
=
-\frac1B\sum_i
\log
\sum_k
\pi_k\mathcal N(v_i;c_k,\sigma^2 I).
$$

The balance term mildly discourages posterior collapse by penalizing uneven
average component usage within a batch:

$$
\mathcal L_{\mathrm{bal}}
=
\sum_k \bar r_k \log(K\bar r_k),
\qquad
\bar r_k=\frac1B\sum_i r_{ik}.
$$

In the current CIFAR run, the mixture NLL mostly acts as a stable responsibility
model after k-means initialization. The important thing is that assignments are
explicit and likelihood-derived, and the sampler is trained around those
assignments rather than around labels or copied image particles.

## Why This Is a Mixture Model

`cifar_class_flow` improved over the unconditional baseline, but its mixture
variable was simply the CIFAR label:

$$
p(x)=\sum_y p(y)p_\theta(x\mid y).
$$

That is useful evidence but not the original mixture-likelihood idea. In
`cifar_likelihood_flow`, the component variable is learned from a Gaussian
mixture objective:

$$
p(x)\approx \sum_k \pi_k p_\theta(x\mid k),
\qquad
k \text{ trained through } p_\alpha(v).
$$

The mixture part is therefore unsupervised, finite, likelihood-bearing, and
sampled at generation time. The frozen features are the comparison space for
this likelihood; the pixel-space flow is the stochastic renderer attached to
the learned mixture components.

It is useful to think of this as a soft mixture of conditional generators:

$$
p(x)
\approx
\sum_k \pi_k p_\theta(x\mid k).
$$

The important detail is how the component information is learned. In the hard
variant, `k` is sampled from likelihood-derived responsibilities `p(k | v_i)`,
not from class labels and not from nearest training examples. In the soft
variant, those same responsibilities are used directly as a weighted component
embedding. During sampling, the hard variant draws `k` from the learned mixture
weights, while the soft variant draws a descriptor from the learned mixture and
uses its posterior responsibilities. The implementation still has one
optimizer, one EMA model, and one shared UNet; only the conditioning vector
changes.

## CIFAR Results

All rows below use 10k training steps, UNet dim 64, `dim_mults=1,2,4`, 32 flow
sampling steps, and 2048 generated samples for FID.

| run | method | FID vs CIFAR test | gen-to-train median NN | exact train rate |
|---|---|---:|---:|---:|
| `cifar_rectified_flow_v1/0504_153840` | Gaussian RF baseline | `55.9566` | `3.1667` | `0.0` |
| `cifar_class_flow_v1/0504_162216` | class-conditioned RF | `52.4794` | `3.1815` | `0.0` |
| `cifar_likelihood_flow_v1/0504_165326` | explicit likelihood mixture flow | `49.7249` | `3.3222` | `0.0` |

For reference, held-out CIFAR test images have train-nearest median distance
`3.4104` under the same diagnostic. The explicit mixture model is therefore
not showing the train-neighbor collapse seen in the earlier data-particle runs.

Later evaluations used 10k generated samples against the CIFAR-10 test set and
made the hard/soft distinction explicit. With the manual flow objective, the
hard-prior checkpoint from `cifar_likelihood_flow_v1/0504_165326` reached
`27.9123` FID at guidance `1.5` and 64 sampler steps. The soft-prior checkpoint
from `cifar_likelihood_flow_soft_v1/0505_220501` reached `26.8285` FID at
guidance `1.8` and 64 sampler steps. The best soft-prior NanoFlow checkpoint so
far, `0705_0858_cifar_flow_objective_nano_10k`, reached `26.7827` FID at
guidance `2.0` and 64 sampler steps. All three had exact train rate `0.0`.

## Runtime Notes

The original full runs were not instrumented with explicit phase timers, so the
end-to-end numbers are approximate and taken from progress/file timestamps on
the local GH200 run.

The unconditional RF baseline ran from about `15:38` to `15:54`, roughly
16 minutes end-to-end for 10k steps plus sampling/eval. The explicit likelihood
flow ran from about `16:53` to `17:21`, roughly 28 minutes end-to-end. The
training progress bar for the likelihood run reported `27:54` for 10k steps,
about `5.97 it/s`.

The likelihood run was therefore about `1.7x` slower end-to-end in that
particular comparison. That should not be read as the cost of the explicit
likelihood training step. The full likelihood run used different sample/save
cadence, and guided sampling at eval time evaluates the UNet twice per ODE step.
Feature precompute plus k-means took only a few seconds on CIFAR, and the
training-time responsibility computation is small:
`batch_size x mode_count = 256 x 128` distance work in descriptor space.

A short matched timing probe removes periodic samples, checkpointing, FID eval,
and guidance:

```bash
# RF baseline
.venv/bin/python mixture.py --pipeline cifar_rectified_flow --dataset cifar10 \
  --device cuda --steps 300 --sample-every 0 --save-every 0 \
  --log-every 0 --flow-eval-samples 0 --out-dir runs/timing_cifar_rf

# Explicit likelihood flow
.venv/bin/python mixture.py --pipeline cifar_likelihood_flow --dataset cifar10 \
  --device cuda --steps 300 --mode-count 128 --sample-every 0 \
  --save-every 0 --log-every 0 --flow-eval-samples 0 \
  --flow-guidance-scale 1.0 --out-dir runs/timing_cifar_likelihood
```

The RF probe reported `300/300 [00:28, 10.55 it/s]` and wall time `43.172s`.
The explicit likelihood probe reported `300/300 [00:25, 11.61 it/s]` and wall
time `34.655s`, despite also doing feature precompute and k-means. The exact
numbers have normal run-to-run noise, but the conclusion is clear: the explicit
likelihood training loop can run at essentially RF speed when guidance and eval
bookkeeping are matched.

For a cleaner future comparison, log `time.perf_counter()` around training and
evaluation separately. The useful split is:

```text
feature_precompute_seconds
train_seconds
sample_eval_seconds
steps_per_second
```

That would separate the one-time mixture setup cost from the per-step training
cost and the guidance-driven sampling cost.

## Current Read

This is the first CIFAR variant that both preserves the explicit mixture
likelihood story and beats the RF baseline without obvious replay. The result
suggests that the mixture components should live in semantic feature space,
while the image sampler should be a shared conditional generator around those
components.

The ImageNet version should probably make the mixture hierarchical: class or
coarse semantic component at the top, learned feature-space submodes underneath,
and one shared conditional generator. The thing to avoid is returning to
training-image particles as components; the nearest-neighbor diagnostics showed
that route can look good while failing the generalization test.
