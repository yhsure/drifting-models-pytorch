---
marp: true
title: Explicit Likelihood Mixture Generators
description: A compact deck explaining the CIFAR prototype and ImageNet scaling plan.
theme: default
paginate: true
math: katex
style: |
  section {
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #18202a;
  }
  h1, h2 {
    color: #0e293f;
  }
  h1 {
    font-size: 2.15rem;
  }
  h2 {
    font-size: 1.35rem;
    margin-bottom: 0.55rem;
  }
  p, li {
    font-size: 0.92rem;
    line-height: 1.35;
  }
  table {
    font-size: 0.72rem;
  }
  code {
    color: #12364f;
  }
  .kicker {
    color: #516070;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }
  .big {
    font-size: 1.38rem;
    line-height: 1.25;
  }
  .note {
    color: #516070;
    font-size: 0.72rem;
  }
  .compact p {
    font-size: 0.78rem;
    line-height: 1.22;
  }
  .compact mjx-container {
    font-size: 82%;
  }
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    align-items: start;
  }
  .metric {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0e293f;
  }
---

<!-- _class: lead -->

# Explicit Likelihood Mixture Generators

<div class="kicker">CIFAR prototype → ImageNet drifting setup</div>

A finite likelihood model supplies semantic responsibilities.  
A shared conditional generator turns those responsibilities into images.

---

## The Problem

Plain drifting and rectified-flow baselines can generate images, but the latent
random variable is often implicit: a noise vector goes in, an image comes out.

The mixture idea is more explicit. We want a learned prior with visible
responsibilities:

$$
p(x) \approx \sum_k \pi_k p_\theta(x\mid k).
$$

The component variable should not be a class label or a nearest training image.
It should be learned from a likelihood in a semantic comparison space.

---

## Decoder-Style Latent Mixture

The first implementation stayed close to the original proposal. It trained a
small VAE and used its latent space as the decodable model space.

The mixture generator did **not** output images directly. It output latent
component centers and local noise scales:

$$
m_k,\ s_k
\quad\text{or}\quad
m=f_\theta(\epsilon),\ s=s_\psi(m).
$$

Sampling was:

$$
k\sim\mathrm{Cat}(\pi),\qquad
\eta\sim\mathcal N(0,I),\qquad
u=m_k+\operatorname{diag}(s_k)\eta,\qquad
\hat x=D(u).
$$

So the stochasticity lived in VAE latent space; the decoder turned each latent
sample into an image.

---

## Decoder-Style Training

<div class="compact">

$$
z_i=E(x_i),\qquad
v_i=\phi(z_i),\qquad
\hat v_j=\phi(u_j).
$$

Real images were encoded into VAE latents. Generated local latents were compared
to them in identity space or a frozen latent-feature space.

$$
\hat p(v_i)=\frac1M\sum_j\mathcal N(v_i;\hat v_j,\sigma^2I),
\qquad
\mathcal L=-\frac1B\sum_i\log\hat p(v_i).
$$

The learned generator is the explicit mixture `p_mix(u)` sampled before `x = D(u)`. Unlike a usual VAE, the prior is not `z ~ N(0, I)`. The VAE supplies the latent coordinate system and decoder. 

Shown: mixture samples decoded by `D`, not reconstructions.

</div>

![bg right:43% fit](../runs/cifar10_mixture_full_v1/0504_132941/samples_final.png)

---

## The Current Split

The model separates two jobs:

<div class="two">
<div>

**Likelihood prior**

Frozen descriptor space:

$$
v(x)=\frac{\phi(x)-\mu_\phi}{\sigma_\phi}.
$$

Learned Gaussian mixture:

$$
p_\alpha(v)=\sum_k \pi_k \mathcal N(v;c_k,\sigma^2I).
$$

</div>
<div>

**Image renderer**

Pixel-space or latent-space generator:

$$
\hat x = G_\theta(\epsilon, a).
$$

The conditioning vector `a` is derived from mixture responsibilities.

</div>
</div>

The frozen features define geometry; the generator remains directly samplable.

---

## Responsibilities Are the Interface

For a real image descriptor `v_i`, the mixture posterior is

$$
r_{ik}=p(k\mid v_i)
=
\frac{\pi_k\exp(-\lVert v_i-c_k\rVert^2/2\sigma^2)}
{\sum_\ell\pi_\ell\exp(-\lVert v_i-c_\ell\rVert^2/2\sigma^2)}.
$$

Hard conditioning samples one component:

$$
k_i\sim \mathrm{Cat}(r_i),\qquad a_i=e_{k_i}.
$$

Soft conditioning uses the posterior-weighted embedding:

$$
a_i=e(r_i)=\sum_k r_{ik}e_k.
$$

---

## Flow Variant on CIFAR

In the CIFAR explicit likelihood flow, the renderer is a shared UNet trained
with a rectified-flow objective:

$$
x_t=(1-t)x_0+tx_1,\qquad u_t=x_1-x_0.
$$

$$
\mathcal L_\text{flow}=
\lVert f_\theta(x_t,t,a_i)-(x_1-x_0)\rVert_2^2.
$$

The full loss keeps the mixture likelihood alive:

$$
\mathcal L=
\mathcal L_\text{flow}
+\lambda_\text{nll}\mathcal L_\text{mix}/d
+\lambda_\text{bal}\mathcal L_\text{bal}.
$$

---

## Approximate Mixture of Flows

A literal mixture of flows would define

$$
p(x)=\sum_k \pi_k p_{\theta_k}(x\mid k),
$$

or a shared conditional version

$$
p(x)\approx\sum_k\pi_k p_\theta(x\mid k).
$$

Our hard-prior training uses a stochastic approximation to the
responsibility-weighted conditional-flow objective:

$$
k_i\sim r_i,\qquad
\mathbb E_{k_i\sim r_i}\mathcal L_\text{flow}(x_i,k_i)
=
\sum_k r_{ik}\mathcal L_\text{flow}(x_i,k).
$$

Soft-prior training is a further shared-network approximation: instead of
choosing one expert, it conditions on the posterior mean embedding
`sum_k r_ik e_k`.

So it is not an exact log-likelihood mixture of normalizing flows. It is a
mixture-prior conditional flow, trained with likelihood-derived
responsibilities.

---

## Relation to GMFlow and GMM-VAE

`GMFlow` (arXiv:2504.05304) also puts a Gaussian mixture into flow matching, but
at a different level. It predicts a dynamic Gaussian mixture distribution over
velocity or denoising targets at each noisy point, then uses GM-SDE/ODE solvers
for few-step sampling.

Our model keeps the velocity model ordinary. The mixture lives in frozen
semantic descriptor space and produces a global component posterior used for
conditioning and prior sampling.

The GMM-VAE analogy is closer: GMM-VAE and VaDE replace the simple VAE latent
prior with a mixture prior for clustering and generation. Our method is similar
in spirit for flows: a mixture prior gives semantic responsibilities. The
difference is that we do not train an encoder/decoder ELBO or reconstruct the
frozen descriptor; the descriptor likelihood only defines responsibilities.

<div class="note">Refs: GMFlow arXiv:2504.05304; GMM-VAE arXiv:1611.02648; VaDE arXiv:1611.05148.</div>

---

## CIFAR Samples

![bg right:53% fit](../runs/0705_0858_cifar_flow_objective_nano_10k/samples_final.png)

The best CIFAR flow prototype uses the explicit likelihood prior with soft
conditioning and the NanoFlow objective.

Key checks:

- samples are not exact train duplicates;
- the prior can be sampled without a real image;
- soft conditioning slightly improved over hard conditioning.

---

## CIFAR Results

| Model | Samples | Denoise steps | FID ↓ |
|---|---:|---:|---:|
| Decoder-style latent mixture | 10k | 0, VAE decode | 83.11 |
| Rectified flow baseline | 2k | 32 | 55.96 |
| Class-conditioned flow | 2k | 32 | 52.48 |
| Explicit likelihood flow | 2k | 32 | 49.72 |
| Hard-prior likelihood flow | 10k | 64 | 27.91 |
| Soft-prior likelihood flow | 10k | 64 | 26.83 |
| Soft-prior NanoFlow | 10k | 64 | **26.78** |

The strongest CIFAR result is not just more parameters. It changes how the
generator is conditioned: the prior is learned by likelihood in feature space.

---

## Drifting Variant

The same likelihood prior can condition a one-step drifting generator instead
of an ODE flow.

![bg right:48% fit](../runs/cifar_likelihood_drift_multiscale_ft_10k_eval/0507_130600_soft/samples_final.png)

10k-sample CIFAR comparison:

| Model | Steps | FID ↓ |
|---|---:|---:|
| MeanFlow, one-step | 10k | 46.47 |
| Likelihood drift, soft prior | 18k | **41.48** |

The drift version is closer in spirit to the original drifting model, but the
flow version is currently the stronger CIFAR path.

---

## What Soft Prior Changed

Hard prior:

$$
k\sim p(k\mid v),\qquad a=e_k.
$$

Soft prior:

$$
a=\sum_k p(k\mid v)e_k.
$$

At sampling time, soft prior draws a synthetic descriptor from the learned
mixture, recomputes responsibilities, and conditions on the resulting weighted
embedding.

This makes the component address less brittle while keeping the model explicitly
sampled from the learned mixture.

---

## ImageNet Scaling Plan

The ImageNet version should live inside the latent drifting setup:

$$
z\in\mathbb R^{32\times32\times4},\qquad
v(z)=\operatorname{pool}(\phi_\text{MAE}(z)).
$$

Use a large shared mixture, class-conditional weights, and sparse-soft
responsibilities:

$$
a_i=\sum_{k\in \operatorname{TopK}(r_i,K_\text{sparse})}
\tilde r_{ik}e_k.
$$

Recommended first prototype:

```text
5k steps, K=8192, top-k 16 -> 8, tau 1.5 -> 0.45
train on Slurm, evaluate on the login node
```

---

## ImageNet Yardstick

Primary comparison: existing 5k Torch-port latent ablations, evaluated with
50k samples at `cfg=2.5`.

| Reference | Step | CFG | FID ↓ | IS ↑ |
|---|---:|---:|---:|---:|
| MAE640 baseline | 5k | 2.5 | 29.50 | 51.76 |
| Pos-enhanced feature run A | 5k | 2.5 | 24.59 | 68.04 |
| Pos-enhanced feature run B | 5k | 2.5 | **23.37** | **75.80** |

Secondary comparison: 20k `EXPERIMENTS.md` results at `cfg=1.0`, where the
best 20k reference is `24.68` FID for pos-enhanced feature anchors.

---

## What Would Count as Success?

For the first ImageNet prototype:

1. train a matched 5k sparse-soft likelihood prior run;
2. evaluate with the exact 5k reference settings: `cfg=2.5`, 50k samples;
3. sample the same checkpoint with sparse-soft top-8, hard top-1, and one
   nearby top-k;
4. inspect nearest-train neighbors, not only FID.

The first real bar is the 5k `23.37` FID reference. If sparse-soft gets close or
beats it without replay, the method deserves a longer 20k run.

---

## Takeaway

<div class="big">
The mixture is no longer trying to be the whole generator.
</div>

It learns semantic responsibilities through an explicit likelihood. The
generator learns how to render from those responsibilities.

That gives us a model with:

- a real prior over components;
- direct image or latent sampling;
- a natural bridge back to drifting models;
- a clean ImageNet prototype path.
