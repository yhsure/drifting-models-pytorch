# ImageNet Decoder-Style Likelihood Plan

Use the pre-cached SDVAE latents as the decodable model space, but compute the
mixture likelihood in frozen latent-MAE feature space.

Let `z_i = E(x_i)` be the cached SDVAE latent and let

$$
v_i = T(z_i)
    = \operatorname{standardize}\!\left(\operatorname{pool}(\phi_{\rm MAE}(z_i))\right).
$$

The generator is an explicit stochastic mixture in SDVAE latent space:

$$
k \sim \pi_{y}, \qquad
\eta \sim \mathcal N(0,I), \qquad
u = m_k + S_k \eta, \qquad
\hat x = D(u).
$$

Here `m_k` is an SDVAE-latent component center, `S_k` is initially diagonal or
channel-wise, and `pi_y` is class-conditional. The SDVAE decoder `D` and
latent-MAE feature extractor are frozen.

Training uses the likelihood of real descriptors under generated latent
samples pushed through the same frozen feature map:

$$
\hat p(v_i \mid y_i)
=
\sum_{k=1}^K
\pi_{k\mid y_i}
\mathbb E_{\eta}
\left[
\mathcal N\!\left(
v_i;\,
T(m_k+S_k\eta),\,
\sigma^2 I
\right)
\right],
$$

with Monte Carlo samples for the expectation. The loss is

$$
\mathcal L
=
-\frac1B\sum_i \log \hat p(v_i\mid y_i)
+\lambda_{\rm bal}\mathcal L_{\rm bal}
+\lambda_{\rm anch}\sum_k\lVert m_k-m_k^{(0)}\rVert^2.
$$

Initialize `K=8192` or `16384` by mini-batch k-means in `v`-space, then set
`m_k^(0)` to the mean SDVAE latent of each cluster. This avoids direct
one-component-per-image memorization while keeping the components decodable.

The main run should use latent-MAE features for `T`. A raw SDVAE-latent
likelihood,

$$
T(z)=\operatorname{standardize}(z),
$$

is only a diagnostic ablation. It is simpler, but likely worse because raw
SDVAE latent distances need not match semantic neighborhoods.

Run a 5k-step prototype and evaluate with the exact same ImageNet validation
FID/IS protocol and CFG setting as the existing 5k drifting runs. The first
target is the `23.37` FID 5k reference.
