# Learnable Per-Feature-Key Sigma for JSD MoG Loss

## Goal

Replace the fixed `mog_sigma` scalar in `train_step` with one learnable
`nn.Parameter` per feature key, trained end-to-end via backprop.  Each key
corresponds to a distinct feature group produced by `build_activation_function`
(e.g. different MAE blocks × patch/std/mean transformations), which can have
different channel dimensions `D` and therefore need different bandwidth values.

---

## Change 1 — `drift_loss.py`: support tensor sigma in `_make_mog`

**File:** `drift_loss.py`, function `_make_mog`

**Current:**
```python
Independent(Normal(loc=locs, scale=torch.full_like(locs, sigma)), 1)
```

**Change to:**
```python
Independent(Normal(loc=locs, scale=sigma * torch.ones_like(locs)), 1)
```

`torch.full_like` does not propagate gradients through a tensor `sigma`
argument; multiplying `sigma * torch.ones_like(locs)` does.

Also widen the type annotation so callers can pass either a float or a tensor:

```python
def _make_mog(locs: torch.Tensor, sigma: float | torch.Tensor) -> MixtureSameFamily:
```

---

## Change 2 — `train.py`: add `mog_log_sigmas` to `TrainState`

**File:** `train.py`, `TrainState` dataclass (lines 35–43)

Add one field:

```python
mog_log_sigmas: Optional[dict] = None   # {feature_key: nn.Parameter(log sigma)}
```

---

## Change 3 — `train.py`: lazy-initialise sigmas in `train_step`

**File:** `train.py`, `train_step`

After `sg_features` is computed (line ~125) but before the feature loop,
insert a lazy-init block that runs only on the first step with `lambda_mog > 0`:

```python
if lambda_mog > 0.0 and state.mog_log_sigmas is None:
    import math
    state.mog_log_sigmas = {}
    for k, v in sg_features.items():
        D = v.shape[-1]                          # channel dim of this key
        log_sigma_init = math.log(D ** 0.5)     # sigma_0 = sqrt(D)
        param = torch.nn.Parameter(
            torch.tensor(log_sigma_init, dtype=torch.float32, device=device)
        )
        state.mog_log_sigmas[k] = param
    state.optimizer.add_param_group({"params": list(state.mog_log_sigmas.values())})
```

Why log-space: sigma must stay positive; learning `log(sigma)` and exposing
`exp(log_sigma)` gives unconstrained optimisation with a natural positive
constraint.

---

## Change 4 — `train.py`: use learnable sigma in the per-key loop

**File:** `train.py`, inside the `for k in sg_features.keys()` loop (lines 179–184)

Replace:
```python
_sigma = mog_sigma if mog_sigma is not None else float(feature_gen_mog.shape[-1] ** 0.5)
mog_k = jsd_mog_loss(feature_gen_mog, feature_pos, sigma=_sigma)
total_loss = total_loss + lambda_mog * mog_k
total_info[f"mog_loss/{k}"] = float(mog_k.detach().cpu().item())
```

With:
```python
sigma_k = state.mog_log_sigmas[k].exp()
mog_k = jsd_mog_loss(feature_gen_mog, feature_pos, sigma=sigma_k)
total_loss = total_loss + lambda_mog * mog_k
total_info[f"mog_loss/{k}"] = float(mog_k.detach().cpu().item())
total_info[f"mog_sigma/{k}"] = float(sigma_k.detach().cpu().item())
```

The `total_loss.backward()` call that already exists a few lines later will
propagate gradients through `sigma_k` into `state.mog_log_sigmas[k]`.

The `optimizer.step()` that follows updates all param groups, including the
newly added mog sigma group.

---

## Change 5 — `train.py`: remove `mog_sigma` parameter

**File:** `train.py`, `train_step` and `train_gen` signatures

Remove `mog_sigma=None` from both `train_step` (line 98) and `train_gen`
(wherever it is forwarded).  The per-key learnable params replace this.

---

## Change 6 — `utils/ckpt_util.py`: checkpoint `mog_log_sigmas`

**File:** `utils/ckpt_util.py`

### `save_checkpoint`

Add to the payload dict:

```python
"mog_log_sigmas": (
    {k: v.detach().cpu() for k, v in state.mog_log_sigmas.items()}
    if getattr(state, "mog_log_sigmas", None) is not None else None
),
```

### `restore_checkpoint`

After restoring the optimizer state, add:

```python
mog_payload = payload.get("mog_log_sigmas")
if mog_payload is not None and getattr(state, "mog_log_sigmas", None) is not None:
    for k, v in mog_payload.items():
        if k in state.mog_log_sigmas:
            state.mog_log_sigmas[k].data.copy_(v.to(device=state.mog_log_sigmas[k].device))
```

Note: `restore_checkpoint` is called before `mog_log_sigmas` are initialised
(they are lazily created on the first forward pass).  The simplest handling is
to store the raw tensors in `state` at restore time and let the lazy-init block
check for them.  Concretely:

- Add a second optional field to `TrainState`: `mog_log_sigma_init: Optional[dict] = None`
- In `restore_checkpoint`, write recovered tensors to `state.mog_log_sigma_init`
- In the lazy-init block in `train_step`, if `state.mog_log_sigma_init` is not
  None *and* contains the current key `k`, use that value instead of `log(sqrt(D))`

This avoids any ordering dependency between checkpoint restore and the first
training step.

---

## Change order (safe to implement sequentially)

1. `drift_loss.py` — Change 1 (smallest, isolated, unblocks gradient flow)
2. `TrainState` — Change 2 (dataclass field addition, no logic)
3. `train_step` lazy init — Change 3
4. Per-key loop — Change 4
5. Remove `mog_sigma` — Change 5
6. Checkpoint — Change 6

---

## What is NOT changing

- `jsd_mog_loss` signature stays `(gen_features, real_features, sigma)` — pure function
- The flat-MoG approach (`batch_shape=()`, `reshape(-1, D)`) stays unchanged
- Drift loss code is untouched
- The `sigma_latent` parameter (used for the two-stage model noise level) is separate and stays
