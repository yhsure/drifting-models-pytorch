## Generative Models via Drifting Models in Torch
* **Reference**: https://arxiv.org/html/2602.04770 (see `paper/main.tex` and `paper/paper.pdf`)
* **Framework**: PyTorch (`torch`)

A generator is trained to output samples at equilibrium under a drift field. Sampling requires one forward pass.

### Core Idea

Match the model distribution $q$ to the data distribution $p$ utilizing a kernel function. Define the anti-symmetric drift:

$$V_{p,q}(x)=V_p^+(x)-V_q^-(x)$$

When distributions match:

$$q=p \implies V_{p,q}=V_{q,p}=-V_{p,q} \implies V_{p,q}=0$$

For each generated sample $x$:
* Pull toward nearby real data.
* Push away from nearby generated samples.

### Training Objective

Let $\epsilon \sim \mathcal{N}(0,I)$.

$$\mathcal{L}=\text{MSE}(f(\epsilon),\text{stopgrad}(f(\epsilon)+V_{p,q}(f(\epsilon))))$$

### Drift Definition

Define the kernel function where $y^+$ is a data sample from $p$ and $y^-$ is a generated sample from $q$:

$$k(x,y)=\exp\left(-\frac{\|x-y\|}{\tau}\right)$$

Under contrastive mean-shift, the components are:

$$V_p^+(x)=\frac{\mathbb{E}[k(x,y^+)(y^+-x)]}{Z_p}$$

$$V_q^-(x)=\frac{\mathbb{E}[k(x,y^-)(y^--x)]}{Z_q}$$

Normalization factors $Z_p$ and $Z_q$ are approximated via batch kernel similarities.

---

## Environment & Execution

* **Virtual Environment & Dependencies**: Managed via `uv`.
    * Install: `uv add <package-name>`
    * Execute: `uv run <script-name>.py`
* **Tasks**: Managed via `invoke`. Run `uv run invoke --list`.
* **Pre-commit Hooks**: Run `uv run pre-commit run --all-files`.
* **Commit Helper**: Use `uv run invoke commit -m "Message"` to auto-run pre-commit fixes and continue the commit if no errors remain.
* **Experiment Entrypoints**:
    * Toy: `uv run examples/train_toy.py --dataset swissroll --method drifting --tau 0.08`
    * CIFAR-10: `uv run examples/train_cifar10.py --method drifting --unet-dim 64 --unet-dim-mults 1,2,4`
    * Comparison: `uv run examples/compare_methods.py --domain toy --dataset swissroll`
* **CI Workflows**:
    * `.github/workflows/ci-lint-type.yml`
    * `.github/workflows/ci-tests.yml`
    * `.github/workflows/ci-precommit.yml`

## Code Style & Tooling

* **Formatting & Linting**: `ruff`
    * Format: `uv run ruff format .`
    * Lint: `uv run ruff check . --fix`
* **Type Checking**: `ty`
    * Check: `uv run ty check`
* **Testing**: `pytest`
    * Run: `uv run pytest tests/`
* **Constraints**:
    * Line length $\le$ 120 characters.
    * Use type hints and f-strings.
    * Keep inline comments to a minimum.

## Documentation

* **Framework**: `mkdocs` (build locally with `uv run mkdocs serve`). Not enabled in this project.
* **Docstrings**: Google style mandatory for all functions and classes.
* **Commits**: Keep commit messages short and with a capitalized first letter.
* **Maintenance**: Update this `AGENTS.md` file upon introducing new tools or commands.
