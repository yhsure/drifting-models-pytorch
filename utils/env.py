"""Global paths for the Drift PyTorch port."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _PROJECT_ROOT.parents[2]

IMAGENET_PATH = os.environ.get("IMAGENET_PATH", str(_WORKSPACE_ROOT / "imagenet"))
IMAGENET_CACHE_PATH = os.environ.get("IMAGENET_CACHE_PATH", str(_WORKSPACE_ROOT / "imagenet_latent_cache"))
IMAGENET_FID_NPZ = os.environ.get("IMAGENET_FID_NPZ", str(_WORKSPACE_ROOT / "imagenet_stats" / "imagenet_256_fid_stats.npz"))
IMAGENET_PR_NPZ = os.environ.get("IMAGENET_PR_NPZ", str(_WORKSPACE_ROOT / "imagenet_stats" / "imagenet_val_prc_arr0.npz"))

HF_REPO_ID = os.environ.get("HF_REPO_ID", "Goodeat/drifting")
HF_ROOT = os.environ.get("HF_ROOT", str(_PROJECT_ROOT / "hf_cache"))
