from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Override workspace/Slurm cache exports so test imports stay hermetic.
_cache_root = ROOT / ".test-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_cache_root / "torchinductor")
os.environ["TORCH_HOME"] = str(_cache_root / "torch")
os.environ["HF_ROOT"] = str(_cache_root / "hf")
os.environ["UV_CACHE_DIR"] = str(_cache_root / "uv")
for _key in (
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_HOME",
    "HF_ROOT",
    "UV_CACHE_DIR",
):
    Path(os.environ[_key]).mkdir(parents=True, exist_ok=True)
