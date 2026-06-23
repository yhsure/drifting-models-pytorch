from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Run generator training without stamping the workdir.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    from train import main as train_main

    train_main(args)


if __name__ == "__main__":
    main()
