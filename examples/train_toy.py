"""Toy training entrypoint placeholder."""

from drifting_models_pytorch import DriftingModelConfig, build_model


def main() -> None:
    """Run a placeholder toy training setup."""
    _ = build_model(DriftingModelConfig(dim=2))
    raise NotImplementedError("Toy training is not implemented yet.")


if __name__ == "__main__":
    main()
