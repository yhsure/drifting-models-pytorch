import importlib


MODULES = [
    "drift_loss",
    "memory_bank",
    "dataset.dataset",
    "dataset.latent",
    "dataset.vae",
    "models.generator",
    "models.hf",
    "models.mae_model",
    "utils.fid_util",
    "utils.model_builder",
]


def test_imports():
    for name in MODULES:
        assert importlib.import_module(name) is not None
