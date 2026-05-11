from __future__ import annotations

from types import SimpleNamespace

import torch

from likelihood_prior import _save_likelihood_fit_cache, _validate_cached_fit
from models.generator import DitGen
from utils.ckpt_util import save_params_ema_artifact
from utils.init_util import load_generator_model_and_params


def test_likelihood_fit_cache_roundtrip(tmp_path):
    path = tmp_path / "likelihood_prior.pt"
    centers = torch.randn(4, 3)
    feature_mean = torch.randn(3)
    feature_std = torch.rand(3).clamp_min(0.1)
    class_log_probs = torch.log_softmax(torch.randn(5, 4), dim=-1)

    _save_likelihood_fit_cache(
        path,
        centers=centers,
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_log_probs=class_log_probs,
        sigma=0.75,
        descriptor_keys=["layer4_mean", "layer4_std"],
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    loaded = _validate_cached_fit(payload, component_count=4, num_classes=5)
    loaded_centers, loaded_mean, loaded_std, loaded_log_probs, loaded_sigma = loaded
    torch.testing.assert_close(loaded_centers, centers)
    torch.testing.assert_close(loaded_mean, feature_mean)
    torch.testing.assert_close(loaded_std, feature_std)
    torch.testing.assert_close(loaded_log_probs, class_log_probs)
    assert loaded_sigma == 0.75


def test_likelihood_ema_artifact_roundtrip_restores_prior(tmp_path):
    model = DitGen(
        cond_dim=16,
        num_classes=3,
        input_size=4,
        in_channels=4,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=4,
        out_channels=4,
        likelihood_component_count=4,
        likelihood_descriptor_dim=0,
        likelihood_sigma=0.0,
    )
    centers = torch.randn(4, 2)
    feature_mean = torch.randn(2)
    feature_std = torch.rand(2).clamp_min(0.1)
    class_log_probs = torch.log_softmax(torch.randn(3, 4), dim=-1)
    model.set_likelihood_prior(
        centers=centers,
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_log_probs=class_log_probs,
        sigma=0.5,
    )
    state = SimpleNamespace(step=5000, ema_decay=0.999, ema_model=model)
    save_params_ema_artifact(
        state,
        workdir=str(tmp_path),
        kind="gen",
        model_config=model.model_config,
    )

    loaded_model, params, metadata = load_generator_model_and_params(str(tmp_path))
    loaded_model.resize_likelihood_prior_from_state_dict(params)
    loaded_model.load_state_dict(params, strict=False)

    assert metadata["step"] == 5000
    assert metadata["model_config"]["likelihood_descriptor_dim"] == 2
    assert metadata["model_config"]["likelihood_sigma"] == 0.5
    assert loaded_model.has_likelihood_prior()
    assert loaded_model.likelihood_sigma == 0.5
    torch.testing.assert_close(loaded_model.likelihood_centers, centers)
    torch.testing.assert_close(loaded_model.likelihood_feature_mean, feature_mean)
    torch.testing.assert_close(loaded_model.likelihood_feature_std, feature_std)
    torch.testing.assert_close(loaded_model.likelihood_class_logits, class_log_probs)
