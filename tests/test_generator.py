import torch

from models.generator import DitGen, RMSNorm


def test_generator_uses_jax_compatible_precision_config():
    model = DitGen(
        cond_dim=64,
        input_size=8,
        in_channels=4,
        patch_size=2,
        hidden_size=64,
        depth=1,
        num_heads=4,
        out_channels=4,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        n_cls_tokens=2,
        noise_classes=4,
        noise_coords=2,
        use_bf16=True,
        attn_fp32=True,
    )

    assert model.use_bf16 is True
    assert model.attn_fp32 is True
    rms_norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
    assert rms_norms
    assert all(module.eps == 1e-6 for module in rms_norms)

    out = model(torch.tensor([1, 2]), cfg_scale=torch.tensor([1.0, 2.0]))["samples"]
    assert out.shape == (2, 8, 8, 4)
    assert torch.isfinite(out).all()
