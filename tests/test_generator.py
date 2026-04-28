import torch

from models.generator import Attention, DitGen, RMSNorm, apply_rope


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


def test_attention_sdpa_matches_manual_scaled_attention():
    torch.manual_seed(0)
    attn = Attention(dim=32, num_heads=4, qkv_bias=True, qk_norm=True, use_rmsnorm=True, use_rope=True, attn_fp32=True)
    attn.eval()
    x = torch.randn(2, 10, 32)

    with torch.no_grad():
        out, _ = attn(x)

        bsz, seqlen, dim = x.shape
        head_dim = dim // attn.num_heads
        qkv = attn.qkv(x).reshape(bsz, seqlen, 3, attn.num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = apply_rope(q, k, dtype=torch.float32)
        q = (q.float() * (head_dim ** -0.5)).permute(0, 2, 1, 3)
        k = k.float().permute(0, 2, 1, 3)
        v = v.float().permute(0, 2, 1, 3)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        manual = torch.matmul(weights, v).permute(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        manual = attn.proj(manual)

    torch.testing.assert_close(out, manual, rtol=1e-5, atol=1e-6)
