from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.env import HF_REPO_ID, HF_ROOT


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    embed_dim_half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def sincos_init(embed_dim, num_patches):
    grid_size = int(np.sqrt(num_patches))
    pe = get_2d_sincos_pos_embed(embed_dim, grid_size)
    return torch.from_numpy(pe).float().unsqueeze(0)


class TorchLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
        dtype: Any = torch.float32,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dtype = dtype

        if weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.linear.weight)
        elif weight_init == "zeros":
            nn.init.zeros_(self.linear.weight)
        elif weight_init == "normal":
            nn.init.normal_(self.linear.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.linear.weight)

        if self.linear.bias is not None:
            if bias_init == "zeros":
                nn.init.zeros_(self.linear.bias)
            else:
                nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x.to(self.linear.weight.dtype))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        if self.elementwise_affine:
            normed = normed * self.weight
        return normed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(q, k, dtype=torch.float32):
    bsz, seqlen, nheads, dim = q.shape
    half_dim = dim // 2
    freqs = (1.0 / (10000 ** (torch.arange(0, half_dim, device=q.device, dtype=dtype) / half_dim))).to(dtype)
    t = torch.arange(seqlen, device=q.device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = torch.cos(emb)[None, :, None, :]
    sin = torch.sin(emb)[None, :, None, :]

    def rotate_half(x):
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _dynamo_disabled(fn):
    if hasattr(torch, "_dynamo"):
        return torch._dynamo.disable(fn)
    return fn


@_dynamo_disabled
def _likelihood_embedding_condition(
    component_embed: nn.Embedding,
    likelihood_components: torch.Tensor,
    likelihood_weights: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    if likelihood_components.ndim == 1:
        out = component_embed(likelihood_components)
    else:
        embeds = component_embed(likelihood_components)
        if likelihood_weights is None:
            weights = torch.full(
                likelihood_components.shape,
                1.0 / likelihood_components.shape[1],
                device=likelihood_components.device,
                dtype=embeds.dtype,
            )
        else:
            weights = likelihood_weights.to(device=likelihood_components.device, dtype=embeds.dtype)
        out = (embeds * weights.unsqueeze(-1)).sum(dim=1)
    return out * float(scale)


def _temperature_tensor(temperature, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(temperature, device=reference.device, dtype=reference.dtype).clamp_min(1e-6)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: Any = torch.float32):
        super().__init__()
        self.w1 = TorchLinear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.w3 = TorchLinear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.w2 = TorchLinear(intermediate_size, hidden_size, bias=True, dtype=dtype)

    def forward(self, x):
        w1 = self.w1(x)
        w3 = self.w3(x)
        out = F.silu(w1) * w3
        return self.w2(out)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_fp32: bool = True,
        dtype: Any = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.use_rmsnorm = use_rmsnorm
        self.use_rope = use_rope
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.attn_fp32 = attn_fp32

        self.qkv = TorchLinear(dim, dim * 3, bias=qkv_bias, dtype=dtype)
        self.proj = TorchLinear(dim, dim, bias=True, dtype=dtype)
        head_dim = dim // num_heads
        if qk_norm:
            if use_rmsnorm:
                self.q_norm = RMSNorm(head_dim)
                self.k_norm = RMSNorm(head_dim)
            else:
                self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, deterministic=True, return_qk=False):
        bsz, seqlen, dim = x.shape
        head_dim = dim // self.num_heads

        qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            rope_dtype = torch.float32 if self.attn_fp32 else q.dtype
            q, k = apply_rope(q, k, dtype=rope_dtype)

        qk = (q, k) if return_qk else None

        if self.attn_fp32:
            q = q.float()
            k = k.float()
            v = v.float()
        else:
            q = q.to(x.dtype)
            k = k.to(x.dtype)
            v = v.to(x.dtype)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if not deterministic else 0.0,
            scale=head_dim ** -0.5,
        )
        out = out.permute(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        out = self.proj(out)
        if self.proj_drop > 0:
            out = F.dropout(out, p=self.proj_drop, training=not deterministic)
        return out, qk


class StandardMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int, dtype: Any = torch.float32):
        super().__init__()
        self.fc1 = TorchLinear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype)
        self.fc2 = TorchLinear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="none"))


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        cond_dim: Optional[int] = None,
        use_rope: bool = False,
        attn_fp32: bool = True,
        dtype: Any = torch.float32,
    ):
        super().__init__()
        del cond_dim
        self.hidden_size = hidden_size
        self.dtype = dtype

        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
            attn_fp32=attn_fp32,
            dtype=dtype,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            hid_size = int(2 / 3 * mlp_hidden_dim)
            hid_size = (hid_size + 31) // 32 * 32
            self.mlp = SwiGLUFFN(hidden_size, hid_size, dtype=dtype)
        else:
            self.mlp = StandardMLP(hidden_size, mlp_hidden_dim, dtype=dtype)

        self.adaln = nn.Sequential(
            nn.SiLU(),
            TorchLinear(hidden_size, 6 * hidden_size, bias=True, weight_init="zeros", bias_init="zeros", dtype=torch.float32),
        )

    def forward(self, x, c, deterministic=True):
        chunks = self.adaln(c.float()).to(x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(chunks, 6, dim=1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, deterministic=deterministic)[0]

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = False,
        cond_dim: Optional[int] = None,
        dtype: Any = torch.float32,
    ):
        super().__init__()
        del cond_dim
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            TorchLinear(hidden_size, 2 * hidden_size, bias=True, weight_init="zeros", bias_init="zeros", dtype=torch.float32),
        )
        self.linear = TorchLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
            dtype=dtype,
        )

    def forward(self, x, c):
        shift, scale = torch.chunk(self.adaln(c.float()).to(x.dtype), 2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 32,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        cond_dim: Optional[int] = None,
        n_cls_tokens: int = 0,
        attn_fp32: bool = True,
        dtype: Any = torch.float32,
        use_remat: bool = False,
    ):
        super().__init__()
        del use_remat
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.use_qknorm = use_qknorm
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.cond_dim = cond_dim
        self.n_cls_tokens = n_cls_tokens
        self.attn_fp32 = attn_fp32
        self.dtype = dtype

        num_patches = (input_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        self.patch_embed = TorchLinear(patch_dim, hidden_size, bias=True, dtype=dtype)
        self.pos_embed = nn.Parameter(sincos_init(hidden_size, num_patches))

        if n_cls_tokens > 0:
            cond_in = cond_dim if cond_dim is not None else hidden_size
            self.cls_proj = TorchLinear(cond_in, hidden_size, bias=True, dtype=dtype)
            self.cls_embed = nn.Parameter(torch.randn(1, n_cls_tokens, hidden_size) * 0.02)
        else:
            self.cls_proj = None
            self.register_parameter("cls_embed", None)

        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    cond_dim=cond_dim,
                    use_rope=use_rope,
                    attn_fp32=attn_fp32,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            use_rmsnorm=use_rmsnorm,
            cond_dim=cond_dim,
            dtype=dtype,
        )

    def forward(self, x, c, deterministic=True):
        bsz, h, w, channels = x.shape
        p = self.patch_size

        target_grid = self.input_size // p
        num_patches = target_grid * target_grid
        effective_p = h // target_grid
        grid_h, grid_w = target_grid, target_grid

        x = x.reshape(bsz, grid_h, effective_p, grid_w, effective_p, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(bsz, num_patches, effective_p * effective_p * channels)
        x = self.patch_embed(x)

        pos = self.pos_embed
        if pos.shape[1] != num_patches:
            pos = sincos_init(self.hidden_size, num_patches).to(device=x.device, dtype=x.dtype)
        x = (x + pos).to(x.dtype)

        if self.n_cls_tokens > 0:
            c_tokens = self.cls_proj(c).unsqueeze(1).repeat(1, self.n_cls_tokens, 1)
            c_tokens = c_tokens + self.cls_embed
            x = torch.cat([c_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x, c, deterministic=deterministic)

        x = self.final_layer(x, c)
        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens :, :]

        x = x.reshape(bsz, grid_h, grid_w, p, p, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(bsz, self.input_size, self.input_size, self.out_channels)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, dtype: Any = torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            TorchLinear(frequency_embedding_size, hidden_size, bias=True, weight_init="normal", dtype=dtype),
            nn.SiLU(),
            TorchLinear(hidden_size, hidden_size, bias=True, weight_init="normal", dtype=dtype),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class DitGen(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        num_classes: int = 1001,
        noise_classes: int = 0,
        noise_coords: int = 1,
        input_size: int = 32,
        in_channels: int = 3,
        n_cls_tokens: int = 0,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 3,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        use_bf16: bool = False,
        attn_fp32: bool = True,
        use_remat: bool = False,
        likelihood_component_count: int = 0,
        likelihood_descriptor_dim: int = 0,
        likelihood_sigma: float = 1.0,
        likelihood_conditioning: str = "none",
        likelihood_topk: int = 1,
        likelihood_embedding_scale: float = 1.0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords
        self.input_size = input_size
        self.in_channels = in_channels
        self.n_cls_tokens = n_cls_tokens
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.use_qknorm = use_qknorm
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_bf16 = use_bf16
        self.attn_fp32 = attn_fp32
        self.use_remat = use_remat
        self.likelihood_component_count = int(likelihood_component_count)
        self.likelihood_descriptor_dim = int(likelihood_descriptor_dim)
        self.likelihood_sigma = float(likelihood_sigma)
        self.likelihood_conditioning = str(likelihood_conditioning)
        self.likelihood_topk = int(likelihood_topk)
        self.likelihood_embedding_scale = float(likelihood_embedding_scale)
        self.model_config = {
            "cond_dim": cond_dim,
            "num_classes": num_classes,
            "noise_classes": noise_classes,
            "noise_coords": noise_coords,
            "input_size": input_size,
            "in_channels": in_channels,
            "n_cls_tokens": n_cls_tokens,
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "out_channels": out_channels,
            "use_qknorm": use_qknorm,
            "use_swiglu": use_swiglu,
            "use_rope": use_rope,
            "use_rmsnorm": use_rmsnorm,
            "use_bf16": use_bf16,
            "attn_fp32": attn_fp32,
            "use_remat": use_remat,
            "likelihood_component_count": self.likelihood_component_count,
            "likelihood_descriptor_dim": self.likelihood_descriptor_dim,
            "likelihood_sigma": self.likelihood_sigma,
            "likelihood_conditioning": self.likelihood_conditioning,
            "likelihood_topk": self.likelihood_topk,
            "likelihood_embedding_scale": self.likelihood_embedding_scale,
        }

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        self.class_embed = nn.Embedding(num_classes, cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        if noise_classes > 0:
            self.noise_embeds = nn.ModuleList()
            for _ in range(noise_coords):
                emb = nn.Embedding(noise_classes, cond_dim)
                nn.init.normal_(emb.weight, std=0.02)
                self.noise_embeds.append(emb)
        else:
            self.noise_embeds = nn.ModuleList()

        self.cfg_embedder = TimestepEmbedder(cond_dim, dtype=dtype)
        self.cfg_norm = RMSNorm(cond_dim)
        if self.likelihood_component_count > 0:
            self.likelihood_component_embed = nn.Embedding(self.likelihood_component_count + 1, cond_dim)
            nn.init.normal_(self.likelihood_component_embed.weight, std=0.02)
            nn.init.zeros_(self.likelihood_component_embed.weight[self.likelihood_component_count])
            self.likelihood_class_logits = nn.Parameter(
                torch.zeros(num_classes, self.likelihood_component_count, dtype=torch.float32)
            )
            centers = torch.zeros(
                self.likelihood_component_count,
                max(0, self.likelihood_descriptor_dim),
                dtype=torch.float32,
            )
            self.register_buffer("likelihood_centers", centers, persistent=True)
            self.register_buffer(
                "likelihood_feature_mean",
                torch.zeros(max(0, self.likelihood_descriptor_dim), dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "likelihood_feature_std",
                torch.ones(max(0, self.likelihood_descriptor_dim), dtype=torch.float32),
                persistent=True,
            )
        else:
            self.likelihood_component_embed = None
            self.register_parameter("likelihood_class_logits", None)
            self.register_buffer("likelihood_centers", torch.empty(0, 0, dtype=torch.float32), persistent=True)
            self.register_buffer("likelihood_feature_mean", torch.empty(0, dtype=torch.float32), persistent=True)
            self.register_buffer("likelihood_feature_std", torch.empty(0, dtype=torch.float32), persistent=True)

        self.model = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            cond_dim=cond_dim,
            n_cls_tokens=n_cls_tokens,
            attn_fp32=attn_fp32,
            dtype=dtype,
            use_remat=use_remat,
        )

    def dummy_input(self):
        return {
            "c": torch.ones(1, dtype=torch.long),
            "cfg_scale": 1.0,
            "temp": 1.0,
            "deterministic": True,
        }

    def rng_keys(self):
        return ["noise"]

    def generate_image(self, x, cond, deterministic=True):
        return self.model(x, cond, deterministic=deterministic)

    @property
    def likelihood_null_index(self) -> int:
        return max(0, self.likelihood_component_count)

    def has_likelihood_prior(self) -> bool:
        return (
            self.likelihood_component_count > 0
            and self.likelihood_centers.ndim == 2
            and self.likelihood_centers.shape[0] == self.likelihood_component_count
            and self.likelihood_centers.shape[1] > 0
        )

    def resize_likelihood_prior_from_state_dict(self, state_dict: Dict[str, Any]) -> None:
        centers = state_dict.get("likelihood_centers")
        if centers is None or not isinstance(centers, torch.Tensor) or centers.ndim != 2:
            return
        if centers.shape == self.likelihood_centers.shape:
            return
        component_count, descriptor_dim = int(centers.shape[0]), int(centers.shape[1])
        if component_count != self.likelihood_component_count:
            raise ValueError(
                "Checkpoint likelihood component count does not match model config: "
                f"{component_count} != {self.likelihood_component_count}"
            )
        device = self.class_embed.weight.device
        self.likelihood_descriptor_dim = descriptor_dim
        self.model_config["likelihood_descriptor_dim"] = descriptor_dim
        self.likelihood_centers = torch.zeros(component_count, descriptor_dim, device=device, dtype=torch.float32)
        self.likelihood_feature_mean = torch.zeros(descriptor_dim, device=device, dtype=torch.float32)
        self.likelihood_feature_std = torch.ones(descriptor_dim, device=device, dtype=torch.float32)

    def set_likelihood_prior(
        self,
        *,
        centers: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        class_log_probs: torch.Tensor,
        sigma: float,
    ) -> None:
        if self.likelihood_component_count <= 0:
            raise ValueError("This generator was built without likelihood components.")
        centers = centers.detach().float()
        feature_mean = feature_mean.detach().float()
        feature_std = feature_std.detach().float().clamp_min(1e-6)
        class_log_probs = class_log_probs.detach().float()
        if centers.ndim != 2:
            raise ValueError(f"centers must be 2D, got shape={tuple(centers.shape)}")
        if centers.shape[0] != self.likelihood_component_count:
            raise ValueError(
                f"Expected {self.likelihood_component_count} centers, got {centers.shape[0]}"
            )
        if feature_mean.shape != (centers.shape[1],) or feature_std.shape != (centers.shape[1],):
            raise ValueError("feature_mean and feature_std must match center descriptor dimension.")
        if class_log_probs.shape != (self.num_classes, self.likelihood_component_count):
            raise ValueError(
                "class_log_probs must have shape "
                f"({self.num_classes}, {self.likelihood_component_count}), got {tuple(class_log_probs.shape)}"
            )
        device = self.class_embed.weight.device
        self.likelihood_descriptor_dim = int(centers.shape[1])
        self.likelihood_sigma = float(sigma)
        self.model_config["likelihood_descriptor_dim"] = self.likelihood_descriptor_dim
        self.model_config["likelihood_sigma"] = self.likelihood_sigma
        self.likelihood_centers = centers.to(device=device)
        self.likelihood_feature_mean = feature_mean.to(device=device)
        self.likelihood_feature_std = feature_std.to(device=device)
        if self.likelihood_class_logits is None:
            raise ValueError("likelihood_class_logits is missing.")
        self.likelihood_class_logits.data.copy_(class_log_probs.to(device=device))

    def standardize_likelihood_descriptors(self, descriptors: torch.Tensor) -> torch.Tensor:
        if not self.has_likelihood_prior():
            raise ValueError("Likelihood prior has not been initialized.")
        mean = self.likelihood_feature_mean.to(device=descriptors.device, dtype=descriptors.dtype)
        std = self.likelihood_feature_std.to(device=descriptors.device, dtype=descriptors.dtype).clamp_min(1e-6)
        return (descriptors - mean) / std

    def likelihood_logits_for_descriptors(self, descriptors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        v = self.standardize_likelihood_descriptors(descriptors.float())
        centers = self.likelihood_centers.to(device=v.device, dtype=v.dtype)
        dist_sq = torch.cdist(v, centers).square()
        labels = labels.to(device=v.device, dtype=torch.long).clamp(0, self.num_classes - 1)
        log_pi = F.log_softmax(self.likelihood_class_logits[labels].to(device=v.device, dtype=v.dtype), dim=-1)
        sigma_sq = max(self.likelihood_sigma, 1e-6) ** 2
        return log_pi - dist_sq / (2.0 * sigma_sq)

    def likelihood_nll_and_responsibilities(
        self,
        descriptors: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.likelihood_logits_for_descriptors(descriptors, labels)
        dim = max(1, int(self.likelihood_centers.shape[1]))
        const = 0.5 * dim * math.log(2.0 * math.pi * max(self.likelihood_sigma, 1e-6) ** 2)
        nll = const - torch.logsumexp(logits, dim=-1)
        resp = F.softmax(logits / _temperature_tensor(temperature, logits), dim=-1)
        return nll.mean(), resp

    @torch.no_grad()
    def likelihood_topk_responsibilities(
        self,
        descriptors: torch.Tensor,
        labels: torch.Tensor,
        *,
        temperature: float = 1.0,
        topk: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        logits = self.likelihood_logits_for_descriptors(descriptors, labels)
        resp = F.softmax(logits / _temperature_tensor(temperature, logits), dim=-1)
        k = max(1, min(int(topk), resp.shape[-1]))
        weights, components = torch.topk(resp, k=k, dim=-1)
        topk_mass = weights.sum(dim=-1).clamp_min(1e-8)
        weights = weights / topk_mass[:, None]
        entropy = -(resp * resp.clamp_min(1e-8).log()).sum(dim=-1)
        info = {
            "posterior_entropy": float(entropy.mean().detach().cpu().item()),
            "topk_mass": float(topk_mass.mean().detach().cpu().item()),
            "topk": float(k),
        }
        return components, weights, info

    @torch.no_grad()
    def likelihood_assign(self, descriptors: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        components, _, _ = self.likelihood_topk_responsibilities(
            descriptors,
            labels,
            temperature=temperature,
            topk=1,
        )
        return components[:, 0]

    @torch.no_grad()
    def sample_likelihood_components(
        self,
        labels: torch.Tensor,
        *,
        conditioning: str | None = None,
        topk: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.likelihood_component_count <= 0:
            return None, None
        mode = (conditioning or self.likelihood_conditioning or "hard").lower()
        if mode in {"none", "off", "false"}:
            return None, None
        k = int(topk if topk is not None else self.likelihood_topk)
        if mode in {"hard", "top1"}:
            k = 1
        k = max(1, min(k, self.likelihood_component_count))
        labels = labels.to(device=self.likelihood_class_logits.device, dtype=torch.long).clamp(0, self.num_classes - 1)
        logits = self.likelihood_class_logits[labels] / _temperature_tensor(temperature, self.likelihood_class_logits)
        probs = F.softmax(logits, dim=-1)
        components = torch.multinomial(probs, num_samples=k, replacement=False)
        weights = probs.gather(1, components).clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        if mode in {"hard", "top1"}:
            return components[:, 0], None
        return components, weights

    def _likelihood_condition(
        self,
        labels: torch.Tensor,
        likelihood_components: torch.Tensor | None = None,
        likelihood_weights: torch.Tensor | None = None,
        likelihood_conditioning: str | None = None,
        likelihood_topk: int | None = None,
        likelihood_temperature: float = 1.0,
    ) -> torch.Tensor | None:
        if self.likelihood_component_count <= 0 or self.likelihood_component_embed is None:
            return None
        mode = (likelihood_conditioning or self.likelihood_conditioning or "hard").lower()
        if mode in {"none", "off", "false"}:
            return None
        if likelihood_components is None:
            likelihood_components, likelihood_weights = self.sample_likelihood_components(
                labels,
                conditioning=mode,
                topk=likelihood_topk,
                temperature=likelihood_temperature,
            )
        if likelihood_components is None:
            return None
        likelihood_components = likelihood_components.to(device=labels.device, dtype=torch.long)
        return _likelihood_embedding_condition(
            self.likelihood_component_embed,
            likelihood_components,
            likelihood_weights,
            self.likelihood_embedding_scale,
        )

    def c_cfg_noise_to_cond(
        self,
        c,
        cfg_scale,
        noise_labels,
        likelihood_components: torch.Tensor | None = None,
        likelihood_weights: torch.Tensor | None = None,
        likelihood_conditioning: str | None = None,
        likelihood_topk: int | None = None,
        likelihood_temperature: float = 1.0,
    ):
        bsz = c.shape[0]
        cond = self.class_embed(c)
        if self.noise_classes > 0:
            for i in range(self.noise_coords):
                cond = cond + self.noise_embeds[i](noise_labels[:, i])

        if isinstance(cfg_scale, (float, int)):
            cfg_scale_t = torch.full((bsz,), float(cfg_scale), device=c.device, dtype=torch.float32)
        else:
            cfg_scale_t = torch.as_tensor(cfg_scale, device=c.device, dtype=torch.float32)
            if cfg_scale_t.ndim == 0:
                cfg_scale_t = cfg_scale_t.unsqueeze(0).repeat(bsz)
        cfg_scale_t = self.cfg_norm(self.cfg_embedder(cfg_scale_t))
        cond = cond + cfg_scale_t * 0.02
        likelihood_cond = self._likelihood_condition(
            c,
            likelihood_components=likelihood_components,
            likelihood_weights=likelihood_weights,
            likelihood_conditioning=likelihood_conditioning,
            likelihood_topk=likelihood_topk,
            likelihood_temperature=likelihood_temperature,
        )
        if likelihood_cond is not None:
            cond = cond + likelihood_cond

        if self.use_bf16:
            cond = cond.to(torch.bfloat16)
        return cond

    def forward(
        self,
        c,
        cfg_scale=1.0,
        temp=1.0,
        deterministic=True,
        train=False,
        likelihood_components: torch.Tensor | None = None,
        likelihood_weights: torch.Tensor | None = None,
        likelihood_conditioning: str | None = None,
        likelihood_topk: int | None = None,
        likelihood_temperature: float = 1.0,
        likelihood_loss_descriptors: torch.Tensor | None = None,
        likelihood_loss_labels: torch.Tensor | None = None,
        likelihood_resp_temperature: float = 1.0,
    ):
        del train
        bsz = c.shape[0]
        device = c.device

        x = torch.randn((bsz, self.input_size, self.input_size, self.in_channels), device=device, dtype=torch.float32)
        x = x * float(temp)
        if self.use_bf16:
            x = x.to(torch.bfloat16)

        noise_labels = torch.randint(
            low=0,
            high=max(1, self.noise_classes),
            size=(bsz, max(1, self.noise_coords)),
            device=device,
        )
        cond = self.c_cfg_noise_to_cond(
            c,
            cfg_scale,
            noise_labels,
            likelihood_components=likelihood_components,
            likelihood_weights=likelihood_weights,
            likelihood_conditioning=likelihood_conditioning,
            likelihood_topk=likelihood_topk,
            likelihood_temperature=likelihood_temperature,
        )
        samples = self.generate_image(x, cond, deterministic=deterministic)

        noise_dict = {
            "x": x,
            "noise_labels": noise_labels,
            "likelihood_components": likelihood_components,
            "likelihood_weights": likelihood_weights,
        }
        out = {
            "samples": samples,
            "noise": noise_dict,
        }
        if likelihood_loss_descriptors is not None:
            loss_labels = c if likelihood_loss_labels is None else likelihood_loss_labels
            nll, resp = self.likelihood_nll_and_responsibilities(
                likelihood_loss_descriptors,
                loss_labels,
                temperature=likelihood_resp_temperature,
            )
            out["likelihood_nll"] = nll
            out["likelihood_resp"] = resp
        return out


def build_generator_from_config(model_config: Dict[str, Any]) -> DitGen:
    return DitGen(**dict(model_config))


def load_hf(
    name: str,
    *,
    dir: str = HF_ROOT,
):
    from models.hf import load_generator_torch

    return load_generator_torch(
        name=name,
        repo_id=HF_REPO_ID,
        prefix=None,
        output_root=dir,
    )
