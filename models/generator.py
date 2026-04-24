from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.env import HF_REPO_ID, HF_ROOT
from utils.misc import sanitize_model_config


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    embed_dim_half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def sincos_init(embed_dim, num_patches):
    grid_size = int(math.sqrt(num_patches))
    pe = get_2d_sincos_pos_embed(embed_dim, grid_size)
    return pe.float().unsqueeze(0)

# PORT FROM https://github.com/facebookresearch/DiT/blob/main/models.py

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos().view(1, max_seq_len, 1, -1), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().view(1, max_seq_len, 1, -1), persistent=False)

    def forward(self, q, k):
        seq_len = q.size(1)
        cos = self.cos_cached[:, :seq_len].to(q.dtype)
        sin = self.sin_cached[:, :seq_len].to(q.dtype)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_out, k_out


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=True)

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
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.use_rmsnorm = use_rmsnorm
        self.use_rope = use_rope
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        head_dim = dim // num_heads
        self.rope = RotaryEmbedding(head_dim) if use_rope else None
        if qk_norm:
            if use_rmsnorm:
                self.q_norm = nn.RMSNorm(head_dim)
                self.k_norm = nn.RMSNorm(head_dim)
            else:
                self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, return_qk=False):
        bsz, seqlen, dim = x.shape
        head_dim = dim // self.num_heads

        qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if self.qk_norm:
            q = F.rms_norm(q, self.q_norm.normalized_shape, weight=self.q_norm.weight.to(q.dtype), eps=self.q_norm.eps)
            k = F.rms_norm(k, self.k_norm.normalized_shape, weight=self.k_norm.weight.to(k.dtype), eps=self.k_norm.eps)
        if self.use_rope:
            q, k = self.rope(q, k)

        qk = (q, k) if return_qk else None

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.permute(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        out = self.proj(out)
        if self.proj_drop > 0:
            out = F.dropout(out, p=self.proj_drop, training=self.training)
        return out, qk


class StandardMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim, bias=True)
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        if use_rmsnorm:
            self.norm1 = nn.RMSNorm(hidden_size)
            self.norm2 = nn.RMSNorm(hidden_size)
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
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            hid_size = int(2 / 3 * mlp_hidden_dim)
            hid_size = (hid_size + 31) // 32 * 32
            self.mlp = SwiGLUFFN(hidden_size, hid_size)
        else:
            self.mlp = StandardMLP(hidden_size, mlp_hidden_dim)

        adaln_linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        nn.init.zeros_(adaln_linear.weight)
        nn.init.zeros_(adaln_linear.bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), adaln_linear)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm = modulate(F.rms_norm(x, self.norm1.normalized_shape, weight=self.norm1.weight.to(x.dtype), eps=self.norm1.eps), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)[0]

        x_norm = modulate(F.rms_norm(x, self.norm2.normalized_shape, weight=self.norm2.weight.to(x.dtype), eps=self.norm2.eps), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = nn.RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        adaln_linear = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        nn.init.zeros_(adaln_linear.weight)
        nn.init.zeros_(adaln_linear.bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), adaln_linear)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(F.rms_norm(x, self.norm_final.normalized_shape, weight=self.norm_final.weight.to(x.dtype), eps=self.norm_final.eps), shift, scale)
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
        checkpointing: bool = False,
    ):
        super().__init__()
        self.checkpointing = checkpointing
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

        num_patches = (input_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        self.patch_embed = nn.Linear(patch_dim, hidden_size, bias=True)
        self.pos_embed = nn.Parameter(sincos_init(hidden_size, num_patches))

        if n_cls_tokens > 0:
            cond_in = cond_dim if cond_dim is not None else hidden_size
            self.cls_proj = nn.Linear(cond_in, hidden_size, bias=True)
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
                    use_rope=use_rope,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            use_rmsnorm=use_rmsnorm,
        )

    def forward(self, x, c):
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
            if self.checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)

        x = self.final_layer(x, c)
        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens :, :]

        x = x.reshape(bsz, grid_h, grid_w, p, p, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(bsz, self.input_size, self.input_size, self.out_channels)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        l1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        nn.init.normal_(l1.weight, std=0.02)
        l2 = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.normal_(l2.weight, std=0.02)
        self.mlp = nn.Sequential(l1, nn.SiLU(), l2)

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
        checkpointing: bool = False,
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
        self.checkpointing = checkpointing
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
            "checkpointing": checkpointing,
        }

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

        self.cfg_embedder = TimestepEmbedder(cond_dim)
        self.cfg_norm = nn.RMSNorm(cond_dim)

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
            checkpointing=checkpointing,
        )

    def generate_image(self, x, cond):
        return self.model(x, cond)

    def c_cfg_noise_to_cond(self, c, cfg_scale, noise_labels):
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
        cfg_scale_t = self.cfg_embedder(cfg_scale_t)
        cfg_scale_t_normalized = F.rms_norm(cfg_scale_t, self.cfg_norm.normalized_shape, weight=self.cfg_norm.weight.to(cfg_scale_t.dtype), eps=self.cfg_norm.eps)
        cfg_scale_t = cfg_scale_t_normalized.to(cond.dtype)
        cond = cond + cfg_scale_t * 0.02

        return cond

    def forward(self, c, cfg_scale=1.0, temp=1.0):
        bsz = c.shape[0]
        device = c.device

        x = torch.randn((bsz, self.input_size, self.input_size, self.in_channels), device=device, dtype=torch.float32)
        x = x * temp

        noise_labels = torch.randint(
            low=0,
            high=max(1, self.noise_classes),
            size=(bsz, max(1, self.noise_coords)),
            device=device,
        )
        cond = self.c_cfg_noise_to_cond(c, cfg_scale, noise_labels)
        samples = self.generate_image(x, cond)

        noise_dict = {
            "x": x,
            "noise_labels": noise_labels,
        }
        return {
            "samples": samples,
            "noise": noise_dict,
        }


def build_generator_from_config(model_config: Dict[str, Any]) -> DitGen:
    cfg = dict(sanitize_model_config(model_config))
    return DitGen(**cfg)


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
