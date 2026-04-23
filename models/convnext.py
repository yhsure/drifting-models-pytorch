"""ConvNeXt feature model (PyTorch)."""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.logging import log_for_0


def safe_std(x, axis, eps=1e-6):
    x32 = x.float()
    mean = x32.mean(dim=axis, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=axis, keepdim=False)
    var = torch.clamp(var, min=0.0)
    return torch.sqrt(var + eps)


class ConvNextLayerNorm(nn.Module):
    """LayerNorm on the last channel for NHWC tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_dtype = x.dtype
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight * x + self.bias
        return x.to(old_dtype)


class ConvNextGRN(nn.Module):
    """Global Response Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_dtype = x.dtype
        gx = torch.sqrt((x**2).sum(dim=(1, 2), keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return (self.gamma * (x * nx) + self.beta + x).to(old_dtype)


class ConvNextBlock(nn.Module):
    """ConvNeXtV2 residual block in NHWC."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.grn = ConvNextGRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.permute(0, 3, 1, 2)
        y = self.dwconv(y)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = F.gelu(y, approximate="none")
        y = self.grn(y)
        y = self.pwconv2(y)
        return residual + y


class ConvNextV2(nn.Module):
    def __init__(self, model_name: str = "base", dtype: torch.dtype = torch.float32):
        super().__init__()
        if model_name == "base":
            load_name = "facebook/convnextv2-base-22k-224"
        elif model_name == "tiny":
            load_name = "facebook/convnextv2-tiny-22k-224"
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        from transformers import ConvNextV2Model

        self.model = ConvNextV2Model.from_pretrained(load_name)
        self.dtype = dtype
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _normalize(self, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        y = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True) + 1e-3)
        return y

    def get_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        outs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outs.hidden_states
        feature_dict: dict[str, torch.Tensor] = {}

        for i, feat in enumerate(hidden_states[1:5]):
            feat = feat.permute(0, 2, 3, 1)
            x_normed = self._normalize(feat)
            if i > 0:
                feature_dict[f"convenxt_stage_{i}"] = rearrange(x_normed, "b h w c -> b (h w) c")
            feature_dict[f"convenxt_stage_{i}_mean"] = x_normed.mean(dim=(1, 2), keepdim=False)[:, None, :]
            feature_dict[f"convenxt_stage_{i}_std"] = safe_std(rearrange(x_normed, "b h w c -> b (h w) c"), axis=1)[:, None, :]

        last = hidden_states[-1].permute(0, 2, 3, 1)
        feature_dict["global_mean"] = last.mean(dim=(1, 2), keepdim=False)[:, None, :]
        feature_dict["global_std"] = safe_std(rearrange(self._normalize(last), "b h w c -> b (h w) c"), axis=1)[:, None, :]
        return feature_dict

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        outs = self.model(pixel_values=x)
        return outs.pooler_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def ConvNextBase(**kwargs):
    return ConvNextV2(model_name="base", **kwargs)


def ConvNextTiny(**kwargs):
    return ConvNextV2(model_name="tiny", **kwargs)


def convert_weights_to_jax(jax_params: dict, module_pt, hf: bool = False):
    """Compatibility alias retained for old callsites.

    In the PyTorch port this returns a key-normalized state dict-like mapping.
    """

    del hf
    pt_params = dict(module_pt)
    out = {}
    for key in list(pt_params.keys()):
        mapped = re.sub(r"^convnextv2\.", "", key)
        out[mapped] = pt_params[key]
    return out if out else jax_params


def load_convnext_torch_model(model_name: str = "base", use_bf16: bool = False):
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = ConvNextV2(model_name=model_name, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log_for_0("Loaded ConvNeXt model %s", model_name)
    return model, model.state_dict()


def load_convnext_jax_model(model_name: str = "base", use_bf16: bool = False):
    """Compatibility alias for JAX-era callsites."""

    return load_convnext_torch_model(model_name=model_name, use_bf16=use_bf16)
