"""Self-contained PyTorch MAE-ResNet (no model indirection)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.env import HF_REPO_ID, HF_ROOT
from utils.init_util import load_init_entry
from utils.misc import maybe_compile, sanitize_model_config


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


class _BasicBlock(nn.Module):
    def __init__(
        self,
        filters: int,
        in_channels: Optional[int] = None,
        stride: int = 1,
        gn_max_groups: int = 32,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        in_channels = int(in_channels if in_channels is not None else filters)
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)
        self.drop = nn.Dropout(dropout_prob)

        if stride != 1 or in_channels != filters:
            self.proj_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False)
            self.proj_gn = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)
        else:
            self.proj_conv = None
            self.proj_gn = None

    def forward(self, x: torch.Tensor, *, train: bool) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        y = self.drop(y) if train else y
        y = self.conv2(y)
        y = self.gn2(y)

        if self.proj_conv is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_gn(residual)

        return F.relu(residual + y)


class _ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(base_channels, gn_max_groups), base_channels)

        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()

        ch = base_channels
        in_ch = base_channels
        for stage_idx, num_blocks in enumerate(layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = ch * (2**stage_idx) if stage_idx > 0 else ch
            blocks = []
            blocks.append(
                _BasicBlock(
                    out_ch,
                    in_channels=in_ch,
                    stride=stride,
                    dropout_prob=dropout_prob,
                )
            )
            for _ in range(1, num_blocks):
                blocks.append(_BasicBlock(out_ch, in_channels=out_ch, stride=1, dropout_prob=dropout_prob))
            self.stages.append(nn.ModuleList(blocks))
            self.stage_norms.append(nn.GroupNorm(_choose_gn_groups(out_ch, gn_max_groups), out_ch))
            in_ch = out_ch

    def forward(self, x: torch.Tensor, *, train: bool, return_block_outputs: bool = False):
        feats: Dict[str, torch.Tensor] = {}
        block_outputs: Dict[str, List[torch.Tensor]] = {}

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        feats["conv1"] = x

        for i, blocks in enumerate(self.stages):
            layer_name = f"layer{i + 1}"
            outs: List[torch.Tensor] = []
            for block in blocks:
                x = block(x, train=train)
                outs.append(x)
            block_outputs[layer_name] = outs
            x = self.stage_norms[i](x)
            feats[layer_name] = x

        if return_block_outputs:
            return feats, block_outputs
        return feats


class _ConvGNReLU(nn.Module):
    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(channels, 32), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.concat_norm_fn = nn.GroupNorm(_choose_gn_groups(in_channels, 32), in_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_choose_gn_groups(out_channels, 32), out_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = _ConvGNReLU(out_channels, kernel=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm_fn(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class _UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c1 = base_channels
        c2 = base_channels
        c3 = base_channels * 2
        c4 = base_channels * 4
        c5 = base_channels * 8

        self.bridge = _ConvGNReLU(c5)
        self.up43 = _UpBlock(c5 + c4, c4)
        self.up32 = _UpBlock(c4 + c3, c3)
        self.up21 = _UpBlock(c3 + c2, c2)
        self.up10 = _UpBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


def patch_input(x: torch.Tensor, input_patch_size: int) -> torch.Tensor:
    return rearrange(
        x,
        "b (h1 h2) (w1 w2) c -> b h1 w1 (h2 w2 c)",
        h2=input_patch_size,
        w2=input_patch_size,
    )


def make_patch_mask(x: torch.Tensor, mask_ratio: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    bsz, h, w, _ = x.shape
    nh, nw = h // patch_size, w // patch_size
    noise = torch.rand((bsz, nh, nw), device=x.device, dtype=x.dtype)
    mask = (noise < mask_ratio[:, None, None]).to(x.dtype)
    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return mask[..., None]


def safe_std(x: torch.Tensor, axis, eps: float = 1e-6, keepdims: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=axis, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=axis, keepdim=keepdims)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)


class MAEResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_size: int = 4,
        dropout_prob: float = 0.0,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        input_patch_size: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.patch_size = patch_size
        self.dropout_prob = dropout_prob
        self.layers = tuple(layers)
        self.input_patch_size = input_patch_size

        self.encoder = _ResNetEncoder(
            in_channels=self.in_channels * self.input_patch_size * self.input_patch_size,
            base_channels=self.base_channels,
            layers=self.layers,
            dropout_prob=self.dropout_prob,
        )
        self.decoder = _UNetDecoder(
            base_channels=self.base_channels,
            out_channels=self.in_channels * self.input_patch_size * self.input_patch_size,
        )
        self.fc = nn.Linear(self.base_channels * 8, self.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        train: bool = True,
    ):
        labels = labels.long()

        x = patch_input(x, self.input_patch_size)
        ratio = torch.rand((x.shape[0],), device=x.device, dtype=torch.float32)
        mask_ratio = ratio * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask = make_patch_mask(x, mask_ratio, self.patch_size).to(x.dtype)
        x_in = x * (1.0 - mask)

        feats = self.encoder(x_in.permute(0, 3, 1, 2).float(), train=train)
        top = feats["layer4"]
        pooled = top.mean(dim=(2, 3))
        logits = self.fc(pooled)
        recon = self.decoder(feats).permute(0, 2, 3, 1).to(x.dtype)

        cls_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        mse = (recon - x) ** 2
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss

        metrics = {
            "loss": loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "accuracy": (logits.argmax(dim=-1) == labels).to(x.dtype),
            "mask_ratio": mask.mean(dim=(1, 2, 3)),
        }
        return loss, metrics

    def get_activations(
        self,
        x: torch.Tensor,
        *,
        patch_mean_size: Optional[List[int]] = [2, 4],
        patch_std_size: Optional[List[int]] = [2, 4],
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> Dict[str, torch.Tensor]:
        patch_mean_size = patch_mean_size or []
        patch_std_size = patch_std_size or []

        x = patch_input(x, self.input_patch_size)

        need_blocks = isinstance(every_k_block, (int, float)) and not math.isinf(float(every_k_block)) and every_k_block >= 1
        if need_blocks:
            feats, block_outputs = self.encoder(x.permute(0, 3, 1, 2).float(), train=False, return_block_outputs=True)
        else:
            feats = self.encoder(x.permute(0, 3, 1, 2).float(), train=False)
            block_outputs = {}

        out: Dict[str, torch.Tensor] = {}
        out["norm_x"] = torch.sqrt((x.float() ** 2).mean(dim=(1, 2)) + 1e-6)[:, None, :]

        def _to_bhwc(feat: torch.Tensor) -> torch.Tensor:
            return feat.permute(0, 2, 3, 1)

        def process_feat(name: str, feat_nchw: torch.Tensor) -> None:
            feat = _to_bhwc(feat_nchw)
            bsz, h, w, c = feat.shape
            _ = bsz, c
            out[name] = rearrange(feat, "b h w c -> b (h w) c")
            if use_mean:
                out[f"{name}_mean"] = feat.mean(dim=(1, 2), keepdim=False)[:, None, :]
            if use_std:
                out[f"{name}_std"] = safe_std(feat, axis=(1, 2))[:, None, :]

            for size in patch_mean_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat, "b (h s1) (w s2) c -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_mean_{size}"] = reshaped.mean(dim=2)

            for size in patch_std_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat, "b (h s1) (w s2) c -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_std_{size}"] = safe_std(reshaped, axis=2)

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                lname = f"layer{i}"
                blocks = block_outputs.get(lname, [])
                for blk_idx, feat_i in enumerate(blocks, start=1):
                    if blk_idx % k == 0:
                        process_feat(f"{lname}_blk{blk_idx}", feat_i)

        return out

    def dummy_input(self) -> Dict[str, Any]:
        p = self.input_patch_size
        return {
            "x": torch.zeros((1, 32 * p, 32 * p, self.in_channels), dtype=torch.float32),
            "labels": torch.zeros((1,), dtype=torch.long),
            "lambda_cls": 0.0,
            "mask_ratio_min": 0.75,
            "mask_ratio_max": 0.75,
            "train": False,
        }


def load_mae_hf(
    name: str,
    *,
    dir: str = HF_ROOT,
):
    from models.hf import load_mae_torch

    repo_id = HF_REPO_ID
    prefix = None
    model, params, metadata = load_mae_torch(
        name,
        repo_id=repo_id,
        prefix=prefix,
        output_root=dir,
    )
    model.load_state_dict(params, strict=False)
    return model, model.state_dict(), metadata


def _mae_from_metadata(metadata: Dict[str, Any]) -> MAEResNet:
    model_config = dict(sanitize_model_config(metadata.get("model_config", {}) or {}))
    num_classes = int(model_config.pop("num_classes", 1000))
    return MAEResNet(num_classes=num_classes, **model_config)


# Backward-compatible alias for older import paths.
MAEResNetJAX = MAEResNet


def build_feature_model_and_params(
    path: str = "",
    use_convnext: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_convnext:
        from models.convnext import load_convnext_torch_model

        model, _ = load_convnext_torch_model(model_name="base")
        model.eval()
        return model, {"model": model}

    if not path:
        raise ValueError("`path` is required when use_convnext=False.")

    entry, metadata = load_init_entry("mae", path, hf_cache_dir=HF_ROOT)
    if not metadata:
        raise ValueError(f"MAE artifact is missing metadata required to rebuild the model: {path}")
    feature_model = _mae_from_metadata(metadata).to(device)
    feature_model.load_state_dict(entry, strict=False)
    feature_model.eval()
    for p in feature_model.parameters():
        p.requires_grad = False
    return feature_model, {"model": feature_model}


def build_activation_function(
    mae_path: str = "",
    use_convnext=False,
    use_mae=True,
    postprocess_fn=lambda x: x,
    compile_level: int = 2,
):
    variables = {}
    if use_mae:
        feature_model, feature_params = build_feature_model_and_params(path=mae_path)
        feature_model.get_activations = maybe_compile(feature_model.get_activations, compile_level)
        variables["mae_model"] = feature_model
        variables["mae_params"] = feature_params

    if use_convnext:
        convnext_model, convnext_feature_params = build_feature_model_and_params(
            use_convnext=True,
        )
        convnext_model.get_activations = maybe_compile(convnext_model.get_activations, compile_level)
        variables["convnext_model"] = convnext_model
        variables["convnext_params"] = convnext_feature_params

    def activation_fn(x, convnext_kwargs=dict(), has_scale=False, **kwargs):
        usual_feats = {}
        usual_feats["global"] = x.reshape(x.shape[0], 1, -1)
        if has_scale:
            usual_feats["norm_x"] = torch.sqrt((x.float() ** 2).mean(dim=(1, 2)) + 1e-6)[:, None, :]

        if use_mae:
            mae_feats = variables["mae_model"].get_activations(x, **kwargs)
            usual_feats = {**usual_feats, **mae_feats}

        if use_convnext:
            x_post = postprocess_fn(x)
            x_post = x_post.permute(0, 2, 3, 1)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x_post.device, dtype=x_post.dtype)
            std = torch.tensor([0.229, 0.224, 0.225], device=x_post.device, dtype=x_post.dtype)
            x_post = (x_post - mean) / std
            convnext_feats = variables["convnext_model"].get_activations(x_post, **convnext_kwargs)
            usual_feats = {**usual_feats, **convnext_feats}
        return usual_feats

    return activation_fn, variables
