"""Compatibility shim for historical JAX Inception definitions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3 as _FIDInceptionV3


class Dense(nn.Module):
    def __init__(self, in_features: int = 2048, out_features: int = 1000):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 32, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InceptionB(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InceptionC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InceptionD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InceptionE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InceptionAux(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BatchNorm(nn.BatchNorm2d):
    pass


def _absolute_dims(rank, dims):
    del rank
    return tuple(abs(int(v)) for v in dims)


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
    del init, reduce_fn, padding
    if isinstance(window_shape, int):
        kernel = (window_shape, window_shape)
    else:
        kernel = tuple(window_shape)
    if strides is None:
        strides = kernel
    if isinstance(strides, int):
        strides = (strides, strides)
    return F.avg_pool2d(inputs, kernel_size=kernel, stride=strides)


def avg_pool(inputs, window_shape, strides=None, padding="VALID"):
    return pool(inputs, None, None, window_shape, strides, padding)


class InceptionV3(nn.Module):
    def __init__(self, pretrained=True, include_head=True, transform_input=False):
        super().__init__()
        del pretrained, include_head, transform_input
        block_idx = _FIDInceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = _FIDInceptionV3([block_idx])

    def forward(self, x, train=False):
        del train
        pooled = self.model(x)[0]
        pooled = pooled.squeeze(-1).squeeze(-1)
        logits = torch.zeros((pooled.shape[0], 1000), device=pooled.device, dtype=pooled.dtype)
        return pooled, None, logits

    def apply(self, params, x, train=False):
        del params
        return self.forward(x, train=train)
