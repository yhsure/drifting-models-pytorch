"""Compatibility shim for historical JAX FID conversion helpers."""

from __future__ import annotations


class ddd(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = ddd()
        return super().__getitem__(key)


def load_bbasiconv2d(op, ob, p):
    del op, ob, p
    raise NotImplementedError(
        "load_bbasiconv2d is a legacy JAX conversion helper and is not used in the PyTorch path."
    )


def load_inceptionA(op, ob, p):
    del op, ob, p
    raise NotImplementedError(
        "load_inceptionA is a legacy JAX conversion helper and is not used in the PyTorch path."
    )


def load_inceptionB(op, ob, p):
    del op, ob, p
    raise NotImplementedError(
        "load_inceptionB is a legacy JAX conversion helper and is not used in the PyTorch path."
    )


def load_inceptionC(op, ob, p):
    del op, ob, p
    raise NotImplementedError(
        "load_inceptionC is a legacy JAX conversion helper and is not used in the PyTorch path."
    )


def load_inceptionD(op, ob, p):
    del op, ob, p
    raise NotImplementedError(
        "load_inceptionD is a legacy JAX conversion helper and is not used in the PyTorch path."
    )


def load_inceptionE(op, ob, p):
    del op, ob, p
    raise NotImplementedError(
        "load_inceptionE is a legacy JAX conversion helper and is not used in the PyTorch path."
    )


def load_all():
    raise NotImplementedError(
        "load_all is a legacy JAX conversion helper and is not used in the PyTorch path."
    )
