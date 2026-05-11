from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ArrayMemoryBank:
    """Class-wise ring buffer for feature/image samples used by generator training."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype)

    def add(self, samples, labels) -> None:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def sample(self, labels, n_samples: int):
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
            else:
                sample_indices[i] = np.random.choice(valid, n_samples, replace=(valid < n_samples))

        out = self.bank[labels[:, None], sample_indices]
        return torch.from_numpy(out)


class ComponentArrayMemoryBank(ArrayMemoryBank):
    """Class-wise ring buffer with a primary likelihood component per sample."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        super().__init__(num_classes=num_classes, max_size=max_size, dtype=dtype)
        self.components: Optional[np.ndarray] = None

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        super()._init_bank(sample_shape)
        self.components = np.full((self.num_classes, self.max_size), -1, dtype=np.int64)

    def add(self, samples, labels, components=None) -> None:
        if components is None:
            components = np.full((len(labels),), -1, dtype=np.int64)
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(components, torch.Tensor):
            components = components.detach().cpu().numpy()
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        components = np.asarray(components)
        if self.bank is None:
            self._init_bank(samples.shape[1:])
        assert self.components is not None

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.components[lbl, idx] = int(components[i])
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def sample_likelihood(
        self,
        labels,
        active_components,
        active_weights,
        n_samples: int,
        uniform_fraction: float = 0.2,
    ):
        if self.bank is None or self.feature_shape is None or self.components is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(active_components, torch.Tensor):
            active_components = active_components.detach().cpu().numpy()
        if isinstance(active_weights, torch.Tensor):
            active_weights = active_weights.detach().cpu().numpy()
        labels = np.asarray(labels)
        active_components = np.asarray(active_components)
        active_weights = np.asarray(active_weights, dtype=np.float64)
        bsz = labels.shape[0]
        out_indices = np.empty((bsz, n_samples), dtype=np.int32)
        uniform_fraction = float(np.clip(uniform_fraction, 0.0, 1.0))

        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                out_indices[i] = np.zeros((n_samples,), dtype=np.int32)
                continue
            valid_components = self.components[lbl, :valid]
            cls_indices = np.arange(valid, dtype=np.int32)
            weights = active_weights[i]
            if weights.sum() <= 0:
                weights = np.full_like(weights, 1.0 / max(1, weights.shape[0]), dtype=np.float64)
            else:
                weights = weights / weights.sum()

            for j in range(n_samples):
                use_uniform = np.random.random() < uniform_fraction
                if use_uniform:
                    out_indices[i, j] = int(np.random.choice(cls_indices))
                    continue
                component = int(np.random.choice(active_components[i], p=weights))
                candidates = cls_indices[valid_components == component]
                if candidates.shape[0] == 0:
                    candidates = cls_indices
                out_indices[i, j] = int(np.random.choice(candidates))

        out = self.bank[labels[:, None], out_indices]
        return torch.from_numpy(out)
