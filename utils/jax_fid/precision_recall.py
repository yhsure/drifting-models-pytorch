from __future__ import annotations

from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    if num_workers <= 0:
        return []
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


class ManifoldEstimator:
    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        self.distance_block = DistanceBlock()
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float64)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float64)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]
            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]
                distance_batch[0 : end1 - begin1, begin2:end2] = self.distance_block.pairwise_distances(row_batch, col_batch)
            radii[begin1:end1, :] = np.concatenate(
                [x[:, self.nhood_sizes] for x in _numpy_partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ):
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = min(begin_1 + self.row_batch_size, len(features_1))
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = min(begin_2 + self.col_batch_size, len(features_2))
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1,
                    radii_1[begin_1:end_1],
                    batch_2,
                    radii_2[begin_2:end_2],
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


class DistanceBlock:
    def __init__(self):
        pass

    def pairwise_distances(self, U, V):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        return _batch_pairwise_distances(U, V)

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        distances = self.pairwise_distances(batch_1, batch_2)
        batch_1_in = np.any(distances[..., None] <= radii_2, axis=1)
        batch_2_in = np.any(distances[..., None] <= radii_1[:, None], axis=0)
        return batch_1_in, batch_2_in


def _batch_pairwise_distances(U, V):
    norm_u = np.sum(np.square(U), axis=1)
    norm_v = np.sum(np.square(V), axis=1)
    norm_u = np.reshape(norm_u, [-1, 1])
    norm_v = np.reshape(norm_v, [1, -1])
    return np.maximum(norm_u - 2 * np.matmul(U, V.T) + norm_v, 0.0)


def compute_precision_recall(features_real, features_fake, k=3):
    evaluator = ManifoldEstimator(nhood_sizes=(k,))
    radii_real = evaluator.manifold_radii(np.asarray(features_real, dtype=np.float64))
    radii_fake = evaluator.manifold_radii(np.asarray(features_fake, dtype=np.float64))
    precision, recall = evaluator.evaluate_pr(features_real, radii_real, features_fake, radii_fake)
    return float(precision[0]), float(recall[0])
