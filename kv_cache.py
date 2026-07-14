"""Didactical KV cache attention and cache compression utilities.

This module implements a small, dependency-free variant of multi-head attention
for autoregressive decoding. The layer stores keys and values internally and
reuses them across decoding steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

Vector = List[float]
HeadVectors = List[Vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(values: Sequence[float]) -> List[float]:
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    denom = sum(exps)
    return [v / denom for v in exps]


def _weighted_sum(weights: Sequence[float], vectors: Sequence[Sequence[float]]) -> Vector:
    out = [0.0 for _ in vectors[0]]
    for w, vec in zip(weights, vectors):
        for i, value in enumerate(vec):
            out[i] += w * value
    return out


@dataclass
class CompressionStats:
    kept_indices_by_head: Dict[int, List[int]]
    removed_indices_by_head: Dict[int, List[int]]


class KnormPress:
    """Simple KV cache compression inspired by KVPress `KnormPress`.

    It keeps at most `max_cache_tokens` tokens per head by selecting key vectors
    with the largest L2 norm and dropping the others.
    """

    def __init__(self, max_cache_tokens: int):
        if max_cache_tokens <= 0:
            raise ValueError("max_cache_tokens must be > 0")
        self.max_cache_tokens = max_cache_tokens

    def compress(self, cache_k: List[HeadVectors], cache_v: List[HeadVectors]) -> CompressionStats:
        kept: Dict[int, List[int]] = {}
        removed: Dict[int, List[int]] = {}

        for head_idx in range(len(cache_k)):
            keys = cache_k[head_idx]
            vals = cache_v[head_idx]
            if len(keys) <= self.max_cache_tokens:
                kept[head_idx] = list(range(len(keys)))
                removed[head_idx] = []
                continue

            norms = [math.sqrt(sum(x * x for x in vec)) for vec in keys]
            top_indices = sorted(
                sorted(range(len(keys)), key=lambda i: norms[i], reverse=True)[: self.max_cache_tokens]
            )
            removed_indices = [i for i in range(len(keys)) if i not in set(top_indices)]

            cache_k[head_idx] = [keys[i] for i in top_indices]
            cache_v[head_idx] = [vals[i] for i in top_indices]

            kept[head_idx] = top_indices
            removed[head_idx] = removed_indices

        return CompressionStats(kept_indices_by_head=kept, removed_indices_by_head=removed)


class MultiHeadAttentionWithKVCache:
    """A small educational MHA layer with an internal KV cache.

    Input format is one decoding step at a time:
      - `q`, `k`, and `v` are lists of shape [num_heads][head_dim].
    """

    def __init__(self, num_heads: int, head_dim: int):
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be > 0")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache_k: List[HeadVectors] = [[] for _ in range(num_heads)]
        self.cache_v: List[HeadVectors] = [[] for _ in range(num_heads)]

    @property
    def cache_size(self) -> int:
        # In this didactical setup we keep each head at the same cache length.
        return len(self.cache_k[0]) if self.cache_k else 0

    def reset_cache(self) -> None:
        self.cache_k = [[] for _ in range(self.num_heads)]
        self.cache_v = [[] for _ in range(self.num_heads)]

    def _validate_step_tensors(self, q: List[Vector], k: List[Vector], v: List[Vector]) -> None:
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if len(tensor) != self.num_heads:
                raise ValueError(f"{name} must contain {self.num_heads} heads")
            for vec in tensor:
                if len(vec) != self.head_dim:
                    raise ValueError(f"{name} vectors must have size {self.head_dim}")

    def forward(
        self,
        q: List[Vector],
        k: List[Vector],
        v: List[Vector],
        compressor: KnormPress | None = None,
    ) -> tuple[List[Vector], CompressionStats | None]:
        """Compute one decoding step and update cache.

        Returns:
          (per-head output vectors, compression stats if compression was applied)
        """
        self._validate_step_tensors(q, k, v)

        for head_idx in range(self.num_heads):
            self.cache_k[head_idx].append(list(k[head_idx]))
            self.cache_v[head_idx].append(list(v[head_idx]))

        stats = None
        if compressor is not None:
            stats = compressor.compress(self.cache_k, self.cache_v)

        scale = 1.0 / math.sqrt(self.head_dim)
        outputs: List[Vector] = []
        for head_idx in range(self.num_heads):
            keys = self.cache_k[head_idx]
            vals = self.cache_v[head_idx]
            query = q[head_idx]

            scores = [_dot(query, key) * scale for key in keys]
            weights = _softmax(scores)
            outputs.append(_weighted_sum(weights, vals))

        return outputs, stats
