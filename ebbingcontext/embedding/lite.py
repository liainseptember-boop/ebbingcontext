"""Lightweight embedding provider — zero external dependencies.

Uses a hash-based bag-of-words approach (hashing trick) to produce
fixed-dimension vectors. Not accurate, but functional for trying the
system without installing sentence-transformers or calling external APIs.
"""

from __future__ import annotations

import hashlib
import math
import re

from ebbingcontext.embedding.base import EmbeddingProvider


class LiteEmbeddingProvider(EmbeddingProvider):
    """Zero-dependency embedding using hashing trick.

    Suitable for development, testing, and basic usage.
    For production quality, use BGE-M3 or OpenAI embeddings.
    """

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Produce a fixed-dimension vector via hashing trick + TF normalization."""
        tokens = _tokenize(text)
        vec = [0.0] * self._dimension

        # Hash each token into a bucket and accumulate
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            bucket = h % self._dimension
            # Use a second hash bit to determine sign (+1 or -1)
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[bucket] += sign

        # Also hash bigrams for basic phrase sensitivity
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i + 1]
            h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            bucket = h % self._dimension
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[bucket] += sign * 0.5

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff\u3040-\u30ff]+", text)
    return tokens
