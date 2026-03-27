"""BGE-M3 embedding provider using sentence-transformers.

Requires: pip install ebbingcontext[bge]
Model: BAAI/bge-m3 (1024 dimensions)
Lazy-loads the model on first call.
"""

from __future__ import annotations

import threading

from ebbingcontext.embedding.base import EmbeddingProvider


class BGEEmbeddingProvider(EmbeddingProvider):
    """Local BGE-M3 embedding via sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", dimension: int = 1024) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model = None
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError:
                        raise ImportError(
                            "sentence-transformers is required for BGE-M3 embeddings. "
                            "Install with: pip install ebbingcontext[bge]"
                        )
                    self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
