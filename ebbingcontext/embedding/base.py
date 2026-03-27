"""Abstract base for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Protocol for embedding providers.

    Implementations: LiteEmbeddingProvider (zero-dep), BGEEmbeddingProvider, OpenAIEmbeddingProvider.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Default: call embed() sequentially."""
        return [self.embed(t) for t in texts]
