"""Embedding provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebbingcontext.config import EmbeddingConfig

from ebbingcontext.embedding.base import EmbeddingProvider


def create_embedding_provider(config: "EmbeddingConfig") -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Fallback chain: requested model -> lite (zero-dependency).
    """
    if config.model == "bge-m3":
        try:
            import sentence_transformers  # noqa: F401

            from ebbingcontext.embedding.bge import BGEEmbeddingProvider

            return BGEEmbeddingProvider(dimension=config.dimension)
        except ImportError:
            # sentence-transformers not installed, fall back to lite
            from ebbingcontext.embedding.lite import LiteEmbeddingProvider

            return LiteEmbeddingProvider()

    elif config.model == "openai":
        from ebbingcontext.embedding.openai_embed import OpenAIEmbeddingProvider

        import os

        api_key = os.environ.get("EBBINGCONTEXT_ADAPTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI embedding requires EBBINGCONTEXT_ADAPTER_API_KEY environment variable."
            )
        return OpenAIEmbeddingProvider(api_key=api_key, dimension=config.dimension)

    elif config.model == "lite":
        from ebbingcontext.embedding.lite import LiteEmbeddingProvider

        return LiteEmbeddingProvider(dimension=config.dimension)

    else:
        # Unknown model, fall back to lite
        from ebbingcontext.embedding.lite import LiteEmbeddingProvider

        return LiteEmbeddingProvider()
