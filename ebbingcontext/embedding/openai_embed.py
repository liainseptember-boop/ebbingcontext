"""OpenAI-compatible embedding provider.

Requires: pip install "ebbingcontext[openai]"
Supports any OpenAI-compatible API endpoint.
"""

from __future__ import annotations

from ebbingcontext.embedding.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dimension: int = 1536,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install with: pip install "ebbingcontext[openai]""
            )

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]
