"""Warm layer storage (vector database).

Warm layer holds memories with θ₂ < strength ≤ θ₁.
These are retrievable via semantic search but don't occupy the token budget.
"""

from __future__ import annotations

import numpy as np

from ebbingcontext.models import MemoryItem, StorageLayer


class WarmStore:
    """Vector-backed store for the Warm layer.

    V1: In-memory numpy implementation for zero external dependencies.
    Production: swap to ChromaDB / pgvector via the same interface.
    """

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    @property
    def count(self) -> int:
        return len(self._items)

    def add(self, item: MemoryItem) -> None:
        """Add a memory item to the Warm layer."""
        item.layer = StorageLayer.WARM
        self._items[item.id] = item
        if item.embedding is not None:
            self._embeddings[item.id] = np.array(item.embedding, dtype=np.float32)

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def remove(self, memory_id: str) -> MemoryItem | None:
        """Remove and return a memory item."""
        self._embeddings.pop(memory_id, None)
        return self._items.pop(memory_id, None)

    def get_all(self, agent_id: str | None = None) -> list[MemoryItem]:
        items = list(self._items.values())
        if agent_id is not None:
            items = [i for i in items if i.agent_id == agent_id]
        return items

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        agent_id: str | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Search for similar memories using cosine similarity.

        Returns list of (MemoryItem, similarity) tuples, sorted by similarity desc.
        """
        if not self._embeddings:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        results: list[tuple[MemoryItem, float]] = []

        for mem_id, emb in self._embeddings.items():
            item = self._items[mem_id]
            if agent_id is not None and item.agent_id != agent_id:
                continue

            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(query_vec, emb / emb_norm))
            results.append((item, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
