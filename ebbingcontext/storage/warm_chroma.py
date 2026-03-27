"""ChromaDB-backed Warm layer storage.

Persistent vector search using ChromaDB.
Same interface as WarmStore for drop-in replacement.
"""

from __future__ import annotations

from ebbingcontext.models import MemoryItem, StorageLayer


class ChromaWarmStore:
    """Warm layer backed by ChromaDB for persistent vector search."""

    def __init__(
        self,
        persist_dir: str = ".ebbingcontext/chroma",
        collection_name: str = "warm",
    ) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._items: dict[str, MemoryItem] = {}

    @property
    def count(self) -> int:
        return len(self._items)

    def add(self, item: MemoryItem) -> None:
        item.layer = StorageLayer.WARM
        self._items[item.id] = item
        if item.embedding is not None:
            self._collection.upsert(
                ids=[item.id],
                embeddings=[item.embedding],
                metadatas=[{"agent_id": item.agent_id, "content": item.content[:512]}],
            )

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def remove(self, memory_id: str) -> MemoryItem | None:
        item = self._items.pop(memory_id, None)
        if item is not None:
            try:
                self._collection.delete(ids=[memory_id])
            except Exception:
                pass  # Item may not exist in ChromaDB
        return item

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
        """Search using ChromaDB's vector similarity."""
        if not self._items:
            return []

        where = {"agent_id": agent_id} if agent_id else None
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                where=where,
            )
        except Exception:
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        output: list[tuple[MemoryItem, float]] = []
        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for mem_id, distance in zip(ids, distances):
            item = self._items.get(mem_id)
            if item is not None:
                # ChromaDB cosine distance = 1 - similarity
                similarity = max(0.0, 1.0 - distance)
                output.append((item, similarity))

        output.sort(key=lambda x: x[1], reverse=True)
        return output[:top_k]
