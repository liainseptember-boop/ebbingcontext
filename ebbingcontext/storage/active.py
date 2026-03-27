"""Active layer storage.

Active layer holds memories with strength > θ₁ (0.6).
These memories participate in prompt assembly and occupy the token budget.
"""

from __future__ import annotations

import json
from pathlib import Path

from ebbingcontext.models import MemoryItem, StorageLayer


class ActiveStore:
    """In-memory store for the Active layer with optional JSON persistence."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._persist_path = persist_path
        if persist_path is not None:
            self._load()

    @property
    def count(self) -> int:
        return len(self._items)

    def add(self, item: MemoryItem) -> None:
        """Add a memory item to the Active layer."""
        item.layer = StorageLayer.ACTIVE
        self._items[item.id] = item
        self._save()

    def get(self, memory_id: str) -> MemoryItem | None:
        """Get a memory item by ID."""
        return self._items.get(memory_id)

    def remove(self, memory_id: str) -> MemoryItem | None:
        """Remove and return a memory item."""
        item = self._items.pop(memory_id, None)
        if item is not None:
            self._save()
        return item

    def get_all(self, agent_id: str | None = None) -> list[MemoryItem]:
        """Get all items, optionally filtered by agent_id."""
        items = list(self._items.values())
        if agent_id is not None:
            items = [i for i in items if i.agent_id == agent_id]
        return items

    def get_pinned(self, agent_id: str | None = None) -> list[MemoryItem]:
        """Get all pinned items."""
        from ebbingcontext.models import DecayStrategy

        items = self.get_all(agent_id)
        return [i for i in items if i.decay_strategy == DecayStrategy.PIN]

    def get_pin_ratio(self, agent_id: str | None = None) -> float:
        """Get the ratio of pinned items to total items."""
        items = self.get_all(agent_id)
        if not items:
            return 0.0
        pinned = sum(1 for i in items if i.decay_strategy.value == "pin")
        return pinned / len(items)

    def _save(self) -> None:
        """Persist current state to JSON file."""
        if self._persist_path is None:
            return
        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [item.model_dump(exclude={"embedding"}) for item in self._items.values()]
        path.write_text(json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        """Load state from JSON file if it exists."""
        if self._persist_path is None:
            return
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for entry in raw:
                item = MemoryItem.model_validate(entry)
                self._items[item.id] = item
        except (json.JSONDecodeError, ValueError):
            pass  # Corrupted file — start fresh
