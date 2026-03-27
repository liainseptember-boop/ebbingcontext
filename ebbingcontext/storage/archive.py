"""Archive layer storage (audit trail).

Archive preserves:
    - Full decay paths
    - Reference chains
    - Context snapshots

Items here are kept for audit/compliance, not for active retrieval.
"""

from __future__ import annotations

from ebbingcontext.models import AuditRecord, MemoryItem, StorageLayer


class ArchiveStore:
    """Persistent store for archived memories and audit records."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._audit_log: list[AuditRecord] = []

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def audit_count(self) -> int:
        return len(self._audit_log)

    def add(self, item: MemoryItem) -> None:
        """Archive a memory item."""
        item.layer = StorageLayer.ARCHIVE
        self._items[item.id] = item

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def get_all(self, agent_id: str | None = None) -> list[MemoryItem]:
        items = list(self._items.values())
        if agent_id is not None:
            items = [i for i in items if i.agent_id == agent_id]
        return items

    def add_audit_record(self, record: AuditRecord) -> None:
        """Append an audit record."""
        self._audit_log.append(record)

    def get_audit_trail(self, memory_id: str) -> list[AuditRecord]:
        """Get the full audit trail for a specific memory."""
        return [r for r in self._audit_log if r.memory_id == memory_id]

    def get_recent_audits(self, limit: int = 50) -> list[AuditRecord]:
        """Get recent audit records."""
        return self._audit_log[-limit:]
