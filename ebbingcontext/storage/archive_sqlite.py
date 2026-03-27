"""SQLite-backed Archive layer storage.

Persistent audit trail and archived memories.
Same interface as ArchiveStore for drop-in replacement.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ebbingcontext.models import (
    AuditRecord,
    DecayStrategy,
    MemoryItem,
    SensitivityLevel,
    StorageLayer,
)


class SQLiteArchiveStore:
    """Archive layer backed by SQLite for persistent audit trail."""

    def __init__(self, db_path: str = ".ebbingcontext/archive.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                decay_strategy TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                importance REAL NOT NULL,
                strength REAL NOT NULL,
                access_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                source_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event TEXT NOT NULL,
                from_layer TEXT,
                to_layer TEXT,
                strength_at_event REAL NOT NULL,
                timestamp REAL NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_audit_memory_id ON audit_log(memory_id);
        """)
        self._conn.commit()

    @property
    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]

    @property
    def audit_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
        return row[0]

    def add(self, item: MemoryItem) -> None:
        item.layer = StorageLayer.ARCHIVE
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
            (id, content, agent_id, decay_strategy, sensitivity, importance,
             strength, access_count, created_at, last_accessed_at, source_type, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id, item.content, item.agent_id,
                item.decay_strategy.value, item.sensitivity.value,
                item.importance, item.strength, item.access_count,
                item.created_at, item.last_accessed_at, item.source_type,
                json.dumps(item.metadata, default=str),
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryItem | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def get_all(self, agent_id: str | None = None) -> list[MemoryItem]:
        if agent_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE agent_id = ?", (agent_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM memories").fetchall()
        return [self._row_to_item(r) for r in rows]

    def add_audit_record(self, record: AuditRecord) -> None:
        self._conn.execute(
            """INSERT INTO audit_log
            (id, memory_id, event, from_layer, to_layer, strength_at_event, timestamp, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.memory_id, record.event,
                record.from_layer.value if record.from_layer else None,
                record.to_layer.value if record.to_layer else None,
                record.strength_at_event, record.timestamp,
                json.dumps(record.details, default=str),
            ),
        )
        self._conn.commit()

    def get_audit_trail(self, memory_id: str) -> list[AuditRecord]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE memory_id = ? ORDER BY timestamp",
            (memory_id,),
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    def get_recent_audits(self, limit: int = 50) -> list[AuditRecord]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            agent_id=row["agent_id"],
            decay_strategy=DecayStrategy(row["decay_strategy"]),
            sensitivity=SensitivityLevel(row["sensitivity"]),
            importance=row["importance"],
            strength=row["strength"],
            access_count=row["access_count"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            source_type=row["source_type"],
            layer=StorageLayer.ARCHIVE,
            metadata=json.loads(row["metadata_json"]),
        )

    def _row_to_audit(self, row: sqlite3.Row) -> AuditRecord:
        return AuditRecord(
            id=row["id"],
            memory_id=row["memory_id"],
            event=row["event"],
            from_layer=StorageLayer(row["from_layer"]) if row["from_layer"] else None,
            to_layer=StorageLayer(row["to_layer"]) if row["to_layer"] else None,
            strength_at_event=row["strength_at_event"],
            timestamp=row["timestamp"],
            details=json.loads(row["details_json"]),
        )
