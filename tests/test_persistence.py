"""Tests for persistence layer (Active JSON, SQLite Archive, ChromaDB Warm)."""

import os

from ebbingcontext.models import (
    AuditRecord,
    DecayStrategy,
    MemoryItem,
    SensitivityLevel,
    StorageLayer,
)
from ebbingcontext.storage.active import ActiveStore
from ebbingcontext.storage.archive_sqlite import SQLiteArchiveStore


class TestActiveJSONPersistence:
    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "active.json")
        store = ActiveStore(persist_path=path)
        item = MemoryItem(content="Persistent fact", importance=0.7)
        store.add(item)
        assert store.count == 1

        # Reload from disk
        store2 = ActiveStore(persist_path=path)
        assert store2.count == 1
        loaded = store2.get(item.id)
        assert loaded is not None
        assert loaded.content == "Persistent fact"
        assert loaded.importance == 0.7

    def test_remove_persists(self, tmp_path):
        path = str(tmp_path / "active.json")
        store = ActiveStore(persist_path=path)
        item = MemoryItem(content="To be removed")
        store.add(item)
        store.remove(item.id)

        store2 = ActiveStore(persist_path=path)
        assert store2.count == 0

    def test_multiple_items_roundtrip(self, tmp_path):
        path = str(tmp_path / "active.json")
        store = ActiveStore(persist_path=path)
        items = [MemoryItem(content=f"Item {i}", agent_id="agent_a") for i in range(5)]
        for item in items:
            store.add(item)

        store2 = ActiveStore(persist_path=path)
        assert store2.count == 5
        all_items = store2.get_all(agent_id="agent_a")
        assert len(all_items) == 5

    def test_preserves_fields(self, tmp_path):
        path = str(tmp_path / "active.json")
        store = ActiveStore(persist_path=path)
        item = MemoryItem(
            content="Full field item",
            agent_id="agent_x",
            decay_strategy=DecayStrategy.PIN,
            sensitivity=SensitivityLevel.SENSITIVE,
            importance=0.9,
            strength=0.75,
            access_count=3,
            source_type="tool",
            metadata={"key": "value"},
        )
        store.add(item)

        store2 = ActiveStore(persist_path=path)
        loaded = store2.get(item.id)
        assert loaded.agent_id == "agent_x"
        assert loaded.decay_strategy == DecayStrategy.PIN
        assert loaded.sensitivity == SensitivityLevel.SENSITIVE
        assert loaded.importance == 0.9
        assert loaded.strength == 0.75
        assert loaded.access_count == 3
        assert loaded.source_type == "tool"
        assert loaded.metadata["key"] == "value"

    def test_no_persist_path_works(self):
        """Without persist_path, behaves as pure in-memory store."""
        store = ActiveStore()
        store.add(MemoryItem(content="Ephemeral"))
        assert store.count == 1

    def test_corrupted_file_starts_fresh(self, tmp_path):
        path = str(tmp_path / "active.json")
        # Write garbage
        with open(path, "w") as f:
            f.write("NOT VALID JSON {{{")

        store = ActiveStore(persist_path=path)
        assert store.count == 0

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "active.json")
        store = ActiveStore(persist_path=path)
        store.add(MemoryItem(content="Deep item"))
        assert os.path.exists(path)


class TestSQLiteArchiveStore:
    def test_add_and_get(self, tmp_path):
        db = str(tmp_path / "archive.db")
        store = SQLiteArchiveStore(db_path=db)
        item = MemoryItem(content="Archived memory", importance=0.3, strength=0.1)
        store.add(item)

        loaded = store.get(item.id)
        assert loaded is not None
        assert loaded.content == "Archived memory"
        assert loaded.layer == StorageLayer.ARCHIVE

    def test_persistence_across_instances(self, tmp_path):
        db = str(tmp_path / "archive.db")
        store1 = SQLiteArchiveStore(db_path=db)
        item = MemoryItem(content="Survives restart")
        store1.add(item)

        store2 = SQLiteArchiveStore(db_path=db)
        assert store2.count == 1
        loaded = store2.get(item.id)
        assert loaded.content == "Survives restart"

    def test_get_all_with_agent_filter(self, tmp_path):
        db = str(tmp_path / "archive.db")
        store = SQLiteArchiveStore(db_path=db)
        store.add(MemoryItem(content="A1", agent_id="alice"))
        store.add(MemoryItem(content="A2", agent_id="alice"))
        store.add(MemoryItem(content="B1", agent_id="bob"))

        alice_items = store.get_all(agent_id="alice")
        assert len(alice_items) == 2
        all_items = store.get_all()
        assert len(all_items) == 3

    def test_audit_record_roundtrip(self, tmp_path):
        db = str(tmp_path / "archive.db")
        store = SQLiteArchiveStore(db_path=db)

        record = AuditRecord(
            memory_id="mem_123",
            event="created",
            from_layer=StorageLayer.ACTIVE,
            to_layer=StorageLayer.ARCHIVE,
            strength_at_event=0.42,
            details={"reason": "decayed"},
        )
        store.add_audit_record(record)

        trail = store.get_audit_trail("mem_123")
        assert len(trail) == 1
        assert trail[0].event == "created"
        assert trail[0].strength_at_event == 0.42
        assert trail[0].details["reason"] == "decayed"
        assert trail[0].from_layer == StorageLayer.ACTIVE
        assert trail[0].to_layer == StorageLayer.ARCHIVE

    def test_get_recent_audits(self, tmp_path):
        db = str(tmp_path / "archive.db")
        store = SQLiteArchiveStore(db_path=db)

        for i in range(10):
            store.add_audit_record(AuditRecord(
                memory_id=f"mem_{i}",
                event="migrated",
                strength_at_event=0.1 * i,
            ))

        recent = store.get_recent_audits(limit=5)
        assert len(recent) == 5
        assert store.audit_count == 10

    def test_preserves_all_memory_fields(self, tmp_path):
        db = str(tmp_path / "archive.db")
        store = SQLiteArchiveStore(db_path=db)
        item = MemoryItem(
            content="Full fields",
            agent_id="agent_z",
            decay_strategy=DecayStrategy.DECAY_IRREVERSIBLE,
            sensitivity=SensitivityLevel.PUBLIC,
            importance=0.8,
            strength=0.2,
            access_count=5,
            source_type="agent",
            metadata={"transferred_from": "agent_a"},
        )
        store.add(item)

        loaded = store.get(item.id)
        assert loaded.decay_strategy == DecayStrategy.DECAY_IRREVERSIBLE
        assert loaded.sensitivity == SensitivityLevel.PUBLIC
        assert loaded.importance == 0.8
        assert loaded.access_count == 5
        assert loaded.metadata["transferred_from"] == "agent_a"

    def test_upsert_overwrites(self, tmp_path):
        db = str(tmp_path / "archive.db")
        store = SQLiteArchiveStore(db_path=db)
        item = MemoryItem(content="Original")
        store.add(item)

        item.content = "Updated"
        store.add(item)  # INSERT OR REPLACE

        assert store.count == 1
        loaded = store.get(item.id)
        assert loaded.content == "Updated"


class TestStorageFactory:
    def test_in_memory_by_default(self):
        from ebbingcontext.config import EbbingConfig
        from ebbingcontext.storage import create_stores

        config = EbbingConfig()
        active, warm, archive = create_stores(config)

        from ebbingcontext.storage.active import ActiveStore
        from ebbingcontext.storage.archive import ArchiveStore
        from ebbingcontext.storage.warm import WarmStore

        assert isinstance(active, ActiveStore)
        assert isinstance(warm, WarmStore)
        assert isinstance(archive, ArchiveStore)

    def test_persistent_stores(self, tmp_path):
        from ebbingcontext.config import EbbingConfig, StorageConfig, VectorStoreConfig
        from ebbingcontext.storage import create_stores

        config = EbbingConfig(
            storage=StorageConfig(
                persist=True,
                active_persist_path=str(tmp_path / "active.json"),
                archive_db_path=str(tmp_path / "archive.db"),
            ),
            vector_store=VectorStoreConfig(
                persist_dir=str(tmp_path / "chroma"),
            ),
        )

        try:
            active, warm, archive = create_stores(config)
            from ebbingcontext.storage.active import ActiveStore
            from ebbingcontext.storage.archive_sqlite import SQLiteArchiveStore
            from ebbingcontext.storage.warm_chroma import ChromaWarmStore

            assert isinstance(active, ActiveStore)
            assert isinstance(warm, ChromaWarmStore)
            assert isinstance(archive, SQLiteArchiveStore)
        except ImportError:
            # ChromaDB not installed — skip
            import pytest
            pytest.skip("chromadb not installed")


class TestEndToEndPersistence:
    def test_active_store_survives_restart(self, tmp_path):
        """Simulate: store → kill → restart → recall."""
        path = str(tmp_path / "active.json")

        # Session 1: store items
        store1 = ActiveStore(persist_path=path)
        items = []
        for content in ["Python is great", "Data science rocks", "Machine learning"]:
            item = MemoryItem(content=content, agent_id="default")
            store1.add(item)
            items.append(item)

        # Session 2: reload and verify
        store2 = ActiveStore(persist_path=path)
        assert store2.count == 3
        for original in items:
            loaded = store2.get(original.id)
            assert loaded is not None
            assert loaded.content == original.content

    def test_sqlite_archive_survives_restart(self, tmp_path):
        """Simulate: archive → kill → restart → query audit."""
        db = str(tmp_path / "archive.db")

        # Session 1
        store1 = SQLiteArchiveStore(db_path=db)
        item = MemoryItem(content="Important memory", strength=0.1)
        store1.add(item)
        store1.add_audit_record(AuditRecord(
            memory_id=item.id,
            event="migrated",
            from_layer=StorageLayer.WARM,
            to_layer=StorageLayer.ARCHIVE,
            strength_at_event=0.1,
        ))

        # Session 2
        store2 = SQLiteArchiveStore(db_path=db)
        loaded = store2.get(item.id)
        assert loaded is not None
        assert loaded.content == "Important memory"

        trail = store2.get_audit_trail(item.id)
        assert len(trail) == 1
        assert trail[0].event == "migrated"
