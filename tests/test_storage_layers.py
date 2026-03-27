"""Unit tests for WarmStore and ArchiveStore."""

from ebbingcontext.models import AuditRecord, MemoryItem, StorageLayer
from ebbingcontext.storage.archive import ArchiveStore
from ebbingcontext.storage.warm import WarmStore


class TestWarmStore:
    def setup_method(self):
        self.store = WarmStore()

    def test_add_and_get(self):
        item = MemoryItem(content="test item")
        self.store.add(item)
        assert self.store.get(item.id) is item

    def test_add_sets_layer_warm(self):
        item = MemoryItem(content="test")
        self.store.add(item)
        assert item.layer == StorageLayer.WARM

    def test_remove_returns_item(self):
        item = MemoryItem(content="test")
        self.store.add(item)
        removed = self.store.remove(item.id)
        assert removed is item
        assert self.store.get(item.id) is None

    def test_remove_nonexistent_returns_none(self):
        assert self.store.remove("nonexistent") is None

    def test_get_all(self):
        items = [MemoryItem(content=f"item {i}") for i in range(3)]
        for item in items:
            self.store.add(item)
        assert len(self.store.get_all()) == 3

    def test_get_all_filters_by_agent_id(self):
        item_a = MemoryItem(content="agent a", agent_id="a")
        item_b = MemoryItem(content="agent b", agent_id="b")
        self.store.add(item_a)
        self.store.add(item_b)
        assert len(self.store.get_all(agent_id="a")) == 1
        assert self.store.get_all(agent_id="a")[0].agent_id == "a"

    def test_count_property(self):
        assert self.store.count == 0
        self.store.add(MemoryItem(content="one"))
        assert self.store.count == 1
        self.store.add(MemoryItem(content="two"))
        assert self.store.count == 2

    def test_search_cosine_similarity_ordering(self):
        """Search returns results sorted by cosine similarity descending."""
        item_close = MemoryItem(content="close", embedding=[1.0, 0.1, 0.0])
        item_far = MemoryItem(content="far", embedding=[0.0, 1.0, 0.0])
        self.store.add(item_close)
        self.store.add(item_far)

        results = self.store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0].id == item_close.id
        assert results[0][1] > results[1][1]

    def test_search_empty_store(self):
        results = self.store.search([1.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_search_zero_norm_query(self):
        item = MemoryItem(content="test", embedding=[1.0, 0.0, 0.0])
        self.store.add(item)
        results = self.store.search([0.0, 0.0, 0.0], top_k=5)
        assert results == []


class TestArchiveStore:
    def setup_method(self):
        self.store = ArchiveStore()

    def test_add_and_get(self):
        item = MemoryItem(content="archived item")
        self.store.add(item)
        assert self.store.get(item.id) is item

    def test_add_sets_layer_archive(self):
        item = MemoryItem(content="test")
        self.store.add(item)
        assert item.layer == StorageLayer.ARCHIVE

    def test_get_all_filters_by_agent_id(self):
        item_a = MemoryItem(content="agent a", agent_id="a")
        item_b = MemoryItem(content="agent b", agent_id="b")
        self.store.add(item_a)
        self.store.add(item_b)
        result = self.store.get_all(agent_id="a")
        assert len(result) == 1
        assert result[0].agent_id == "a"

    def test_add_audit_record_and_get_trail(self):
        record = AuditRecord(memory_id="mem1", event="created", to_layer=StorageLayer.ACTIVE)
        self.store.add_audit_record(record)
        trail = self.store.get_audit_trail("mem1")
        assert len(trail) == 1
        assert trail[0].event == "created"

    def test_get_recent_audits_with_limit(self):
        for i in range(10):
            self.store.add_audit_record(
                AuditRecord(memory_id=f"mem{i}", event="created", to_layer=StorageLayer.ACTIVE)
            )
        recent = self.store.get_recent_audits(limit=3)
        assert len(recent) == 3

    def test_audit_count_property(self):
        assert self.store.audit_count == 0
        self.store.add_audit_record(
            AuditRecord(memory_id="mem1", event="created", to_layer=StorageLayer.ACTIVE)
        )
        assert self.store.audit_count == 1

    def test_count_property(self):
        assert self.store.count == 0
        self.store.add(MemoryItem(content="one"))
        assert self.store.count == 1
