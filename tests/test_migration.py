"""Tests for the migration engine."""

from ebbingcontext.core.migration import MigrationDirection, MigrationEngine
from ebbingcontext.models import MemoryItem, StorageLayer


def make_item(strength: float, layer: StorageLayer) -> MemoryItem:
    item = MemoryItem(content="test", strength=strength)
    item.layer = layer
    return item


class TestMigrationEngine:
    def setup_method(self):
        self.engine = MigrationEngine(theta_active=0.6, theta_archive=0.15)

    def test_active_stays_when_strong(self):
        item = make_item(0.8, StorageLayer.ACTIVE)
        assert self.engine.evaluate(item) is None

    def test_active_demotes_to_warm(self):
        item = make_item(0.4, StorageLayer.ACTIVE)
        action = self.engine.evaluate(item)
        assert action is not None
        assert action.to_layer == StorageLayer.WARM
        assert action.direction == MigrationDirection.DEMOTE

    def test_active_demotes_to_archive_when_very_weak(self):
        item = make_item(0.1, StorageLayer.ACTIVE)
        action = self.engine.evaluate(item)
        assert action is not None
        assert action.to_layer == StorageLayer.ARCHIVE

    def test_warm_stays_in_range(self):
        item = make_item(0.3, StorageLayer.WARM)
        assert self.engine.evaluate(item) is None

    def test_warm_demotes_to_archive(self):
        item = make_item(0.1, StorageLayer.WARM)
        action = self.engine.evaluate(item)
        assert action is not None
        assert action.to_layer == StorageLayer.ARCHIVE

    def test_warm_promotes_to_active(self):
        item = make_item(0.7, StorageLayer.WARM)
        action = self.engine.evaluate(item)
        assert action is not None
        assert action.to_layer == StorageLayer.ACTIVE
        assert action.direction == MigrationDirection.PROMOTE

    def test_archive_no_migration(self):
        item = make_item(0.05, StorageLayer.ARCHIVE)
        assert self.engine.evaluate(item) is None

    def test_batch_evaluate(self):
        items = [
            make_item(0.8, StorageLayer.ACTIVE),  # stays
            make_item(0.4, StorageLayer.ACTIVE),  # demote to warm
            make_item(0.1, StorageLayer.WARM),     # demote to archive
        ]
        actions = self.engine.evaluate_batch(items)
        assert len(actions) == 2
