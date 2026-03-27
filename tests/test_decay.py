"""Tests for the decay engine."""

import math

from ebbingcontext.core.decay import DecayEngine
from ebbingcontext.models import DecayStrategy, MemoryItem, StorageLayer


def make_item(**kwargs) -> MemoryItem:
    defaults = {
        "content": "test memory",
        "importance": 0.5,
        "access_count": 0,
        "last_accessed_at": 1000.0,
        "token_position": 0,
    }
    defaults.update(kwargs)
    return MemoryItem(**defaults)


class TestDecayEngine:
    def setup_method(self):
        self.engine = DecayEngine()

    def test_pin_always_returns_one(self):
        item = make_item(decay_strategy=DecayStrategy.PIN)
        strength = self.engine.compute_strength(item, current_time=99999.0)
        assert strength == 1.0

    def test_no_time_elapsed_returns_full_strength(self):
        item = make_item(last_accessed_at=1000.0)
        strength = self.engine.compute_strength(item, current_time=1000.0)
        assert strength == 1.0

    def test_strength_decays_over_time(self):
        item = make_item(last_accessed_at=1000.0)
        s1 = self.engine.compute_strength(item, current_time=1001.0)
        s2 = self.engine.compute_strength(item, current_time=1010.0)
        s3 = self.engine.compute_strength(item, current_time=1100.0)
        assert s1 > s2 > s3

    def test_floor_retention(self):
        """Strength never goes below ρ (0.1 by default)."""
        item = make_item(last_accessed_at=0.0, importance=0.1)
        strength = self.engine.compute_strength(item, current_time=1_000_000.0)
        assert strength >= self.engine.rho

    def test_higher_importance_decays_slower(self):
        # Use small delta_t so both items are above the floor retention
        item_low = make_item(importance=0.2, last_accessed_at=1000.0)
        item_high = make_item(importance=0.9, last_accessed_at=1000.0)
        t = 1000.5  # small delta
        s_low = self.engine.compute_strength(item_low, current_time=t)
        s_high = self.engine.compute_strength(item_high, current_time=t)
        assert s_high > s_low

    def test_more_access_decays_slower(self):
        item_fresh = make_item(access_count=0, last_accessed_at=1000.0)
        item_accessed = make_item(access_count=10, last_accessed_at=1000.0)
        t = 1000.5  # small delta
        s_fresh = self.engine.compute_strength(item_fresh, current_time=t)
        s_accessed = self.engine.compute_strength(item_accessed, current_time=t)
        assert s_accessed > s_fresh

    def test_token_distance_decay(self):
        # Use high importance for larger stability, so token deltas show effect
        item = make_item(token_position=0, importance=1.0)
        s1 = self.engine.compute_strength(item, current_time=0, current_token_pos=1)
        s2 = self.engine.compute_strength(item, current_time=0, current_token_pos=5)
        assert s1 > s2

    def test_active_beta_faster_than_warm(self):
        """Active layer (β=1.2) should decay faster than Warm (β=0.8)."""
        item_active = make_item(last_accessed_at=1000.0, layer=StorageLayer.ACTIVE)
        item_warm = make_item(last_accessed_at=1000.0, layer=StorageLayer.WARM)
        item_warm.layer = StorageLayer.WARM

        t = 1010.0
        s_active = self.engine.compute_strength(item_active, current_time=t)
        s_warm = self.engine.compute_strength(item_warm, current_time=t)
        # With same delta_t, Active should have lower strength (faster decay)
        assert s_warm > s_active

    def test_stability_formula(self):
        item = make_item(importance=0.5, access_count=5)
        s = self.engine.compute_stability(item)
        expected = 0.5 * 1.0 * (1.0 + 0.3 * math.log(1.0 + 5))
        assert abs(s - expected) < 1e-10

    def test_batch_update_sorts_by_strength(self):
        items = [
            make_item(importance=0.1, last_accessed_at=900.0),
            make_item(importance=0.9, last_accessed_at=999.0),
            make_item(importance=0.5, last_accessed_at=950.0),
        ]
        result = self.engine.batch_update(items, current_time=1000.0)
        strengths = [item.strength for item in result]
        assert strengths == sorted(strengths, reverse=True)

    def test_update_strength_modifies_item(self):
        item = make_item(last_accessed_at=1000.0)
        assert item.strength == 1.0
        self.engine.update_strength(item, current_time=1100.0)
        assert item.strength < 1.0
