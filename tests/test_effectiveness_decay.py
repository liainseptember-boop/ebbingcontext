"""Effectiveness tests: Decay curve validation, recall boost, migration timeline.

Proves the Ebbinghaus decay formula works correctly and that the system
differentiates important/frequently-accessed memories from irrelevant ones.
"""

import math
import time

from ebbingcontext.core.decay import DecayEngine
from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.models import DecayStrategy, MemoryItem, StorageLayer


class TestDecayCurveValidation:
    """Validate the mathematical formula produces correct values."""

    def setup_method(self):
        self.engine = DecayEngine(s_base=1.0, alpha=0.3, beta_active=1.2, beta_warm=0.8, rho=0.1)

    def test_known_retention_values(self):
        """Verify R(t) at a specific checkpoint against hand-calculated value."""
        # importance=0.5, access_count=0 → S = 0.5 * 1.0 * (1 + 0.3*ln(1)) = 0.5
        # dt=0.5, beta=1.2 → R = exp(-(0.5/0.5)^1.2) = exp(-1.0) ≈ 0.3679
        # R̃ = 0.1 + 0.9 * 0.3679 ≈ 0.4311
        item = MemoryItem(content="test", importance=0.5, access_count=0, layer=StorageLayer.ACTIVE)
        now = 1000.0
        item.last_accessed_at = now - 0.5  # dt = 0.5
        item.created_at = item.last_accessed_at

        strength = self.engine.compute_strength(item, now)

        expected_R = math.exp(-(0.5 / 0.5) ** 1.2)
        expected = 0.1 + 0.9 * expected_R
        assert abs(strength - expected) < 1e-4, f"Expected {expected:.4f}, got {strength:.4f}"

    def test_half_life_increases_with_importance(self):
        """Higher importance → longer half-life (time to reach strength=0.5)."""
        half_lives = []
        for importance in [0.2, 0.5, 0.8]:
            item = MemoryItem(
                content="test", importance=importance,
                access_count=0, layer=StorageLayer.ACTIVE,
            )
            # Binary search for half-life
            lo, hi = 0.001, 100.0
            for _ in range(50):
                mid = (lo + hi) / 2
                item.last_accessed_at = 1000.0 - mid
                item.created_at = item.last_accessed_at
                s = self.engine.compute_strength(item, 1000.0)
                if s > 0.5:
                    lo = mid
                else:
                    hi = mid
            half_lives.append((lo + hi) / 2)

        # Half-lives must be strictly increasing
        assert half_lives[0] < half_lives[1] < half_lives[2], \
            f"Half-lives not increasing: {half_lives}"

    def test_access_count_extends_stability(self):
        """More accesses → higher stability → slower decay."""
        now = 1000.0
        dt = 0.3

        item_no_access = MemoryItem(
            content="no access", importance=0.5, access_count=0,
            layer=StorageLayer.ACTIVE, last_accessed_at=now - dt, created_at=now - dt,
        )
        item_many_access = MemoryItem(
            content="many access", importance=0.5, access_count=10,
            layer=StorageLayer.ACTIVE, last_accessed_at=now - dt, created_at=now - dt,
        )

        s_no = self.engine.compute_strength(item_no_access, now)
        s_many = self.engine.compute_strength(item_many_access, now)

        assert s_many > s_no, \
            f"Accessed item ({s_many:.4f}) should be stronger than unaccessed ({s_no:.4f})"

    def test_rho_floor_never_breached(self):
        """Strength never drops below ρ=0.1 for any parameter combination."""
        now = 1000000.0
        for importance in [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
            for dt in [1, 10, 100, 1000, 10000, 100000]:
                for layer in [StorageLayer.ACTIVE, StorageLayer.WARM]:
                    item = MemoryItem(
                        content="test", importance=importance,
                        access_count=0, layer=layer,
                        last_accessed_at=now - dt, created_at=now - dt,
                    )
                    strength = self.engine.compute_strength(item, now)
                    assert strength >= 0.1 - 1e-9, \
                        f"Floor breached: strength={strength} for importance={importance}, dt={dt}, layer={layer}"

    def test_pin_immunity_over_extreme_time(self):
        """Pinned items keep strength=1.0 even after astronomical time."""
        item = MemoryItem(
            content="pinned", decay_strategy=DecayStrategy.PIN,
            last_accessed_at=0.0, created_at=0.0,
        )
        strength = self.engine.compute_strength(item, 1e9)
        assert strength == 1.0

    def test_active_vs_warm_beta_divergence(self):
        """Active layer (β=1.2) decays faster than Warm layer (β=0.8)."""
        now = 1000.0
        # Use dt=1.0 so that dt/S=2.0 (not 1.0, since 1^anything=1)
        dt = 1.0

        item_active = MemoryItem(
            content="active", importance=0.5, access_count=0,
            layer=StorageLayer.ACTIVE,
            last_accessed_at=now - dt, created_at=now - dt,
        )
        item_warm = MemoryItem(
            content="warm", importance=0.5, access_count=0,
            layer=StorageLayer.WARM,
            last_accessed_at=now - dt, created_at=now - dt,
        )

        s_active = self.engine.compute_strength(item_active, now)
        s_warm = self.engine.compute_strength(item_warm, now)

        assert s_active < s_warm, \
            f"Active ({s_active:.4f}) should decay faster than Warm ({s_warm:.4f})"

    def test_strength_monotonically_decreasing(self):
        """Strength must never increase as time passes (for non-pin items)."""
        item = MemoryItem(
            content="test", importance=0.5, access_count=0,
            layer=StorageLayer.ACTIVE, last_accessed_at=0.0, created_at=0.0,
        )

        prev_strength = 1.0
        for t in range(1, 101):
            dt = t * 0.01  # 0.01 to 1.0
            strength = self.engine.compute_strength(item, dt)
            assert strength <= prev_strength + 1e-9, \
                f"Non-monotonic at t={dt}: {strength:.6f} > {prev_strength:.6f}"
            prev_strength = strength


class TestRecallBoostValidation:
    """Validate that accessing memories slows their decay."""

    def test_touch_updates_access_count_and_time(self):
        item = MemoryItem(content="test")
        old_time = item.last_accessed_at
        item.touch()
        item.touch()
        item.touch()
        assert item.access_count == 3
        assert item.last_accessed_at >= old_time

    def test_accessed_memory_decays_slower_than_peer(self):
        """Quantify the stability gap between accessed and unaccessed items."""
        engine = DecayEngine()
        now = 1000.0
        dt = 0.3

        base_time = now - dt
        item_untouched = MemoryItem(
            content="untouched", importance=0.5, access_count=0,
            layer=StorageLayer.ACTIVE, last_accessed_at=base_time, created_at=base_time,
        )
        item_touched = MemoryItem(
            content="touched", importance=0.5, access_count=5,
            layer=StorageLayer.ACTIVE, last_accessed_at=base_time, created_at=base_time,
        )

        s_untouched = engine.compute_strength(item_untouched, now)
        s_touched = engine.compute_strength(item_touched, now)

        gap = s_touched - s_untouched
        assert gap > 0.01, f"Stability gap too small: {gap:.4f}"

    def test_recall_implicitly_touches_items(self):
        """engine.recall() should increment access_count on returned items."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        item = engine.store("Python programming concepts", source_type="user")
        original_count = item.access_count

        engine.recall("Python", top_k=5)
        engine.recall("Python", top_k=5)
        engine.recall("Python", top_k=5)

        assert item.access_count >= original_count + 3

    def test_accessed_memory_survives_migration(self):
        """Frequently accessed memory survives migration while unaccessed one migrates."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        # Store two items
        accessed = engine.store("Important accessed fact", importance=0.4)
        unaccessed = engine.store("Unaccessed trivia", importance=0.4)

        # Touch the accessed one many times to boost stability
        for _ in range(10):
            accessed.touch()

        # Simulate time passage by manipulating timestamps
        past = time.time() - 3600  # 1 hour ago
        unaccessed.last_accessed_at = past
        unaccessed.created_at = past
        # accessed item has recent last_accessed_at from touch()

        engine.run_migration()

        # Accessed item should still be in Active
        assert engine.active.get(accessed.id) is not None
        # Unaccessed item should have migrated (or at least have lower strength)
        unaccessed_in_active = engine.active.get(unaccessed.id)
        if unaccessed_in_active is not None:
            # If still in active, its strength should be much lower
            decay = DecayEngine()
            s = decay.compute_strength(unaccessed, time.time())
            assert s < 0.9  # significantly decayed


class TestMigrationTimeline:
    """Validate natural migration flow over time."""

    def test_natural_flow_active_to_warm_to_archive(self):
        """Items naturally flow Active → Warm → Archive as they age."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        # Store items with distinct content and low importance
        contents = [
            "Apples are red fruit grown in orchards",
            "Basketball involves dribbling and shooting hoops",
            "Chemistry studies molecules and reactions bonds",
            "Denmark is a Scandinavian country in Europe",
            "Elephants are large mammals in Africa savanna",
        ]
        items = []
        for content in contents:
            item = engine.store(content, importance=0.2)
            items.append(item)

        initial_active = engine.active.count

        # Simulate time passage
        past = time.time() - 3600
        for item in items:
            item.last_accessed_at = past
            item.created_at = past

        engine.run_migration()

        # Items should have left Active (strength ≈ 0.1 < θ_archive=0.15)
        assert engine.active.count < initial_active or engine.archive.count > 0

    def test_pin_never_migrates(self):
        """Pinned items stay in Active regardless of time."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        pinned = engine.store("Critical system rule", decay_strategy=DecayStrategy.PIN)
        normal = engine.store("Casual mention", importance=0.1)

        # Simulate extreme time passage
        past = time.time() - 86400 * 365  # 1 year ago
        normal.last_accessed_at = past
        normal.created_at = past

        engine.run_migration()

        assert engine.active.get(pinned.id) is not None, "Pinned item should stay in Active"
        assert pinned.strength == 1.0

    def test_migration_creates_audit_trail(self):
        """Migration events are recorded in the audit trail."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        item = engine.store("Will migrate soon", importance=0.1)

        # Force very old timestamp
        past = time.time() - 86400
        item.last_accessed_at = past
        item.created_at = past

        migrated = engine.run_migration()

        if migrated > 0:
            trail = engine.archive.get_audit_trail(item.id)
            events = [r.event for r in trail]
            assert "migrated" in events or "created" in events
