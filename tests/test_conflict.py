"""Tests for conflict detection and resolution."""

from ebbingcontext.core.conflict import ConflictResolver, ConflictType
from ebbingcontext.models import MemoryItem


def make_item(content: str = "test memory") -> MemoryItem:
    return MemoryItem(content=content)


class TestConflictResolver:
    def setup_method(self):
        self.resolver = ConflictResolver()

    # 1. Empty candidates -> UNRELATED
    def test_empty_candidates_returns_unrelated(self):
        result = self.resolver.detect(make_item(), candidates=[])
        assert result.conflict_type == ConflictType.UNRELATED
        assert result.existing_item is None
        assert result.similarity == 0.0

    # 2. similarity=0.95 -> OVERWRITE
    def test_high_similarity_returns_overwrite(self):
        existing = make_item("existing memory")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(existing, 0.95)],
        )
        assert result.conflict_type == ConflictType.OVERWRITE
        assert result.existing_item is existing
        assert result.similarity == 0.95

    # 3. similarity=0.9 (exact boundary) -> OVERWRITE
    def test_exact_overwrite_boundary_returns_overwrite(self):
        existing = make_item("existing memory")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(existing, 0.9)],
        )
        assert result.conflict_type == ConflictType.OVERWRITE
        assert result.existing_item is existing
        assert result.similarity == 0.9

    # 4. similarity=0.89 -> ASSOCIATE
    def test_below_overwrite_threshold_returns_associate(self):
        existing = make_item("existing memory")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(existing, 0.89)],
        )
        assert result.conflict_type == ConflictType.ASSOCIATE
        assert result.existing_item is existing
        assert result.similarity == 0.89

    # 5. similarity=0.7 (exact boundary) -> ASSOCIATE
    def test_exact_association_boundary_returns_associate(self):
        existing = make_item("existing memory")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(existing, 0.7)],
        )
        assert result.conflict_type == ConflictType.ASSOCIATE
        assert result.existing_item is existing
        assert result.similarity == 0.7

    # 6. similarity=0.69 -> UNRELATED
    def test_below_association_threshold_returns_unrelated(self):
        existing = make_item("existing memory")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(existing, 0.69)],
        )
        assert result.conflict_type == ConflictType.UNRELATED
        assert result.existing_item is None
        assert result.similarity == 0.69

    # 7. similarity=0.0 -> UNRELATED
    def test_zero_similarity_returns_unrelated(self):
        existing = make_item("existing memory")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(existing, 0.0)],
        )
        assert result.conflict_type == ConflictType.UNRELATED
        assert result.existing_item is None
        assert result.similarity == 0.0

    # 8. Multiple candidates: picks the first (highest similarity)
    def test_multiple_candidates_uses_first(self):
        high = make_item("high similarity")
        low = make_item("low similarity")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(high, 0.95), (low, 0.5)],
        )
        assert result.conflict_type == ConflictType.OVERWRITE
        assert result.existing_item is high
        assert result.similarity == 0.95

    # 9. Custom thresholds work
    def test_custom_thresholds(self):
        resolver = ConflictResolver(
            auto_overwrite_threshold=0.8,
            association_threshold=0.5,
        )
        existing = make_item("existing memory")

        # 0.85 >= 0.8 -> OVERWRITE with custom threshold
        result = resolver.detect(
            make_item(), candidates=[(existing, 0.85)],
        )
        assert result.conflict_type == ConflictType.OVERWRITE

        # 0.6 >= 0.5 but < 0.8 -> ASSOCIATE with custom threshold
        result = resolver.detect(
            make_item(), candidates=[(existing, 0.6)],
        )
        assert result.conflict_type == ConflictType.ASSOCIATE

        # 0.4 < 0.5 -> UNRELATED with custom threshold
        result = resolver.detect(
            make_item(), candidates=[(existing, 0.4)],
        )
        assert result.conflict_type == ConflictType.UNRELATED

    # 10. UNRELATED result still carries best_sim value
    def test_unrelated_carries_similarity_value(self):
        existing = make_item("existing memory")
        result = self.resolver.detect(
            make_item("new memory"),
            candidates=[(existing, 0.42)],
        )
        assert result.conflict_type == ConflictType.UNRELATED
        assert result.similarity == 0.42
