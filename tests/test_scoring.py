"""Tests for the scoring engine."""

from ebbingcontext.core.decay import DecayEngine
from ebbingcontext.core.scoring import ScoredMemory, ScoringEngine
from ebbingcontext.models import MemoryItem, StorageLayer


def make_item(**kwargs) -> MemoryItem:
    defaults = {
        "content": "test memory",
        "importance": 0.5,
        "layer": StorageLayer.ACTIVE,
        "last_accessed_at": 1000.0,
        "created_at": 1000.0,
    }
    defaults.update(kwargs)
    return MemoryItem(**defaults)


class TestComputeFinalScore:
    def setup_method(self):
        self.engine = ScoringEngine(decay_engine=DecayEngine())

    def test_typical_values(self):
        assert abs(self.engine.compute_final_score(0.8, 0.9) - 0.72) < 1e-9

    def test_zero_similarity(self):
        assert self.engine.compute_final_score(0.0, 0.9) == 0.0

    def test_perfect_scores(self):
        assert self.engine.compute_final_score(1.0, 1.0) == 1.0


class TestRankMemories:
    def setup_method(self):
        self.decay = DecayEngine()
        self.engine = ScoringEngine(decay_engine=self.decay)
        self.current_time = 1000.0

    def test_sorted_descending_by_final_score(self):
        candidates = [
            (make_item(content="low"), 0.3),
            (make_item(content="high"), 0.9),
            (make_item(content="mid"), 0.6),
        ]
        result = self.engine.rank_memories(candidates, current_time=self.current_time)
        scores = [s.final_score for s in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0].item.content == "high"
        assert result[-1].item.content == "low"

    def test_top_k_limits_results(self):
        candidates = [
            (make_item(content=f"item-{i}"), 0.5)
            for i in range(10)
        ]
        result = self.engine.rank_memories(
            candidates, current_time=self.current_time, top_k=3
        )
        assert len(result) == 3

    def test_top_k_none_returns_all(self):
        candidates = [
            (make_item(content=f"item-{i}"), 0.5)
            for i in range(7)
        ]
        result = self.engine.rank_memories(
            candidates, current_time=self.current_time, top_k=None
        )
        assert len(result) == 7

    def test_top_k_zero_returns_empty(self):
        candidates = [
            (make_item(content="a"), 0.9),
            (make_item(content="b"), 0.8),
        ]
        result = self.engine.rank_memories(
            candidates, current_time=self.current_time, top_k=0
        )
        assert result == []

    def test_empty_candidates_returns_empty(self):
        result = self.engine.rank_memories([], current_time=self.current_time)
        assert result == []


class TestArrangeByPosition:
    def setup_method(self):
        self.engine = ScoringEngine(decay_engine=DecayEngine())

    def _make_scored(self, strength: float, label: str = "") -> ScoredMemory:
        return ScoredMemory(
            item=make_item(content=label),
            similarity=0.8,
            strength=strength,
            final_score=0.8 * strength,
        )

    def test_high_at_edges_medium_in_middle(self):
        scored = [
            self._make_scored(0.9, "h1"),
            self._make_scored(0.8, "h2"),
            self._make_scored(0.75, "h3"),
            self._make_scored(0.5, "m1"),
            self._make_scored(0.4, "m2"),
        ]
        arranged = self.engine.arrange_by_position(scored)
        contents = [s.item.content for s in arranged]
        # mid_point = (3+1)//2 = 2 => head = [h1, h2], tail = [h3]
        assert contents == ["h1", "h2", "m1", "m2", "h3"]

    def test_single_element_returned_as_is(self):
        scored = [self._make_scored(0.9, "only")]
        arranged = self.engine.arrange_by_position(scored)
        assert len(arranged) == 1
        assert arranged[0].item.content == "only"

    def test_all_high_strength_returned_as_is(self):
        scored = [
            self._make_scored(0.9, "h1"),
            self._make_scored(0.8, "h2"),
            self._make_scored(0.75, "h3"),
        ]
        arranged = self.engine.arrange_by_position(scored)
        contents = [s.item.content for s in arranged]
        assert contents == ["h1", "h2", "h3"]

    def test_all_medium_strength_returned_as_is(self):
        scored = [
            self._make_scored(0.5, "m1"),
            self._make_scored(0.4, "m2"),
            self._make_scored(0.3, "m3"),
        ]
        arranged = self.engine.arrange_by_position(scored)
        contents = [s.item.content for s in arranged]
        assert contents == ["m1", "m2", "m3"]
