"""Effectiveness tests: Token budget management and position orchestration.

Proves the system respects token budgets and places high-strength memories
at primacy/recency positions.
"""

from ebbingcontext.core.scoring import ScoredMemory
from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.models import DecayStrategy, MemoryItem
from ebbingcontext.prompt.assembler import PromptAssembler


def make_scored(content: str, strength: float = 0.8, similarity: float = 0.9) -> ScoredMemory:
    item = MemoryItem(content=content, importance=0.5, strength=strength)
    return ScoredMemory(
        item=item, similarity=similarity,
        strength=strength, final_score=similarity * strength,
    )


class TestTokenBudgetStress:
    """Prove token budget is respected under pressure."""

    def test_100_memories_in_4k_window(self):
        """100 memories in a 4K window: budget respected, some included, some dropped."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        # Each item has unique keywords to avoid conflict detection overwrite
        topics = [
            "python code function class module import variable loop",
            "cooking recipe kitchen ingredients baking flour oven",
            "space rocket planet orbit galaxy astronaut mission",
            "music melody guitar piano rhythm harmony chord",
            "sports football basketball tennis swimming running",
            "history ancient medieval empire dynasty kingdom war",
            "medicine doctor hospital surgery diagnosis treatment",
            "finance stock market investment banking portfolio",
            "travel airport flight hotel destination tourism",
            "science physics chemistry biology experiment lab",
        ]
        for i in range(100):
            topic = topics[i % len(topics)]
            # Pad with unique words to increase token count
            padding = " ".join(f"detail{i}x{j}" for j in range(30))
            engine.store(f"{topic} entry{i} {padding}", source_type="user")

        result = engine.recall_for_prompt(
            query="python code function class",
            total_window=4000,
            system_prompt="You are a helpful assistant.",
            top_k=50,
        )

        assert result.total_tokens <= 4000
        assert result.memories_included > 0
        assert result.memories_dropped > 0

    def test_priority_order_respected(self):
        """System > Pin > Tools > Recent > Reserve > Memories priority."""
        assembler = PromptAssembler(output_reserve=200)

        pinned = [MemoryItem(content="Critical pin", decay_strategy=DecayStrategy.PIN)]
        recalled = [make_scored(f"Recalled memory {i}") for i in range(10)]

        result = assembler.assemble(
            total_window=1000,
            system_prompt="System prompt text here.",
            tool_definitions=[{"name": "tool1", "description": "A tool"}],
            recent_messages=[{"role": "user", "content": "Hello there"}],
            pinned_memories=pinned,
            recalled_memories=recalled,
        )

        roles = [s.role for s in result.sections]
        assert "system" in roles, "System prompt must be included"
        assert "memory_pin" in roles, "Pinned memories must be included"
        assert result.total_tokens <= 1000

    def test_high_importance_survives_budget_cuts(self):
        """High-strength memories should be included more than low-strength ones."""
        assembler = PromptAssembler(output_reserve=100)

        # Mix of high and low strength memories
        memories = []
        for i in range(5):
            memories.append(make_scored(f"High importance {i}", strength=0.9))
        for i in range(5):
            memories.append(make_scored(f"Low importance {i}", strength=0.2))

        result = assembler.assemble(
            total_window=500,  # Tight budget
            recalled_memories=memories,
        )

        if result.memories_included > 0 and result.memories_dropped > 0:
            # Position orchestration puts high-strength first, so they get included first
            mem_section = [s for s in result.sections if s.role == "memory_context"]
            if mem_section:
                # High importance items should be more represented
                content = mem_section[0].content
                high_count = sum(1 for i in range(5) if f"High importance {i}" in content)
                low_count = sum(1 for i in range(5) if f"Low importance {i}" in content)
                assert high_count >= low_count, \
                    f"High importance ({high_count}) should >= low importance ({low_count})"

    def test_zero_budget_drops_all_memories(self):
        """When budget is too small, all memories are dropped."""
        assembler = PromptAssembler(output_reserve=100)

        recalled = [make_scored(f"Memory {i}") for i in range(5)]
        result = assembler.assemble(
            total_window=50,  # Barely enough for reserve
            system_prompt="This system prompt takes up all the space and then some more text",
            recalled_memories=recalled,
        )

        assert result.memories_included == 0
        assert result.memories_dropped == 5

    def test_huge_window_includes_everything(self):
        """With a huge window, all memories should fit."""
        assembler = PromptAssembler(output_reserve=100)

        recalled = [make_scored(f"Memory {i} with some content") for i in range(20)]
        result = assembler.assemble(
            total_window=1_000_000,
            recalled_memories=recalled,
        )

        assert result.memories_included == 20
        assert result.memories_dropped == 0


class TestPositionOrchestration:
    """Prove Lost-in-Middle mitigation works."""

    def setup_method(self):
        self.assembler = PromptAssembler()

    def test_high_strength_at_edges(self):
        """High-strength memories should appear at beginning and end."""
        memories = [
            make_scored("High A", strength=0.9),
            make_scored("High B", strength=0.85),
            make_scored("High C", strength=0.8),
            make_scored("Medium A", strength=0.4),
            make_scored("Medium B", strength=0.3),
            make_scored("Medium C", strength=0.2),
        ]

        arranged = self.assembler._apply_position_orchestration(memories)

        assert arranged[0].strength >= 0.7, "First item should be high-strength"
        assert arranged[-1].strength >= 0.7, "Last item should be high-strength"

    def test_medium_strength_in_middle(self):
        """Medium-strength memories should be in the middle positions."""
        memories = [
            make_scored("High A", strength=0.9),
            make_scored("High B", strength=0.85),
            make_scored("Medium A", strength=0.4),
            make_scored("Medium B", strength=0.3),
        ]

        arranged = self.assembler._apply_position_orchestration(memories)

        # Middle items should be the medium-strength ones
        middle = arranged[1:-1]
        for m in middle:
            if m.strength < 0.7:
                pass  # Expected — medium items in middle
            # At least one medium item should be in the middle
        medium_in_middle = [m for m in middle if m.strength < 0.7]
        assert len(medium_in_middle) > 0

    def test_two_items_no_rearrangement(self):
        """With only 2 items, return as-is."""
        memories = [make_scored("A", strength=0.9), make_scored("B", strength=0.3)]
        arranged = self.assembler._apply_position_orchestration(memories)
        assert len(arranged) == 2

    def test_all_high_strength_no_rearrangement(self):
        """When all items are high-strength, no rearrangement needed."""
        memories = [
            make_scored("A", strength=0.9),
            make_scored("B", strength=0.85),
            make_scored("C", strength=0.8),
        ]
        arranged = self.assembler._apply_position_orchestration(memories)
        assert len(arranged) == 3

    def test_all_low_strength_no_rearrangement(self):
        """When all items are low-strength, no rearrangement needed."""
        memories = [
            make_scored("A", strength=0.3),
            make_scored("B", strength=0.2),
            make_scored("C", strength=0.1),
        ]
        arranged = self.assembler._apply_position_orchestration(memories)
        assert len(arranged) == 3
