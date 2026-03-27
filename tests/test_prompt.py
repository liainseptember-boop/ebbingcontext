"""Tests for prompt assembler."""

from ebbingcontext.adapter.token_counter import TokenCounter
from ebbingcontext.core.scoring import ScoredMemory
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.models import DecayStrategy, MemoryItem
from ebbingcontext.prompt.assembler import PromptAssembler


def make_scored(content: str, strength: float = 0.8, similarity: float = 0.9) -> ScoredMemory:
    item = MemoryItem(content=content, importance=0.5, strength=strength)
    return ScoredMemory(
        item=item,
        similarity=similarity,
        strength=strength,
        final_score=similarity * strength,
    )


class TestPromptAssembler:
    def setup_method(self):
        self.assembler = PromptAssembler(output_reserve=100)
        self.counter = TokenCounter()

    def test_empty_assembly(self):
        result = self.assembler.assemble(total_window=4000)
        assert result.total_tokens == 100  # Just output_reserve
        assert result.memories_included == 0
        assert result.memories_dropped == 0

    def test_system_prompt_included(self):
        result = self.assembler.assemble(
            total_window=4000,
            system_prompt="You are a helpful assistant.",
        )
        assert any(s.role == "system" for s in result.sections)
        assert result.total_tokens > 100

    def test_pinned_always_included(self):
        pinned = [
            MemoryItem(content="Critical system rule", decay_strategy=DecayStrategy.PIN),
            MemoryItem(content="Another pinned fact", decay_strategy=DecayStrategy.PIN),
        ]
        result = self.assembler.assemble(
            total_window=4000,
            pinned_memories=pinned,
        )
        pin_section = [s for s in result.sections if s.role == "memory_pin"]
        assert len(pin_section) == 1
        assert "Critical system rule" in pin_section[0].content
        assert len(pin_section[0].source_ids) == 2

    def test_token_budget_respected(self):
        # Very tight budget
        recalled = [make_scored(f"Memory item number {i} with some content") for i in range(20)]
        result = self.assembler.assemble(
            total_window=200,  # Very small window
            system_prompt="System prompt takes tokens.",
            recalled_memories=recalled,
        )
        assert result.total_tokens <= 200
        assert result.memories_dropped > 0

    def test_zero_budget_for_memories(self):
        # System prompt + reserve > total window
        result = self.assembler.assemble(
            total_window=50,
            system_prompt="This system prompt is quite long and takes up tokens",
            recalled_memories=[make_scored("Memory 1"), make_scored("Memory 2")],
        )
        assert result.memories_included == 0
        assert result.memories_dropped == 2

    def test_position_orchestration_applied(self):
        # Mix of high and low strength memories
        memories = [
            make_scored("High strength A", strength=0.9),
            make_scored("High strength B", strength=0.85),
            make_scored("Medium strength", strength=0.5),
            make_scored("High strength C", strength=0.8),
        ]
        arranged = self.assembler._apply_position_orchestration(memories)
        # High strength should be at beginning and end
        assert arranged[0].strength >= 0.7
        assert arranged[-1].strength >= 0.7

    def test_recent_turns_included(self):
        result = self.assembler.assemble(
            total_window=4000,
            recent_messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )
        recent_section = [s for s in result.sections if s.role == "recent_turns"]
        assert len(recent_section) == 1


class TestEngineRecallForPrompt:
    def test_recall_for_prompt_basic(self):
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        engine.store("Important fact about Python", source_type="user")
        engine.store("Another fact about data science", source_type="user")

        result = engine.recall_for_prompt(
            query="Python",
            total_window=4000,
            system_prompt="You are helpful.",
        )
        assert result.total_tokens <= 4000
        assert result.memories_included >= 0

    def test_recall_for_prompt_with_pin(self):
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        engine.store("You must always be polite", decay_strategy=DecayStrategy.PIN)
        engine.store("User likes Python", source_type="user")

        result = engine.recall_for_prompt(total_window=4000)
        pin_sections = [s for s in result.sections if s.role == "memory_pin"]
        assert len(pin_sections) == 1
        assert "polite" in pin_sections[0].content
