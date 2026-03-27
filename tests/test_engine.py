"""Integration tests for MemoryEngine."""

from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.models import DecayStrategy, SensitivityLevel, StorageLayer


class TestMemoryEngine:
    def setup_method(self):
        self.engine = MemoryEngine()

    def test_store_and_recall(self):
        item = self.engine.store("Remember that the user likes Python", source_type="user")
        assert item.id is not None
        assert item.layer == StorageLayer.ACTIVE
        assert self.engine.active.count == 1

        # Recall without embedding returns all active with default similarity
        results = self.engine.recall()
        assert len(results) == 1
        assert results[0].item.id == item.id

    def test_store_system_prompt_is_pinned(self):
        item = self.engine.store("You are a helpful assistant", source_type="system")
        assert item.decay_strategy == DecayStrategy.PIN
        assert item.importance >= 0.8

    def test_store_with_override(self):
        item = self.engine.store(
            "Custom importance item",
            importance=0.95,
            decay_strategy=DecayStrategy.PIN,
            sensitivity=SensitivityLevel.SENSITIVE,
        )
        assert item.importance == 0.95
        assert item.decay_strategy == DecayStrategy.PIN
        assert item.sensitivity == SensitivityLevel.SENSITIVE

    def test_pin_memory(self):
        item = self.engine.store("Some info", source_type="user")
        assert item.decay_strategy != DecayStrategy.PIN

        pinned = self.engine.pin(item.id)
        assert pinned.decay_strategy == DecayStrategy.PIN
        assert pinned.strength == 1.0

    def test_pin_limit(self):
        # Add non-pinned items first to establish a pool
        for i in range(7):
            self.engine.store(f"Normal item {i}", source_type="user")

        # Now add pinned items up to the 30% limit (30% of 7 ~ 2 pinned OK)
        self.engine.store("Pinned 1", decay_strategy=DecayStrategy.PIN)
        self.engine.store("Pinned 2", decay_strategy=DecayStrategy.PIN)

        # 2 pinned / 9 total ≈ 22%, next pin makes 3/10 = 30% which hits the limit
        try:
            self.engine.store("Pinned 3", decay_strategy=DecayStrategy.PIN)
            # 3/10 = 30% which is >= 30%, so should fail
            # But the check is done BEFORE adding, so 2/9 ≈ 22% < 30% → succeeds
            # Then 3/10, next one: 3/10 = 30% >= 30% → fails
            self.engine.store("Pinned 4", decay_strategy=DecayStrategy.PIN)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Pin ratio limit" in str(e)

    def test_forget_memory(self):
        item = self.engine.store("Forget me")
        memory_id = item.id
        assert self.engine.active.count == 1

        forgotten = self.engine.forget(memory_id)
        assert self.engine.active.count == 0
        assert forgotten.layer == StorageLayer.ARCHIVE
        assert self.engine.archive.count == 1

    def test_inspect_memory(self):
        item = self.engine.store("Inspectable")
        info = self.engine.inspect(item.id)
        assert "item" in info
        assert "current_strength" in info
        assert "stability" in info
        assert "audit_trail" in info
        assert len(info["audit_trail"]) >= 1

    def test_transfer_memory(self):
        item = self.engine.store("Transfer this", agent_id="agent_a")
        transferred = self.engine.transfer(
            item.id, from_agent="agent_a", to_agent="agent_b"
        )
        assert transferred is not None
        assert transferred.agent_id == "agent_b"
        assert transferred.id != item.id

    def test_transfer_blocked_by_sensitivity(self):
        item = self.engine.store(
            "Secret: api_key=abc123",
            agent_id="agent_a",
            sensitivity=SensitivityLevel.SENSITIVE,
        )
        result = self.engine.transfer(
            item.id,
            from_agent="agent_a",
            to_agent="agent_b",
            max_sensitivity=SensitivityLevel.INTERNAL,
        )
        assert result is None

    def test_audit_trail(self):
        item = self.engine.store("Auditable")
        self.engine.forget(item.id)
        trail = self.engine.archive.get_audit_trail(item.id)
        events = [r.event for r in trail]
        assert "created" in events
        assert "forgotten" in events

    def test_multiple_agents_isolated(self):
        self.engine.store("Agent A memory", agent_id="a")
        self.engine.store("Agent B memory", agent_id="b")

        results_a = self.engine.recall(agent_id="a")
        results_b = self.engine.recall(agent_id="b")
        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0].item.content == "Agent A memory"
        assert results_b[0].item.content == "Agent B memory"


class TestMemoryEngineWithEmbedding:
    """Tests with LiteEmbeddingProvider to verify semantic retrieval."""

    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider(dimension=256))

    def test_store_auto_embeds(self):
        item = self.engine.store("The weather is sunny today")
        assert item.embedding is not None
        assert len(item.embedding) == 256

    def test_recall_with_query_has_embeddings(self):
        """Recall with a query should use embedding-based scoring, not flat 0.5."""
        self.engine.store("Python is a great programming language for data science")
        self.engine.store("The weather forecast says rain and storms tomorrow")

        results = self.engine.recall(query="Python programming")
        assert len(results) == 2
        # All items should have been scored via embeddings (not all 0.5)
        for r in results:
            assert r.item.embedding is not None

    def test_recall_semantic_scores_vary(self):
        """Different content should produce different final scores."""
        self.engine.store("Python is a programming language for data science")
        self.engine.store("Elephants are the largest land animals on earth")
        self.engine.store("Quantum mechanics describes subatomic particle behavior")

        results = self.engine.recall(query="Python programming data")
        assert len(results) == 3
        # Python-related memory should rank first
        assert "Python" in results[0].item.content

    def test_store_with_explicit_embedding_skips_provider(self):
        custom_emb = [0.1] * 256
        item = self.engine.store("test", embedding=custom_emb)
        assert item.embedding == custom_emb
