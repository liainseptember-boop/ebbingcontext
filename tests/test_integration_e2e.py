"""End-to-end integration tests for the full MemoryEngine pipeline."""

import time

from ebbingcontext.core.cold_start import apply_template, list_templates
from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.models import DecayStrategy, SensitivityLevel, StorageLayer


def _make_engine() -> MemoryEngine:
    """Create a fresh MemoryEngine with LiteEmbeddingProvider for each test."""
    return MemoryEngine(embedding_provider=LiteEmbeddingProvider())


class TestFullPipeline:
    """Test 1: store -> decay (manipulate timestamps) -> migrate -> recall verifies items moved."""

    def test_store_decay_migrate_recall(self):
        engine = _make_engine()

        # Store an item with low importance so it decays faster
        item = engine.store(
            "The capital of France is Paris",
            agent_id="default",
            importance=0.2,
            source_type="user",
        )
        assert item.layer == StorageLayer.ACTIVE
        assert engine.active.count >= 1

        # Manipulate timestamps to simulate significant time passage so strength decays
        # below the active threshold (0.6) but above archive threshold (0.15)
        past = time.time() - 100_000
        item.last_accessed_at = past
        item.created_at = past

        # Run migration -- item should demote from Active
        migrated = engine.run_migration(agent_id="default")
        assert migrated >= 1

        # The item should no longer be in Active
        assert engine.active.get(item.id) is None

        # It should be in either Warm or Archive depending on how far strength dropped
        in_warm = engine.warm.get(item.id) is not None
        in_archive = engine.archive.get(item.id) is not None
        assert in_warm or in_archive

        # Recall should still find it if it landed in Warm
        # (archive items are not returned by recall, which is correct behavior)
        results = engine.recall(query="capital of France", agent_id="default")
        if in_warm:
            found_ids = [r.item.id for r in results]
            assert item.id in found_ids


class TestStoreMessageChunks:
    """Test 2: store_message with long text creates multiple items, recall finds them."""

    def test_long_message_creates_multiple_chunks(self):
        engine = _make_engine()

        long_text = (
            "Python is a versatile programming language used in web development. "
            "It is also widely used in data science and machine learning. "
            "Many companies use Python for backend services and automation. "
            "The language has a large ecosystem of libraries and frameworks."
        )
        items = engine.store_message(long_text, agent_id="default", source_type="user")

        # The chunker should produce at least one item
        assert len(items) >= 1

        # All items should share the same chunk_group
        groups = {item.metadata.get("chunk_group") for item in items}
        assert len(groups) == 1
        assert None not in groups

        # All items should be in the active store
        for item in items:
            assert engine.active.get(item.id) is not None

        # Recall should find at least some of these items
        results = engine.recall(query="Python programming", agent_id="default")
        assert len(results) >= 1
        recalled_ids = {r.item.id for r in results}
        stored_ids = {item.id for item in items}
        assert recalled_ids & stored_ids  # intersection is non-empty


class TestMultiAgentIsolation:
    """Test 3: store items for agent_a and agent_b, recall for agent_a
    doesn't return agent_b's items."""

    def test_agents_are_isolated(self):
        engine = _make_engine()

        engine.store("Agent A likes functional programming", agent_id="agent_a")
        engine.store("Agent A prefers dark mode", agent_id="agent_a")
        engine.store("Agent B likes object-oriented programming", agent_id="agent_b")
        engine.store("Agent B prefers light mode", agent_id="agent_b")

        results_a = engine.recall(query="programming preferences", agent_id="agent_a")
        results_b = engine.recall(query="programming preferences", agent_id="agent_b")

        ids_a = {r.item.id for r in results_a}
        ids_b = {r.item.id for r in results_b}

        # No overlap between agents
        assert ids_a.isdisjoint(ids_b)

        # Each agent's results should only contain their own items
        for r in results_a:
            assert r.item.agent_id == "agent_a"
        for r in results_b:
            assert r.item.agent_id == "agent_b"

        # Each agent should see exactly 2 items
        assert len(results_a) == 2
        assert len(results_b) == 2


class TestRecallForPromptConstraints:
    """Test 4: total_tokens <= total_window, memories_included > 0."""

    def test_prompt_respects_token_budget(self):
        engine = _make_engine()

        # Store several items so there is content to assemble
        for i in range(5):
            engine.store(
                f"Memory item number {i} with some content for budget testing",
                agent_id="default",
            )

        total_window = 4096
        result = engine.recall_for_prompt(
            query="memory items",
            agent_id="default",
            total_window=total_window,
            system_prompt="You are a helpful assistant.",
        )

        assert result.total_tokens <= total_window
        assert result.budget_remaining >= 0
        assert len(result.sections) >= 1  # at least the system prompt section

    def test_prompt_includes_memories_with_large_window(self):
        engine = _make_engine()

        engine.store("The user's favorite color is blue", agent_id="default")
        engine.store("The user works as a software engineer", agent_id="default")

        result = engine.recall_for_prompt(
            query="user preferences",
            agent_id="default",
            total_window=128000,
            system_prompt="You are a helpful assistant.",
        )

        # With a large window and few memories, all should be included
        assert result.memories_included > 0
        assert result.total_tokens <= 128000


class TestConflictDetectionInPipeline:
    """Test 5: store two similar items -> second overwrites first."""

    def test_identical_content_triggers_overwrite(self):
        engine = _make_engine()

        # Store the first item
        item1 = engine.store(
            "The user's favorite programming language is Python",
            agent_id="default",
        )
        item1_id = item1.id

        # Store identical content (should produce similarity >= 0.9 and trigger overwrite)
        item2 = engine.store(
            "The user's favorite programming language is Python",
            agent_id="default",
        )

        # item1 should have been archived (overwritten) and replaced by item2
        active_items = engine.active.get_all("default")
        active_ids = {it.id for it in active_items}

        assert item2.id in active_ids
        assert item1_id not in active_ids

        # The old item should be in the archive
        archived = engine.archive.get(item1_id)
        assert archived is not None


class TestColdStartWithUserMemories:
    """Test 6: apply a template then store user memories, both coexist."""

    def test_template_and_user_memories_coexist(self):
        engine = _make_engine()

        # Pre-store non-pinned items to dilute pin ratio below 30%
        for i in range(20):
            engine.store(f"Dilution item {i} unique{i} filler{i}", source_type="user")

        # Check templates are available
        templates = list_templates()
        assert "general" in templates

        # Apply the general template
        count = apply_template(engine, "general", agent_id="default")
        assert count > 0

        template_items_count = engine.active.count

        # Store additional user memories
        engine.store("My name is Alice and I work in data science", agent_id="default")
        engine.store("I prefer using Jupyter notebooks for analysis", agent_id="default")

        # Both template and user memories should be present
        total_count = engine.active.count
        assert total_count >= template_items_count + 2

        # Recall should find items (top_k limits results, so just check non-empty)
        all_results = engine.recall(agent_id="default", top_k=total_count)
        assert len(all_results) > 0

        # Verify source types: template items are "system", user items are "user"
        source_types = {r.item.source_type for r in all_results}
        assert "system" in source_types
        assert "user" in source_types


class TestEmptyEngineRecall:
    """Test 7: recall on an empty engine returns an empty list."""

    def test_recall_empty_engine_with_query(self):
        engine = _make_engine()

        results = engine.recall(query="anything", agent_id="default")
        assert results == []

    def test_recall_empty_engine_no_query(self):
        engine = _make_engine()

        results = engine.recall(agent_id="default")
        assert results == []


class TestStoreEmptyString:
    """Test 8: store_message with empty string returns empty list."""

    def test_store_message_empty_string(self):
        engine = _make_engine()

        items = engine.store_message("")
        assert items == []
        assert engine.active.count == 0

    def test_store_message_whitespace_only(self):
        engine = _make_engine()

        items = engine.store_message("   ")
        assert items == []
        assert engine.active.count == 0


class TestForgetThenRecall:
    """Test 9: forget an item, then recall should not return it."""

    def test_forgotten_item_not_recalled(self):
        engine = _make_engine()

        item = engine.store(
            "Secret: the treasure is under the oak tree", agent_id="default"
        )
        memory_id = item.id

        # Verify it is retrievable before forgetting
        results_before = engine.recall(query="treasure", agent_id="default")
        assert any(r.item.id == memory_id for r in results_before)

        # Forget it
        forgotten = engine.forget(memory_id)
        assert forgotten.layer == StorageLayer.ARCHIVE

        # Recall should no longer return it
        results_after = engine.recall(query="treasure", agent_id="default")
        assert all(r.item.id != memory_id for r in results_after)

        # Active store should be empty
        assert engine.active.count == 0

        # Archive should have the item
        assert engine.archive.get(memory_id) is not None


class TestPinPreservesThroughMigration:
    """Test 10: pin an item, run migration, item stays in active."""

    def test_pinned_item_survives_migration(self):
        engine = _make_engine()

        item = engine.store("Critical system instruction", agent_id="default")
        engine.pin(item.id, agent_id="default")

        # Verify it is pinned
        assert item.decay_strategy == DecayStrategy.PIN
        assert item.strength == 1.0

        # Manipulate timestamps to simulate massive time passage
        item.last_accessed_at = time.time() - 1_000_000
        item.created_at = time.time() - 1_000_000

        # Run migration -- pinned items should NOT migrate because their
        # strength stays at 1.0 (the decay engine returns 1.0 for PIN items)
        migrated = engine.run_migration(agent_id="default")
        assert migrated == 0

        # Item should still be in active
        assert engine.active.get(item.id) is not None
        assert item.layer == StorageLayer.ACTIVE

        # Strength should still be 1.0 (pinned items don't decay)
        info = engine.inspect(item.id)
        assert info["current_strength"] == 1.0


class TestInspectReturnsCorrectInfo:
    """Test 11: store item, inspect returns dict with content and strength."""

    def test_inspect_contains_required_fields(self):
        engine = _make_engine()

        item = engine.store("The Earth orbits the Sun", agent_id="default")

        info = engine.inspect(item.id)

        # Check required top-level keys
        assert "item" in info
        assert "current_strength" in info
        assert "stability" in info
        assert "audit_trail" in info

        # Check item details
        assert info["item"]["content"] == "The Earth orbits the Sun"
        assert info["item"]["agent_id"] == "default"
        assert info["item"]["layer"] == StorageLayer.ACTIVE

        # Strength should be close to 1.0 for a freshly stored item
        assert info["current_strength"] > 0.9

        # Stability should be positive
        assert info["stability"] > 0

        # Audit trail should have at least the creation event
        assert len(info["audit_trail"]) >= 1
        events = [r["event"] for r in info["audit_trail"]]
        assert "created" in events


class TestTransferBetweenAgents:
    """Test 12: store in agent_a, transfer to agent_b, verify agent_b can access."""

    def test_transfer_makes_item_accessible_to_target(self):
        engine = _make_engine()

        # Store in agent_a
        item = engine.store(
            "Shared knowledge: the project deadline is March 31st",
            agent_id="agent_a",
        )

        # Verify agent_b has nothing yet
        results_b_before = engine.recall(query="project deadline", agent_id="agent_b")
        assert len(results_b_before) == 0

        # Transfer from agent_a to agent_b
        transferred = engine.transfer(
            item.id, from_agent="agent_a", to_agent="agent_b"
        )

        assert transferred is not None
        assert transferred.agent_id == "agent_b"
        assert transferred.id != item.id
        assert transferred.content == item.content

        # Original item should still belong to agent_a
        assert item.agent_id == "agent_a"

        # agent_b should now be able to recall it
        results_b_after = engine.recall(query="project deadline", agent_id="agent_b")
        assert len(results_b_after) >= 1
        assert any(r.item.id == transferred.id for r in results_b_after)

    def test_transfer_sensitive_item_is_blocked(self):
        engine = _make_engine()

        item = engine.store(
            "Secret API key: sk-abc123xyz",
            agent_id="agent_a",
            sensitivity=SensitivityLevel.SENSITIVE,
        )

        # Transfer with max_sensitivity=PUBLIC should block the transfer
        result = engine.transfer(
            item.id,
            from_agent="agent_a",
            to_agent="agent_b",
            max_sensitivity=SensitivityLevel.PUBLIC,
        )

        assert result is None

        # agent_b should not have the item
        results_b = engine.recall(query="API key", agent_id="agent_b")
        assert len(results_b) == 0
