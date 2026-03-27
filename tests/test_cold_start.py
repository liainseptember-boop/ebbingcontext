"""Tests for the cold start module — template listing, loading, and applying."""

from unittest.mock import MagicMock

import pytest

from ebbingcontext.core.cold_start import apply_template, list_templates, load_template
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.models import MemoryItem


class TestListTemplates:
    def test_returns_four_templates(self):
        names = list_templates()
        assert len(names) == 4

    def test_returns_list_of_strings(self):
        names = list_templates()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_known_template_names_present(self):
        names = set(list_templates())
        assert {"general", "code_assistant", "customer_support", "research"} == names


class TestLoadTemplate:
    def test_general_returns_dict(self):
        data = load_template("general")
        assert isinstance(data, dict)

    def test_template_has_memories_key(self):
        data = load_template("general")
        assert "memories" in data
        assert isinstance(data["memories"], list)

    def test_nonexistent_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_template("nonexistent_template_xyz")

    def test_all_memories_have_content(self):
        """Every memory entry in every template must have a non-empty content field."""
        for name in list_templates():
            data = load_template(name)
            for i, entry in enumerate(data.get("memories", [])):
                assert "content" in entry, (
                    f"Template '{name}' memory index {i} missing 'content'"
                )
                assert entry["content"], (
                    f"Template '{name}' memory index {i} has empty content"
                )


class TestApplyTemplate:
    def _make_engine(self) -> MemoryEngine:
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        # Pre-store non-pinned items to dilute pin ratio below 30%
        topics = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                  "golf", "hotel", "india", "juliet", "kilo", "lima", "mike",
                  "november", "oscar", "papa", "quebec", "romeo", "sierra",
                  "tango", "uniform", "victor", "whiskey", "xray", "yankee",
                  "zulu", "red", "green", "blue", "yellow", "orange", "purple",
                  "silver", "gold", "bronze", "copper", "iron", "steel",
                  "north", "south", "east", "west", "center", "edge",
                  "spring", "summer", "autumn", "winter", "dawn", "dusk"]
        for i, topic in enumerate(topics):
            engine.store(f"{topic} information detail {i}", source_type="user")
        return engine

    def test_stores_correct_count(self):
        engine = self._make_engine()
        count = apply_template(engine, "general")
        data = load_template("general")
        expected = sum(1 for m in data.get("memories", []) if m.get("content"))
        assert count == expected

    def test_agent_id_passed_to_engine(self):
        """apply_template passes agent_id to engine.store."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        engine.store = MagicMock(return_value=MemoryItem(content="mock"))
        apply_template(engine, "general", agent_id="agent-42")
        for c in engine.store.call_args_list:
            assert c.kwargs.get("agent_id") == "agent-42"

    def test_skips_empty_content(self):
        engine = self._make_engine()
        engine.store = MagicMock()
        # Monkey-patch load_template to inject an entry with empty content
        import ebbingcontext.core.cold_start as cs
        original = cs.load_template
        cs.load_template = lambda name: {
            "memories": [
                {"content": "valid memory"},
                {"content": ""},
                {"content": None},
                {},
                {"content": "another valid"},
            ]
        }
        try:
            count = apply_template(engine, "dummy")
            assert count == 2
            assert engine.store.call_count == 2
        finally:
            cs.load_template = original

    def test_return_count_matches_stored(self):
        """Each template stores the expected number of memories."""
        for name in list_templates():
            engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
            # Mock store to avoid pin ratio limits and conflict detection
            stored_items = []
            def mock_store(**kwargs):
                item = MemoryItem(content=kwargs.get("content", ""))
                stored_items.append(item)
                return item
            engine.store = lambda **kw: mock_store(**kw)

            data = load_template(name)
            expected = sum(1 for m in data.get("memories", []) if m.get("content"))
            count = apply_template(engine, name)
            assert count == expected, f"Mismatch for template '{name}': got {count}, expected {expected}"
