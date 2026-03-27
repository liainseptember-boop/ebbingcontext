"""Tests for MCP dispatch function and tool definitions."""

import pytest

from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.interface.mcp_server import _dispatch, create_server
from ebbingcontext.interface.tools import TOOL_DEFINITIONS


class TestDispatchStore:
    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

    def test_store_memory_returns_stored(self):
        result = _dispatch(self.engine, "store_memory", {"content": "Remember this fact"})
        assert result["status"] == "stored"
        assert result["count"] > 0
        assert len(result["memory_ids"]) == result["count"]
        assert len(result["items"]) == result["count"]

    def test_store_memory_with_importance_override(self):
        result = _dispatch(
            self.engine,
            "store_memory",
            {"content": "High priority note", "importance": 0.95},
        )
        assert result["status"] == "stored"
        assert result["count"] > 0
        # Verify the importance was propagated to the stored item
        assert any(item["importance"] == 0.95 for item in result["items"])


class TestDispatchRecall:
    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        _dispatch(self.engine, "store_memory", {"content": "The sky is blue"})

    def test_recall_memory_returns_memories(self):
        result = _dispatch(self.engine, "recall_memory", {"query": "sky color"})
        assert "memories" in result
        assert isinstance(result["memories"], list)
        assert result["count"] == len(result["memories"])

    def test_recall_memory_with_custom_top_k(self):
        # Store several items
        for i in range(5):
            _dispatch(self.engine, "store_memory", {"content": f"Fact number {i}"})

        result = _dispatch(
            self.engine,
            "recall_memory",
            {"query": "fact", "top_k": 2},
        )
        assert result["count"] <= 2


class TestDispatchPin:
    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

    def _store_one(self) -> str:
        item = self.engine.store("Important information", source_type="user")
        return item.id

    def test_pin_memory_returns_pinned(self):
        memory_id = self._store_one()
        result = _dispatch(self.engine, "pin_memory", {"memory_id": memory_id})
        assert result["status"] == "pinned"
        assert result["memory_id"] == memory_id

    def test_pin_memory_invalid_id_raises_key_error(self):
        with pytest.raises(KeyError):
            _dispatch(self.engine, "pin_memory", {"memory_id": "nonexistent_id"})


class TestDispatchForget:
    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

    def _store_one(self) -> str:
        item = self.engine.store("Temporary information", source_type="user")
        return item.id

    def test_forget_memory_returns_forgotten(self):
        memory_id = self._store_one()
        result = _dispatch(self.engine, "forget_memory", {"memory_id": memory_id})
        assert result["status"] == "forgotten"
        assert result["memory_id"] == memory_id
        assert result["archived"] is True

    def test_forget_memory_invalid_id_raises_key_error(self):
        with pytest.raises(KeyError):
            _dispatch(self.engine, "forget_memory", {"memory_id": "nonexistent_id"})


class TestDispatchInspect:
    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

    def _store_one(self) -> str:
        item = self.engine.store("Inspectable fact", source_type="user")
        return item.id

    def test_inspect_memory_returns_details(self):
        memory_id = self._store_one()
        result = _dispatch(self.engine, "inspect_memory", {"memory_id": memory_id})
        assert isinstance(result, dict)
        assert "item" in result
        assert "current_strength" in result
        assert "stability" in result
        assert "audit_trail" in result

    def test_inspect_memory_invalid_id_raises_key_error(self):
        with pytest.raises(KeyError):
            _dispatch(self.engine, "inspect_memory", {"memory_id": "nonexistent_id"})


class TestDispatchTransfer:
    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

    def test_transfer_memory_returns_transferred(self):
        item = self.engine.store(
            "Shareable knowledge", agent_id="agent_a", source_type="user"
        )
        result = _dispatch(
            self.engine,
            "transfer_memory",
            {
                "memory_id": item.id,
                "from_agent": "agent_a",
                "to_agent": "agent_b",
            },
        )
        assert result["status"] == "transferred"
        assert "new_memory_id" in result
        assert result["to_agent"] == "agent_b"


class TestDispatchUnknown:
    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

    def test_unknown_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            _dispatch(self.engine, "nonexistent_tool", {})


class TestToolDefinitions:
    def test_has_exactly_six_tools(self):
        assert len(TOOL_DEFINITIONS) == 6

    def test_each_tool_has_required_keys(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing 'description'"
            assert "inputSchema" in tool, f"Tool {tool.get('name')} missing 'inputSchema'"


class TestCreateServer:
    def test_returns_server_instance(self):
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        server = create_server(engine)
        from mcp.server import Server

        assert isinstance(server, Server)

    def test_works_with_default_engine(self):
        server = create_server(None)
        from mcp.server import Server

        assert isinstance(server, Server)
