"""MCP Server — exposes EbbingContext as an MCP service.

Passive mode: 6 tools available for LLM clients to call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ebbingcontext.config import EbbingConfig

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ebbingcontext.engine import MemoryEngine
from ebbingcontext.interface.tools import TOOL_DEFINITIONS
from ebbingcontext.models import SensitivityLevel


def create_server(engine: MemoryEngine | None = None) -> Server:
    """Create and configure the MCP server."""
    server = Server("ebbingcontext")
    if engine is None:
        engine = MemoryEngine()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = _dispatch(engine, name, arguments)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
        except KeyError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        except ValueError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        except PermissionError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _dispatch(engine: MemoryEngine, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Route a tool call to the engine."""
    if tool_name == "store_memory":
        items = engine.store_message(
            content=args["content"],
            source_type=args.get("source_type", "agent"),
            importance=args.get("importance"),
        )
        return {
            "status": "stored",
            "memory_ids": [item.id for item in items],
            "count": len(items),
            "items": [
                {
                    "memory_id": item.id,
                    "decay_strategy": item.decay_strategy.value,
                    "sensitivity": item.sensitivity.value,
                    "importance": item.importance,
                }
                for item in items
            ],
        }

    elif tool_name == "recall_memory":
        results = engine.recall(
            query=args["query"],
            top_k=args.get("top_k", 5),
        )
        return {
            "memories": [
                {
                    "memory_id": sm.item.id,
                    "content": sm.item.content,
                    "strength": round(sm.strength, 4),
                    "similarity": round(sm.similarity, 4),
                    "final_score": round(sm.final_score, 4),
                    "decay_strategy": sm.item.decay_strategy.value,
                }
                for sm in results
            ],
            "count": len(results),
        }

    elif tool_name == "pin_memory":
        item = engine.pin(args["memory_id"])
        return {
            "status": "pinned",
            "memory_id": item.id,
        }

    elif tool_name == "forget_memory":
        item = engine.forget(args["memory_id"])
        return {
            "status": "forgotten",
            "memory_id": item.id,
            "archived": True,
        }

    elif tool_name == "inspect_memory":
        return engine.inspect(args["memory_id"])

    elif tool_name == "transfer_memory":
        max_sens = SensitivityLevel(args.get("max_sensitivity", "internal"))
        item = engine.transfer(
            memory_id=args["memory_id"],
            from_agent=args["from_agent"],
            to_agent=args["to_agent"],
            max_sensitivity=max_sens,
        )
        if item is None:
            return {
                "status": "filtered",
                "reason": "Sensitivity level exceeds transfer limit.",
            }
        return {
            "status": "transferred",
            "new_memory_id": item.id,
            "to_agent": item.agent_id,
        }

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


async def run_server(
    engine: MemoryEngine | None = None,
    config: "EbbingConfig | None" = None,
) -> None:
    """Run the MCP server over stdio."""
    if engine is None and config is not None:
        engine = MemoryEngine.from_config(config)
    server = create_server(engine)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
