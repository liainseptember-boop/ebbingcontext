"""MCP Tool definitions — 6 tools for LLM interaction.

Tools:
    - store_memory: Store information
    - recall_memory: Retrieve relevant memories
    - pin_memory: Mark as unforgettable
    - forget_memory: Explicitly remove (→ Archive)
    - inspect_memory: View memory state
    - transfer_memory: Transfer between agents
"""

from __future__ import annotations



TOOL_DEFINITIONS = [
    {
        "name": "store_memory",
        "description": (
            "Store a piece of information in memory. The system automatically classifies "
            "it with a decay strategy (pin/decay-recoverable/decay-irreversible) and "
            "sensitivity level (public/internal/sensitive). You decide WHAT to store; "
            "the system decides HOW to store it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to store.",
                },
                "importance": {
                    "type": "number",
                    "description": "Optional importance override (0.0-1.0). Leave empty for auto-assessment.",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["user", "agent", "tool"],
                    "description": "Source of this information.",
                    "default": "agent",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Retrieve relevant memories for a query. Returns memories ranked by "
            "similarity × strength. Searches Active layer first, then Warm layer "
            "if Active results are insufficient."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for relevant memories.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of memories to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "pin_memory",
        "description": (
            "Pin a memory to prevent it from decaying. Pinned memories always stay "
            "in the Active layer. Use this for critical information that must not be "
            "forgotten. Subject to pin ratio limits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to pin.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "forget_memory",
        "description": (
            "Explicitly forget a memory. The memory is moved to Archive (not deleted) "
            "for audit purposes. Use this when information is confirmed to be outdated "
            "or incorrect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to forget.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "inspect_memory",
        "description": (
            "View the current state of a memory: its strength, classification, "
            "decay path, and audit trail."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to inspect.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "transfer_memory",
        "description": (
            "Transfer a memory from one agent to another. Filtered by sensitivity "
            "level and permissions. Sensitive memories are not transferred by default."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to transfer.",
                },
                "from_agent": {
                    "type": "string",
                    "description": "Source agent ID.",
                },
                "to_agent": {
                    "type": "string",
                    "description": "Target agent ID.",
                },
                "max_sensitivity": {
                    "type": "string",
                    "enum": ["public", "internal", "sensitive"],
                    "description": "Maximum sensitivity level allowed for transfer.",
                    "default": "internal",
                },
            },
            "required": ["memory_id", "from_agent", "to_agent"],
        },
    },
]
