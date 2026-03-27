# API Reference

## Quick Start

```python
from ebbingcontext import MemoryEngine, MemoryItem, DecayStrategy

# Create engine (in-memory, lite embeddings)
engine = MemoryEngine()

# Or from config file
from ebbingcontext import load_config
config = load_config("config.yaml")
engine = MemoryEngine.from_config(config)
```

## MemoryEngine

The central orchestrator. Coordinates classification, decay, storage, migration, and conflict resolution.

### Constructor

```python
MemoryEngine(
    s_base: float = 1.0,
    alpha: float = 0.3,
    beta_active: float = 1.2,
    beta_warm: float = 0.8,
    rho: float = 0.1,
    theta_active: float = 0.6,
    theta_archive: float = 0.15,
    auto_overwrite_threshold: float = 0.9,
    association_threshold: float = 0.7,
    pin_max_ratio: float = 0.3,
    embedding_provider: EmbeddingProvider | None = None,
    warm_retrieval_threshold: float = 0.5,
)
```

### `store(content, ...) -> MemoryItem`

Store a single memory item. The system auto-classifies decay strategy, sensitivity, and importance.

```python
item = engine.store(
    content="User prefers Python for data science",
    agent_id="default",          # Agent namespace
    source_type="user",          # "user" | "agent" | "tool"
    importance=None,             # Override auto-classification (0.0~1.0)
    decay_strategy=None,         # Override: PIN | DECAY_RECOVERABLE | DECAY_IRREVERSIBLE
    sensitivity=None,            # Override: PUBLIC | INTERNAL | SENSITIVE
    embedding=None,              # Pre-computed embedding (auto-computed if provider available)
    metadata=None,               # Additional metadata dict
)
```

### `store_message(content, ...) -> list[MemoryItem]`

Parse, chunk, and store a full message. Automatically splits by source type:
- **user** messages → split by sentence
- **agent** messages → split by paragraph
- **tool** messages → split by JSON fields (fallback: by line)

```python
items = engine.store_message(
    content="The API returned: {\"status\": \"ok\", \"data\": [1,2,3]}",
    source_type="tool",
)
# Returns multiple MemoryItem objects, one per chunk
```

### `recall(query, ...) -> list[ScoredMemory]`

Retrieve relevant memories ranked by `similarity × effective_retention`.

```python
results = engine.recall(
    query="Python data science",     # Text query (auto-embedded)
    query_embedding=None,            # Or provide pre-computed embedding
    agent_id="default",
    top_k=10,
    current_token_pos=None,          # For intra-session token distance decay
)

for scored in results:
    print(scored.item.content, scored.final_score)
```

### `recall_for_prompt(...) -> AssembledPrompt`

Recall memories and assemble into a token-budgeted prompt.

```python
result = engine.recall_for_prompt(
    query="Python",
    agent_id="default",
    total_window=128000,
    system_prompt="You are a helpful assistant.",
    tool_definitions=[...],
    recent_messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ],
    top_k=20,
)

print(result.total_tokens)        # Total tokens used
print(result.budget_remaining)    # Tokens still available
print(result.memories_included)   # Number of memories in prompt
print(result.memories_dropped)    # Number that didn't fit

for section in result.sections:
    print(section.role, section.token_count)
```

### `pin(memory_id) -> MemoryItem`

Mark a memory as unforgettable (strength = 1.0, no decay).

```python
engine.pin(item.id)
```

### `forget(memory_id) -> MemoryItem`

Explicitly forget a memory. Moves to Archive (not deleted).

```python
engine.forget(item.id)
```

### `inspect(memory_id) -> dict`

View a memory's current state including computed strength and audit trail.

```python
info = engine.inspect(item.id)
print(info["current_strength"])
print(info["stability"])
print(info["audit_trail"])
```

### `transfer(memory_id, from_agent, to_agent, ...) -> MemoryItem | None`

Transfer a memory between agents with sensitivity filtering.

```python
transferred = engine.transfer(
    memory_id=item.id,
    from_agent="agent_a",
    to_agent="agent_b",
    max_sensitivity=SensitivityLevel.INTERNAL,  # Block SENSITIVE items
)
# Returns None if blocked by sensitivity filter
```

### `run_migration(agent_id) -> int`

Run a migration cycle: update strengths and move items between layers.

```python
migrated_count = engine.run_migration(agent_id="default")
```

## MemoryItem

Pydantic model representing a single memory.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | str | auto UUID | Unique identifier |
| `content` | str | required | Memory content |
| `agent_id` | str | "default" | Agent namespace |
| `decay_strategy` | DecayStrategy | DECAY_RECOVERABLE | PIN / DECAY_RECOVERABLE / DECAY_IRREVERSIBLE |
| `sensitivity` | SensitivityLevel | INTERNAL | PUBLIC / INTERNAL / SENSITIVE |
| `importance` | float | 0.5 | Importance score (0.0~1.0) |
| `strength` | float | 1.0 | Current retention strength |
| `access_count` | int | 0 | Number of times recalled |
| `created_at` | float | time.time() | Creation timestamp |
| `last_accessed_at` | float | time.time() | Last access timestamp |
| `layer` | StorageLayer | ACTIVE | ACTIVE / WARM / ARCHIVE |
| `embedding` | list[float] | None | Vector embedding |
| `source_type` | str | "user" | user / agent / tool |
| `metadata` | dict | {} | Additional metadata |

## EmbeddingProvider

Abstract base class for embedding providers.

```python
from ebbingcontext.embedding.base import EmbeddingProvider

class MyProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        # Your embedding logic
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
```

### Built-in Providers

```python
from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.embedding.bge import BGEEmbeddingProvider        # requires sentence-transformers
from ebbingcontext.embedding.openai_embed import OpenAIEmbeddingProvider  # requires openai
```

## Enums

```python
from ebbingcontext import DecayStrategy, SensitivityLevel, StorageLayer

# Decay strategies
DecayStrategy.PIN                  # Never decays, strength = 1.0
DecayStrategy.DECAY_RECOVERABLE   # Decays but access boosts strength
DecayStrategy.DECAY_IRREVERSIBLE  # Monotonic decay, no recovery

# Sensitivity levels
SensitivityLevel.PUBLIC            # Freely transferable
SensitivityLevel.INTERNAL          # Transfer within org
SensitivityLevel.SENSITIVE         # No transfer

# Storage layers
StorageLayer.ACTIVE                # In-context (strength > 0.6)
StorageLayer.WARM                  # Vector search pool (0.15 < s ≤ 0.6)
StorageLayer.ARCHIVE               # Audit trail (s ≤ 0.15)
```
