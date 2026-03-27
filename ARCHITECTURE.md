# EbbingContext Architecture

## System Overview

EbbingContext is an MCP middleware that manages LLM Agent context using principles from the Ebbinghaus Forgetting Curve. It sits between the Agent/LLM and the conversation, automatically classifying, decaying, migrating, and assembling memories into the context window.

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM / Agent Client                         │
│              Operates memory via 6 MCP Tools                    │
├─────────────────────────────────────────────────────────────────┤
│                         MCP Server                              │
│     store / recall / pin / forget / inspect / transfer          │
├─────────────────────────────────────────────────────────────────┤
│                        Adapter Layer                            │
│        Default: built-in classifier | Optional: your LLM       │
├──────────┬────────────┬───────────────┬─────────────────────────┤
│Ingestion │  Scoring   │   Decay &     │  Retrieval &            │
│  Layer   │  Layer     │  Migration    │  Injection Layer        │
│          │            │   Layer       │                         │
│ Parse &  │ Importance │ Exponential   │ Semantic search         │
│ Chunk    │ Dual-label │ Recall boost  │ Strength-weighted rank  │
│ Metadata │ Rule+LLM   │ Threshold     │ Position orchestration  │
│          │            │ Audit log     │ Token budget fill       │
├──────────┴────────────┴───────────────┴─────────────────────────┤
│                        Memory Storage                           │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│  │ Active Layer │→│  Warm Layer   │→│  Archive Layer         │ │
│  │ (in-memory)  │  │ (vector DB)   │  │ (SQLite audit trail)  │ │
│  │ strength>0.6 │  │ 0.15<s≤0.6   │  │ strength≤0.15         │ │
│  └──────────────┘  └───────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Store Flow

```
User/Agent message
       │
       ▼
  MessageChunker ─── split by source_type (sentence/paragraph/JSON field)
       │
       ▼ (per chunk)
  Classifier ─── rule-based (~60-70%) + optional LLM fallback
       │
       ├─ decay_strategy: PIN | DECAY_RECOVERABLE | DECAY_IRREVERSIBLE
       ├─ sensitivity: PUBLIC | INTERNAL | SENSITIVE
       └─ importance: 0.0 ~ 1.0
       │
       ▼
  EmbeddingProvider ─── compute vector (Lite/BGE/OpenAI)
       │
       ▼
  ConflictResolver ─── similarity ≥ 0.9: overwrite | 0.7~0.9: associate | <0.7: new
       │
       ▼
  ActiveStore.add() ─── store in Active layer
       │
       ▼
  AuditRecord ─── log creation event
```

### Recall Flow

```
Query string
       │
       ▼
  EmbeddingProvider ─── compute query vector
       │
       ▼
  Active layer ─── compute similarity for all active items
       │
       ├─ best_sim < warm_threshold (0.5)?
       │     ▼
       │   Warm layer ─── vector search top-K
       │
       ▼
  ScoringEngine.rank_memories()
       │
       ├─ score = similarity × effective_retention
       ├─ R̃(t) = ρ + (1-ρ) × R(t)    (floor guarantee)
       └─ R(t) = exp(-(Δt/S)^β)       (Ebbinghaus formula)
       │
       ▼
  PromptAssembler ─── position orchestration + token budget fill
       │
       ├─ Priority: System > Pin > Tools > Recent turns > Output reserve
       ├─ Remaining budget → recalled memories
       └─ Lost-in-Middle: high-strength at start/end, medium in middle
       │
       ▼
  AssembledPrompt ─── ready for LLM context injection
```

### Migration Flow

```
  Periodic / on-demand
       │
       ▼
  DecayEngine.batch_update() ─── update all strengths
       │
       ▼
  MigrationEngine.evaluate_batch()
       │
       ├─ strength > θ_active (0.6)  → stays in Active
       ├─ θ_archive < s ≤ θ_active   → demote to Warm
       └─ strength ≤ θ_archive (0.15) → demote to Archive
       │
       ▼
  AuditRecord ─── log migration event
```

## Core Formulas

### Ebbinghaus Decay

```
R(t) = exp( -(Δt / S)^β )

Where:
  Δt  = time elapsed since last access (seconds)
  S   = stability = S₀ × (1 + α × ln(1 + n))
  S₀  = base stability (default: 1.0)
  α   = access boost coefficient (default: 0.3)
  n   = access count
  β   = layer-specific exponent
        Active: β = 1.2 (fast decay → keeps only relevant items)
        Warm:   β = 0.8 (slow decay → preserves moderately important items)
```

### Effective Retention (Floor Guarantee)

```
R̃(t) = ρ + (1 - ρ) × R(t)

Where:
  ρ = floor retention (default: 0.1)
  Ensures no memory ever drops to absolute zero strength.
```

### Final Score (Recall Ranking)

```
score_final = sim(query, memory) × R̃(t)

Where:
  sim = cosine similarity between query and memory embeddings
  R̃(t) = effective retention (see above)
```

### Stability Growth

Each time a memory is accessed, its stability grows logarithmically:

```
S(n) = S₀ × (1 + α × ln(1 + n))
```

This models the "spacing effect": repeated recalls strengthen a memory, but with diminishing returns.

## Dual-Label Classification

Every memory receives two independent labels:

| Label | Values | Determines |
|-------|--------|------------|
| **Decay Strategy** | `PIN`, `DECAY_RECOVERABLE`, `DECAY_IRREVERSIBLE` | How the memory decays over time |
| **Sensitivity Level** | `PUBLIC`, `INTERNAL`, `SENSITIVE` | Who can access via cross-agent transfer |

Classification uses a two-stage pipeline:
1. **Rule-based** (regex patterns) — handles ~60-70% of cases with high confidence
2. **LLM fallback** (optional) — for low-confidence rule results, queries the configured LLM

## Storage Layers

| Layer | Threshold | Backend (in-memory) | Backend (persistent) | Purpose |
|-------|-----------|--------------------|--------------------|---------|
| Active | strength > 0.6 | dict | JSON file | In-context, participates in prompt assembly |
| Warm | 0.15 < s ≤ 0.6 | dict + numpy | ChromaDB | Vector search pool for recall |
| Archive | s ≤ 0.15 | dict | SQLite | Audit trail, never deleted |

## Conflict Resolution

When storing a new memory with an embedding, the system checks for semantic conflicts:

| Similarity | Action | Rationale |
|------------|--------|-----------|
| ≥ 0.9 | **Overwrite** — archive old, store new | Near-duplicate, likely an update |
| 0.7 ~ 0.9 | **Associate** — link via metadata | Related but distinct, keep both |
| < 0.7 | **New** — store independently | Unrelated content |

## Pin System

- Pinned memories have strength = 1.0, never decay
- Always included in prompt assembly (non-negotiable allocation)
- **30% ratio cap**: at most 30% of an agent's active memories can be pinned
- LLM can explicitly pin/unpin; system auto-classifies `source_type=system` as pinned

## Prompt Assembly (Token Budget)

```
Available budget = Total window
                 - System prompt tokens
                 - Pinned memory tokens
                 - Tool definition tokens
                 - Recent N turns tokens
                 - Output reserve tokens

Remaining budget → filled greedily with recalled memories
                   (position-orchestrated, highest score first)
```

**Position Orchestration** (Lost-in-Middle mitigation):
- High-strength memories (≥ 0.7) split between start and end
- Medium-strength memories placed in the middle
- This counteracts the known attention pattern where LLMs underweight mid-context content

## Extension Points

| Component | Interface | How to Extend |
|-----------|-----------|---------------|
| Embedding | `EmbeddingProvider` (ABC) | Implement `embed()`, `embed_batch()`, `dimension` |
| LLM Adapter | `LLMAdapter` (ABC) | Implement `complete()` for classification fallback |
| Storage | Drop-in replacement | Match the `add/get/remove/get_all` interface |
| Classifier | `Classifier.classify_with_fallback()` | Provide `LLMAdapter` for LLM-enhanced classification |

## MCP Protocol

EbbingContext exposes 6 tools via the Model Context Protocol:

| Tool | Input | Output | Side Effects |
|------|-------|--------|-------------|
| `store_memory` | content, source_type, agent_id | list[MemoryItem] | Chunks → classifies → embeds → stores |
| `recall_memory` | query, agent_id, top_k | list[ScoredMemory] | Updates decay, touches retrieved items |
| `pin_memory` | memory_id | MemoryItem | Sets strength=1.0, decay=PIN |
| `forget_memory` | memory_id | MemoryItem | Moves to Archive with audit |
| `inspect_memory` | memory_id | dict (state + audit) | Read-only |
| `transfer_memory` | memory_id, from_agent, to_agent | MemoryItem or None | Sensitivity-filtered copy |
