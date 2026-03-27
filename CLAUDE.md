# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

EbbingContext is an Ebbinghaus Forgetting Curve-based Context Management Engine for LLM Agents. It acts as MCP middleware that automatically classifies, decays, migrates, and assembles memories into token-budgeted prompts following cognitive science principles.

Python 3.11+. Build system: Hatchling.

## Commands

```bash
# Install
pip install -e ".[dev]"          # development (pytest, ruff, coverage)
pip install -e ".[bge]"          # with local BGE-M3 embeddings
pip install -e ".[openai]"       # with OpenAI embeddings

# Test
pytest                           # all tests (asyncio_mode=auto)
pytest tests/test_decay.py       # single test file
pytest tests/test_engine.py -k "test_store"  # single test by name
pytest --cov=ebbingcontext       # with coverage

# Lint
ruff check ebbingcontext/       # lint (target: py311, line-length: 100)
ruff check --fix ebbingcontext/ # autofix

# Run MCP server
ebbingcontext serve [--config config.yaml]

# Run demo
python demo/server.py            # HTTP bridge on localhost:8765
```

## Architecture

### Layered Design

```
LLM/Agent Client
  ↓ (6 MCP Tools: store/recall/pin/forget/inspect/transfer)
MCP Server (interface/mcp_server.py, interface/tools.py)
  ↓
Adapter Layer (adapter/) — LLM integration for classification fallback
  ↓
Core Algorithms (core/) — decay, scoring, classification, chunking, conflict, migration
  ↓
Three-Tier Storage:
  Active (>0.6 strength) → Warm (0.15–0.6) → Archive (≤0.15)
  (in-memory)             (vector search)     (audit trail)
```

### Central Orchestrator

`MemoryEngine` (engine.py) coordinates all subsystems. It is the single entry point for store, recall, pin, forget, inspect, transfer, and migration operations.

### Core Algorithm: Dual-Dimension Decay

Retention follows Ebbinghaus: `R(t) = exp(-(Δt / S)^β)` where stability `S = S₀ × (1 + α × ln(1 + access_count))`. Decay uses token distance intra-session and wall-clock time cross-session. Layer-specific β exponents: Active=1.2 (fast), Warm=0.8 (slow). Floor retention ρ=0.1 ensures no memory reaches zero.

### Dual-Label Classification

Every memory gets two orthogonal labels via rule-based patterns (~60-70% coverage) with LLM fallback:
1. **Decay Strategy**: PIN / DECAY_RECOVERABLE / DECAY_IRREVERSIBLE
2. **Sensitivity Level**: PUBLIC / INTERNAL / SENSITIVE

### Key Design Decisions

- **Separation of concerns**: LLM decides "what to store", system decides "how to store"
- **Rule-first classification**: LLM fallback never propagates errors; defaults to rule result on failure
- **Scoring**: `similarity(query, memory) × effective_retention(t)` — combines semantic relevance with temporal freshness
- **Position orchestration**: High-strength memories split between prompt start/end to counter "Lost in the Middle"
- **Token budget**: System prompt → pinned → tool defs → recent turns → output reserve → remaining for recalled memories (greedy fill)
- **Pin ratio cap**: Max 30% of an agent's memories can be pinned
- **Conflict resolution**: similarity ≥0.9 auto-overwrites, 0.7–0.9 associates, <0.7 treats as unrelated
- **Migration**: Threshold-driven demotion Active→Warm→Archive with full audit trail

### Embedding Providers

Factory pattern (`embedding/__init__.py`) with fallback chain:
- `LiteEmbeddingProvider` — hash-based bag-of-words, zero deps, 256-dim
- `BGEEmbeddingProvider` — sentence-transformers, 1024-dim
- `OpenAIEmbeddingProvider` — OpenAI API, 1536-dim

### Configuration

Layered: defaults → YAML file → environment variables. Central config class `EbbingConfig` in config.py with sub-configs for decay, storage, pinning, conflict, prompt assembly, embedding, vector store, and adapter.
