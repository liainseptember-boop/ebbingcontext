# EbbingContext

[![CI](https://github.com/liainseptember-boop/ebbingcontext/actions/workflows/ci.yml/badge.svg)](https://github.com/liainseptember-boop/ebbingcontext/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Coverage: 83%](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)]()

**Ebbinghaus Forgetting Curve-Based Context Management Engine for LLM Agents**

> Let AI Agents manage memory like the human brain: retain what matters, gracefully forget what doesn't.

[中文文档](./README_CN.md) | [Architecture Design](./ARCHITECTURE.md) | [Live Demo →](#demo)

---

## The Problem

Context management in LLM Agents faces three fundamental challenges:

| Challenge | Symptom | Consequence |
|-----------|---------|-------------|
| **Limited context window** | Even 128K tokens gets exhausted in agentic loops | Critical information truncated |
| **Uneven attention** | "Lost in the Middle" — models ignore mid-context info | Important data overlooked |
| **Wasteful retention** | Tool returns 40+ fields, only 5 relevant | Noise drowns out key context |

Existing solutions — manual tool output trimming, progressive summarization, explicit prompt engineering — all require developers to make these decisions by hand, for every Agent project.

## The Solution

EbbingContext engineers the cognitive science behind the Ebbinghaus Forgetting Curve into an adaptive context management engine:

```
Human Memory Principle              EbbingContext Mapping
──────────────────────              ─────────────────────
Rapid initial forgetting    →       New information decays fast initially
Spaced review strengthens   →       Referenced memories auto-strengthen
Meaningful material lasts   →       High-importance info decays slower
Forgetting frees cognition  →       Low-strength info exits context window
Short-term → Long-term      →       Frequent + important info persists to storage
```

## Key Features

- **Three decay strategies** — Pin / Decay-Recoverable / Decay-Irreversible, auto-classified by loss severity
- **Dual-dimension decay** — Token distance within sessions, physical time across sessions
- **Three-tier storage** — Active / Warm / Archive, strength-threshold-driven auto-migration
- **Audit trail** — Archive preserves full decay paths, reference chains, and context snapshots
- **Position orchestration** — High-strength memories placed at context start/end to counter "Lost in the Middle"
- **Security controls** — Dual-label classification (decay strategy + sensitivity level), permission-filtered cross-agent transfer
- **Model-agnostic** — Built-in lightweight classifier for out-of-box use, switchable to developer's own model

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     LLM / Agent Client                       │
│            Operates memory via 6 MCP Tools                   │
├──────────────────────────────────────────────────────────────┤
│                       MCP Server                             │
│   store / recall / pin / forget / inspect / transfer         │
├──────────────────────────────────────────────────────────────┤
│                      Adapter Layer                           │
│      Default: built-in classifier | Optional: your LLM      │
├──────────┬───────────┬──────────────┬────────────────────────┤
│Ingestion │  Scoring  │   Decay &    │  Retrieval &           │
│  Layer   │  Layer    │  Migration   │  Injection Layer       │
│          │           │   Layer      │                        │
│ Parse &  │ Importance│ Exponential  │ Semantic search        │
│ Chunk    │ Dual-label│ Recall boost │ Strength-weighted rank │
│ Metadata │ Rule+LLM │ Threshold    │ Position orchestration │
│          │           │ Audit log    │ Token budget fill      │
├──────────┴───────────┴──────────────┴────────────────────────┤
│                      Memory Storage                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Active Layer │→│  Warm Layer  │→│  Archive Layer      │ │
│  │ (context win)│  │ (vector DB)  │  │ (audit trail/disk) │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## How It Compares

| Dimension | FIFO | RAG | Mem0 | MemGPT | **EbbingContext** |
|-----------|------|-----|------|--------|-------------------|
| **Forgetting** | None (truncate) | None | Simple TTL | None | Adaptive exponential decay |
| **Importance** | ❌ | ❌ | ✅ LLM score | ❌ | ✅ Importance × frequency × time |
| **Recall boost** | ❌ | ❌ | ⚠️ Limited | ❌ | ✅ Auto-strengthen on retrieval |
| **Storage tiers** | Single | Single | Single | Dual | Three + audit trail |
| **Security** | ❌ | ❌ | ❌ | ❌ | ✅ Sensitivity labels + permissions |
| **Position aware** | ❌ | ❌ | ❌ | ❌ | ✅ Counters Lost in the Middle |

## MCP Tools

| Tool | Function |
|------|----------|
| `store_memory` | Store info (LLM decides *whether*, system decides *how*) |
| `recall_memory` | Retrieve relevant memories (ranked by similarity × strength) |
| `pin_memory` | Mark as unforgettable (LLM override > system auto-classification) |
| `forget_memory` | Explicitly remove (moved to Archive, not deleted) |
| `inspect_memory` | View memory state (strength, class, decay path) |
| `transfer_memory` | Transfer between Agents (filtered by sensitivity + permissions) |

## Quick Start

### Install

```bash
# Core (built-in lite embedding, works out of box)
pip install ebbingcontext

# With local BGE-M3 embedding model (recommended for production)
pip install ebbingcontext[bge]

# With OpenAI embedding + LLM fallback
pip install ebbingcontext[openai]

# All optional dependencies
pip install ebbingcontext[bge,openai]
```

### Python API

```python
from ebbingcontext import MemoryEngine

engine = MemoryEngine()

# Store — system auto-classifies decay strategy, sensitivity, importance
engine.store("User prefers concise code style", importance=0.9)
engine.store("API key: sk-xxx", source_type="tool")  # auto-classified as SENSITIVE

# Recall — ranked by similarity × memory strength
results = engine.recall("code style", top_k=5)
for scored in results:
    print(f"{scored.item.content} (score: {scored.final_score:.2f})")

# Recall for prompt — token-budgeted assembly
prompt = engine.recall_for_prompt(
    query="code style",
    total_window=128000,
    system_prompt="You are a helpful assistant.",
)
print(f"Using {prompt.total_tokens} tokens, {prompt.memories_included} memories")

# Pin / Forget / Inspect
engine.pin(results[0].item.id)
info = engine.inspect(results[0].item.id)
engine.forget(results[0].item.id)
```

### MCP Server

```bash
# Start with built-in classifier (works out of box)
ebbingcontext serve

# With custom config
ebbingcontext serve --config config.yaml
```

### With Persistence

```python
from ebbingcontext import MemoryEngine, EbbingConfig, load_config

# Enable persistent storage (JSON + ChromaDB + SQLite)
config = load_config("config.yaml")  # set storage.persist: true
engine = MemoryEngine.from_config(config)

# Data survives restart
engine.store("This memory will persist across sessions")
```

## Demo

Try the interactive demo to see how memories decay in real-time, strengthen when recalled, and migrate across storage tiers:

**[→ Live Demo](./demo/)** (coming soon)

## Benchmarks

Evaluation on standard benchmarks (in progress):

| Benchmark | Metric | Target |
|-----------|--------|--------|
| LoCoMo | Multi-hop F1 | ≥ 29.0 |
| MSC | RP@10 | ≥ 75.0 |
| LTI-Bench | Critical fact retention @ 55% storage | ≥ 80% |

## Documentation

- [Architecture Design → ARCHITECTURE.md](./ARCHITECTURE.md)
- [Configuration Reference](./docs/config.md)
- [API Documentation](./docs/api.md)

## References

- Ebbinghaus, H. (1885). *Über das Gedächtnis*
- Zhong et al. (2024). *MemoryBank: Enhancing LLMs with Long-Term Memory* (AAAI 2024)
- Wei et al. (2026). *FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory*
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*

## License

MIT License
