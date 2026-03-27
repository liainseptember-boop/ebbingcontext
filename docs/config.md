# Configuration Reference

EbbingContext uses a layered configuration system: **defaults → YAML file → environment variables**.

## Usage

```python
from ebbingcontext import load_config, MemoryEngine

# Default config
engine = MemoryEngine()

# From YAML
config = load_config("config.yaml")
engine = MemoryEngine.from_config(config)
```

```bash
# CLI
ebbingcontext serve --config config.yaml
```

## Full Configuration

```yaml
# config.yaml
decay:
  s_base: 1.0          # Base stability S₀
  alpha: 0.3            # Access boost coefficient α
  beta_active: 1.2      # Decay exponent for Active layer (fast decay)
  beta_warm: 0.8        # Decay exponent for Warm layer (slow decay)
  rho: 0.1              # Floor retention ρ (minimum strength)

storage:
  theta_active: 0.6     # Strength threshold: Active → Warm demotion
  theta_archive: 0.15   # Strength threshold: Warm → Archive demotion
  persist: false         # Enable persistent storage (JSON + ChromaDB + SQLite)
  active_persist_path: ".ebbingcontext/active.json"   # Active layer JSON file
  archive_db_path: ".ebbingcontext/archive.db"        # Archive SQLite database

pin:
  max_ratio: 0.3        # Maximum ratio of pinned items per agent (30%)

conflict:
  auto_overwrite_threshold: 0.9    # Similarity ≥ 0.9 → overwrite old memory
  association_threshold: 0.7       # Similarity 0.7~0.9 → associate memories

prompt:
  output_reserve: 1024             # Tokens reserved for LLM output
  recent_turns: 3                  # Number of recent conversation turns to include
  warm_retrieval_threshold: 0.5    # If best active similarity < this, also search Warm
  warm_top_k: 10                   # Max items to retrieve from Warm layer

embedding:
  model: "bge-m3"       # Embedding model: "lite", "bge-m3", "openai"
  dimension: 1024        # Embedding dimension (auto-set by provider)

vector_store:
  backend: "chromadb"              # Vector store backend
  persist_dir: ".ebbingcontext/chroma"  # ChromaDB persistence directory

adapter:
  provider: "builtin"   # LLM adapter: "builtin" (rules only) or "openai"
  base_url: null         # OpenAI-compatible API base URL
  model: null            # Model name for LLM classification fallback
  api_key: null          # API key
```

## Environment Variables

Environment variables override YAML values:

| Variable | Maps to | Example |
|----------|---------|---------|
| `EBBINGCONTEXT_EMBEDDING_MODEL` | `embedding.model` | `lite` |
| `EBBINGCONTEXT_EMBEDDING_DIMENSION` | `embedding.dimension` | `256` |
| `EBBINGCONTEXT_VECTOR_BACKEND` | `vector_store.backend` | `chromadb` |
| `EBBINGCONTEXT_PERSIST_DIR` | `vector_store.persist_dir` | `/data/chroma` |
| `EBBINGCONTEXT_ADAPTER_PROVIDER` | `adapter.provider` | `openai` |
| `EBBINGCONTEXT_ADAPTER_BASE_URL` | `adapter.base_url` | `https://api.openai.com/v1` |
| `EBBINGCONTEXT_ADAPTER_MODEL` | `adapter.model` | `gpt-4o-mini` |
| `EBBINGCONTEXT_ADAPTER_API_KEY` | `adapter.api_key` | `sk-...` |

## Embedding Providers

| Model | Dependency | Dimension | Use Case |
|-------|-----------|-----------|----------|
| `lite` | None (built-in) | 256 | Development, testing, zero-dependency |
| `bge-m3` | `pip install "ebbingcontext[bge]"` | 1024 | Production (local, no API calls) |
| `openai` | `pip install "ebbingcontext[openai]"` | 1536 | Production (API-based) |

The system automatically falls back to `lite` if the configured model is unavailable.

## Persistence

When `storage.persist: true`:

| Layer | Backend | Path |
|-------|---------|------|
| Active | JSON file | `storage.active_persist_path` |
| Warm | ChromaDB | `vector_store.persist_dir` |
| Archive | SQLite | `storage.archive_db_path` |

When `storage.persist: false` (default): all layers use in-memory storage. Data is lost on restart.
