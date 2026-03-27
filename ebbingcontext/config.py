"""Configuration loading for EbbingContext.

Supports: default values, YAML file override, environment variable override.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class DecayConfig(BaseModel):
    s_base: float = 1.0
    alpha: float = 0.3
    beta_active: float = 1.2
    beta_warm: float = 0.8
    rho: float = 0.1


class StorageConfig(BaseModel):
    theta_active: float = 0.6
    theta_archive: float = 0.15
    persist: bool = False
    active_persist_path: str = ".ebbingcontext/active.json"
    archive_db_path: str = ".ebbingcontext/archive.db"


class PinConfig(BaseModel):
    max_ratio: float = 0.3


class ConflictConfig(BaseModel):
    auto_overwrite_threshold: float = 0.9
    association_threshold: float = 0.7


class PromptConfig(BaseModel):
    output_reserve: int = 1024
    recent_turns: int = 3
    warm_retrieval_threshold: float = 0.5
    warm_top_k: int = 10


class EmbeddingConfig(BaseModel):
    model: str = "bge-m3"
    dimension: int = 1024


class VectorStoreConfig(BaseModel):
    backend: str = "chromadb"
    persist_dir: str = ".ebbingcontext/chroma"


class AdapterConfig(BaseModel):
    provider: str = "builtin"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class EbbingConfig(BaseModel):
    decay: DecayConfig = DecayConfig()
    storage: StorageConfig = StorageConfig()
    pin: PinConfig = PinConfig()
    conflict: ConflictConfig = ConflictConfig()
    prompt: PromptConfig = PromptConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    adapter: AdapterConfig = AdapterConfig()


def load_config(path: str | None = None) -> EbbingConfig:
    """Load configuration from YAML file with environment variable overrides.

    Priority: env vars > YAML file > defaults.
    """
    data: dict = {}

    if path is not None:
        import yaml

        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    data = loaded

    # Environment variable overrides
    env_map = {
        "EBBINGCONTEXT_EMBEDDING_MODEL": ("embedding", "model"),
        "EBBINGCONTEXT_EMBEDDING_DIMENSION": ("embedding", "dimension"),
        "EBBINGCONTEXT_VECTOR_BACKEND": ("vector_store", "backend"),
        "EBBINGCONTEXT_PERSIST_DIR": ("vector_store", "persist_dir"),
        "EBBINGCONTEXT_ADAPTER_PROVIDER": ("adapter", "provider"),
        "EBBINGCONTEXT_ADAPTER_BASE_URL": ("adapter", "base_url"),
        "EBBINGCONTEXT_ADAPTER_MODEL": ("adapter", "model"),
        "EBBINGCONTEXT_ADAPTER_API_KEY": ("adapter", "api_key"),
    }

    for env_var, (section, key) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            data.setdefault(section, {})
            # Try to convert numeric values
            if key == "dimension":
                value = int(value)
            data[section][key] = value

    return EbbingConfig(**data)
