"""Tests for configuration loading."""

import os
import tempfile

from ebbingcontext.config import EbbingConfig, load_config
from ebbingcontext.engine import MemoryEngine


class TestEbbingConfig:
    def test_defaults(self):
        config = EbbingConfig()
        assert config.decay.s_base == 1.0
        assert config.decay.alpha == 0.3
        assert config.storage.theta_active == 0.6
        assert config.pin.max_ratio == 0.3
        assert config.embedding.model == "bge-m3"
        assert config.embedding.dimension == 1024
        assert config.vector_store.backend == "chromadb"

    def test_load_config_no_file(self):
        config = load_config(None)
        assert config.decay.s_base == 1.0

    def test_load_config_from_yaml(self):
        yaml_content = """
decay:
  s_base: 2.0
  alpha: 0.5
storage:
  theta_active: 0.7
embedding:
  model: lite
  dimension: 128
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        assert config.decay.s_base == 2.0
        assert config.decay.alpha == 0.5
        assert config.storage.theta_active == 0.7
        assert config.embedding.model == "lite"
        assert config.embedding.dimension == 128
        # Unchanged defaults
        assert config.decay.beta_active == 1.2

    def test_load_config_nonexistent_file(self):
        config = load_config("/nonexistent/path.yaml")
        # Falls back to defaults
        assert config.decay.s_base == 1.0

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("EBBINGCONTEXT_EMBEDDING_MODEL", "lite")
        monkeypatch.setenv("EBBINGCONTEXT_EMBEDDING_DIMENSION", "512")
        config = load_config(None)
        assert config.embedding.model == "lite"
        assert config.embedding.dimension == 512

    def test_env_overrides_yaml(self, monkeypatch):
        yaml_content = """
embedding:
  model: bge-m3
  dimension: 1024
"""
        monkeypatch.setenv("EBBINGCONTEXT_EMBEDDING_MODEL", "lite")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        # Env var wins
        assert config.embedding.model == "lite"


class TestFromConfig:
    def test_engine_from_config(self):
        config = EbbingConfig(
            embedding={"model": "lite", "dimension": 128},
            decay={"s_base": 2.0},
        )
        engine = MemoryEngine.from_config(config)
        assert engine.decay_engine.s_base == 2.0
        assert engine.embedding_provider is not None
        assert engine.embedding_provider.dimension == 128

    def test_engine_from_default_config(self):
        """from_config with defaults should produce a working engine (lite fallback)."""
        config = EbbingConfig(embedding={"model": "lite"})
        engine = MemoryEngine.from_config(config)
        assert engine.embedding_provider is not None
