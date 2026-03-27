"""Tests for embedding providers."""

import math

from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.embedding import create_embedding_provider
from ebbingcontext.config import EmbeddingConfig


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class TestLiteEmbeddingProvider:
    def setup_method(self):
        self.provider = LiteEmbeddingProvider(dimension=256)

    def test_dimension(self):
        assert self.provider.dimension == 256

    def test_embed_returns_correct_length(self):
        vec = self.provider.embed("Hello world")
        assert len(vec) == 256

    def test_embed_is_normalized(self):
        vec = self.provider.embed("Hello world")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_identical_texts_high_similarity(self):
        v1 = self.provider.embed("The cat sat on the mat")
        v2 = self.provider.embed("The cat sat on the mat")
        assert cosine_sim(v1, v2) > 0.99

    def test_similar_texts_moderate_similarity(self):
        v1 = self.provider.embed("The cat sat on the mat")
        v2 = self.provider.embed("The cat sat on the rug")
        sim = cosine_sim(v1, v2)
        assert sim > 0.3  # High word overlap should produce decent similarity

    def test_unrelated_texts_lower_similarity(self):
        v1 = self.provider.embed("The cat sat on the mat")
        v2 = self.provider.embed("Quantum mechanics describes particle behavior")
        sim_related = cosine_sim(
            self.provider.embed("The cat sat on the mat"),
            self.provider.embed("A cat was sitting on a mat"),
        )
        sim_unrelated = cosine_sim(v1, v2)
        assert sim_related > sim_unrelated

    def test_empty_text(self):
        vec = self.provider.embed("")
        assert len(vec) == 256
        # All zeros (no tokens to hash)
        assert all(x == 0.0 for x in vec)

    def test_embed_batch_consistency(self):
        texts = ["Hello world", "Goodbye world", "Test text"]
        batch_results = self.provider.embed_batch(texts)
        individual_results = [self.provider.embed(t) for t in texts]
        for batch_vec, ind_vec in zip(batch_results, individual_results):
            assert batch_vec == ind_vec

    def test_custom_dimension(self):
        provider = LiteEmbeddingProvider(dimension=64)
        vec = provider.embed("test")
        assert len(vec) == 64


class TestEmbeddingFactory:
    def test_lite_model(self):
        config = EmbeddingConfig(model="lite", dimension=128)
        provider = create_embedding_provider(config)
        assert provider.dimension == 128
        vec = provider.embed("test")
        assert len(vec) == 128

    def test_unknown_model_falls_back_to_lite(self):
        config = EmbeddingConfig(model="unknown_model", dimension=256)
        provider = create_embedding_provider(config)
        # Should not raise, falls back to lite
        vec = provider.embed("test")
        assert len(vec) == 256

    def test_bge_fallback_to_lite(self):
        """BGE-M3 should fall back to lite if sentence-transformers not installed."""
        config = EmbeddingConfig(model="bge-m3", dimension=1024)
        provider = create_embedding_provider(config)
        # Either BGE (if installed, 1024-dim) or Lite fallback (256-dim)
        vec = provider.embed("test")
        assert len(vec) in (256, 1024)
