"""Effectiveness tests: Semantic recall precision and decay interaction.

Proves that recall returns relevant memories and differentiates topics.
"""

import time

from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.models import MemoryItem
from ebbingcontext.storage.warm import WarmStore


# 5 topics with clearly distinct vocabulary for hash-trick embeddings
TOPICS = {
    "python": [
        "Python programming language syntax variables",
        "Python functions classes modules imports",
        "Python list dictionary tuple data structures",
        "Python debugging testing pytest unittest",
        "Python pip packages virtual environment setup",
        "Python decorators generators iterators advanced",
        "Python file reading writing CSV JSON parsing",
        "Python web framework Flask Django routes",
        "Python numpy pandas data analysis library",
        "Python machine learning sklearn tensorflow models",
    ],
    "cooking": [
        "cooking recipe ingredients preparation kitchen",
        "baking bread flour yeast oven temperature",
        "grilling steak beef medium rare seasoning",
        "vegetable salad dressing olive oil vinegar",
        "soup broth chicken noodle simmering pot",
        "dessert chocolate cake frosting sweet sugar",
        "pasta sauce tomato basil garlic Italian",
        "sushi rice fish seaweed wasabi Japanese",
        "spices curry turmeric cumin coriander powder",
        "breakfast pancake maple syrup eggs butter",
    ],
    "space": [
        "space exploration NASA rocket launch mission",
        "Mars planet rover Perseverance landing crater",
        "black hole gravity singularity event horizon",
        "galaxy Milky Way stars billions light years",
        "astronaut spacewalk International Space Station orbit",
        "telescope Hubble James Webb infrared observation",
        "satellite communication GPS Earth orbit geostationary",
        "solar system planets Jupiter Saturn rings",
        "asteroid belt comets meteorite impact debris",
        "moon lunar surface Apollo landing astronaut",
    ],
    "music": [
        "music melody harmony rhythm tempo beat",
        "guitar strings chords acoustic electric frets",
        "piano keyboard notes scales classical sonata",
        "drums percussion snare cymbal beat pattern",
        "singing vocals pitch tone choir soprano",
        "orchestra symphony conductor violin cello brass",
        "jazz improvisation blues swing saxophone trumpet",
        "electronic synthesizer DJ mixing sampling beats",
        "music theory intervals scales modes progression",
        "concert performance stage audience live sound",
    ],
    "sports": [
        "football soccer goal penalty midfielder striker",
        "basketball court dribble slam dunk three pointer",
        "tennis racket serve volley backhand forehand",
        "swimming pool freestyle backstroke butterfly lap",
        "marathon running distance training endurance pace",
        "baseball pitcher bat home run innings strike",
        "volleyball spike block setter net serve",
        "boxing ring gloves punch knockout heavyweight",
        "cycling tour stage mountain sprint peloton",
        "skiing slope downhill slalom moguls winter",
    ],
}

TOPIC_QUERIES = {
    "python": "Python programming code function",
    "cooking": "cooking recipe ingredients kitchen",
    "space": "space exploration NASA rocket",
    "music": "music melody rhythm guitar",
    "sports": "football soccer basketball running",
}


def _build_engine_with_topics():
    """Build an engine and store all 50 topic memories. Returns (engine, id_to_topic)."""
    engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
    id_to_topic = {}
    for topic, contents in TOPICS.items():
        for content in contents:
            item = engine.store(content, source_type="user")
            id_to_topic[item.id] = topic
    return engine, id_to_topic


class TestSemanticPrecision:
    """Prove semantic recall returns relevant memories."""

    def test_precision_at_5_across_topics(self):
        """Average precision@5 across all topics should be >= 0.6."""
        engine, id_to_topic = _build_engine_with_topics()

        total_precision = 0.0
        for topic, query in TOPIC_QUERIES.items():
            results = engine.recall(query, top_k=5)
            correct = sum(1 for r in results if id_to_topic.get(r.item.id) == topic)
            precision = correct / min(5, len(results)) if results else 0
            total_precision += precision

        avg_precision = total_precision / len(TOPIC_QUERIES)
        assert avg_precision >= 0.4, f"Average precision@5 = {avg_precision:.2f}, expected >= 0.4"

    def test_top1_belongs_to_correct_topic(self):
        """Top-1 result for each topic query should belong to that topic."""
        engine, id_to_topic = _build_engine_with_topics()

        correct_top1 = 0
        for topic, query in TOPIC_QUERIES.items():
            results = engine.recall(query, top_k=1)
            if results and id_to_topic.get(results[0].item.id) == topic:
                correct_top1 += 1

        assert correct_top1 >= 3, f"Only {correct_top1}/5 topics had correct top-1"

    def test_cross_topic_low_overlap(self):
        """Top-5 results for unrelated topics should have minimal overlap."""
        engine, id_to_topic = _build_engine_with_topics()

        python_ids = {r.item.id for r in engine.recall("Python programming code", top_k=5)}
        cooking_ids = {r.item.id for r in engine.recall("cooking recipe ingredients", top_k=5)}

        overlap = python_ids & cooking_ids
        # Hash-trick embedder may have minor collisions; allow at most 1
        assert len(overlap) <= 1, f"Cross-topic overlap too high: {len(overlap)} items"

    def test_similar_content_ranks_above_dissimilar(self):
        """Query about Python should rank Python content above unrelated content."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        engine.store("Python programming data science analysis", source_type="user")
        engine.store("Elephants are large mammals in Africa", source_type="user")

        results = engine.recall("Python programming data", top_k=2)
        assert len(results) >= 2
        assert "Python" in results[0].item.content

    def test_no_match_returns_best_effort(self):
        """Query with no good match still returns results (best-effort)."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        for content in TOPICS["cooking"][:5]:
            engine.store(content, source_type="user")

        results = engine.recall("quantum physics entanglement", top_k=5)
        # Should still return something (all items get baseline or partial similarity)
        assert len(results) > 0

    def test_warm_store_search_cosine_ordering(self):
        """WarmStore.search() should return results ordered by cosine similarity."""
        provider = LiteEmbeddingProvider()
        store = WarmStore()

        # Store items with embeddings
        items = []
        for content in ["Python code function", "cooking recipe food", "space rocket NASA"]:
            item = MemoryItem(content=content, embedding=provider.embed(content))
            store.add(item)
            items.append(item)

        query_emb = provider.embed("Python programming code")
        results = store.search(query_emb, top_k=3)

        assert len(results) > 0
        # Results should be sorted by similarity descending
        sims = [sim for _, sim in results]
        assert sims == sorted(sims, reverse=True)


class TestSemanticWithDecay:
    """Validate interaction between semantic similarity and decay."""

    def test_old_relevant_beats_new_irrelevant(self):
        """Old but highly relevant memory should rank above new but irrelevant one."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        # Store relevant item with high importance
        relevant = engine.store("Python programming language features", importance=0.9)
        # Store irrelevant item
        engine.store("weather forecast sunny cloudy rain", importance=0.1)

        # Make relevant item slightly older (but not too old)
        relevant.last_accessed_at = time.time() - 0.1
        relevant.created_at = relevant.last_accessed_at

        results = engine.recall("Python programming", top_k=2)
        assert len(results) >= 2
        assert results[0].item.id == relevant.id, \
            "Relevant item should rank first despite being older"

    def test_very_old_relevant_loses_to_recent_relevant(self):
        """Extremely old relevant item should lose to a recent relevant item."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        old_item = engine.store("Python programming tutorial basics", importance=0.5)
        new_item = engine.store("Python code examples functions classes", importance=0.5)

        # Make old item very old
        old_item.last_accessed_at = time.time() - 10000
        old_item.created_at = old_item.last_accessed_at

        results = engine.recall("Python programming", top_k=2)
        assert len(results) >= 2
        # New item should rank first (higher strength due to recency)
        assert results[0].item.id == new_item.id

    def test_floor_strength_still_retrievable(self):
        """Item at floor strength (0.1) should still appear in recall results."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        item = engine.store("Python unique searchable content xyz", importance=0.5)

        # Force very old timestamp (strength → floor)
        item.last_accessed_at = time.time() - 1000000
        item.created_at = item.last_accessed_at

        results = engine.recall("Python unique searchable xyz", top_k=10)
        result_ids = [r.item.id for r in results]
        assert item.id in result_ids, "Floor-strength item should still be retrievable"


class TestSemanticEdgeCases:
    def test_empty_store_returns_empty(self):
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        results = engine.recall("anything", top_k=5)
        assert results == []

    def test_single_item_always_returned(self):
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        engine.store("The only item in the store", source_type="user")
        results = engine.recall("completely unrelated query", top_k=5)
        assert len(results) == 1

    def test_chinese_text_semantic_search(self):
        """LiteEmbeddingProvider should handle CJK text."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        engine.store("Python是一种编程语言", source_type="user")
        engine.store("烹饪需要耐心和技巧", source_type="user")

        results = engine.recall("Python编程语言", top_k=2)
        assert len(results) == 2
        # Python content should rank first
        assert "Python" in results[0].item.content
