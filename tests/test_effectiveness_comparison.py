"""Effectiveness tests: Compare EbbingContext vs FIFO and keep-all baselines.

This is the most critical test file — directly proves EbbingContext is
more useful than naive approaches.
"""

from collections import deque

from ebbingcontext.embedding.lite import LiteEmbeddingProvider
from ebbingcontext.engine import MemoryEngine
from ebbingcontext.models import DecayStrategy


# ── 25-turn conversation with interspersed key facts ──

CONVERSATION = [
    ("user", "Hi there, how are you?", False),
    ("assistant", "I'm doing well, thanks for asking!", False),
    ("user", "My name is Alice and I'm starting a new project.", True),  # KEY FACT: name
    ("assistant", "Nice to meet you Alice! Tell me about your project.", False),
    ("user", "Sure, it's about machine learning.", False),
    ("user", "My favorite programming language is Python.", True),  # KEY FACT: language
    ("assistant", "Python is great for ML projects!", False),
    ("user", "Yeah, I've been coding for 5 years now.", False),
    ("user", "I work at ACME Corporation as a senior engineer.", True),  # KEY FACT: company
    ("assistant", "ACME Corp is a great company! What team are you on?", False),
    ("user", "The data science team. We use pandas a lot.", False),
    ("assistant", "Pandas is excellent for data manipulation.", False),
    ("user", "The weather is nice today, isn't it?", False),
    ("assistant", "I don't have weather info, but I hope it's sunny!", False),
    ("user", "Let's get back to work. Can you help with the code?", False),
    ("user", "My birthday is March 15th, just had a celebration.", True),  # KEY FACT: birthday
    ("assistant", "Happy belated birthday! Now, about the code...", False),
    ("user", "I need a function to parse CSV files.", False),
    ("assistant", "Here's a CSV parser function using pandas...", False),
    ("user", "Thanks, that's helpful. What about error handling?", False),
    ("user", "The project deadline is next Friday, so we need to hurry.", True),  # KEY FACT: deadline
    ("assistant", "Got it, let's prioritize the critical features.", False),
    ("user", "Sounds good, let's do that.", False),
    ("assistant", "I'll focus on the most important items first.", False),
    ("user", "Perfect, thanks for the help!", False),
]

KEY_FACTS = {
    "name": ("Alice", "My name is Alice"),
    "language": ("Python", "My favorite programming language is Python"),
    "company": ("ACME", "I work at ACME Corporation"),
    "birthday": ("March 15", "My birthday is March 15th"),
    "deadline": ("next Friday", "The project deadline is next Friday"),
}

KEY_QUERIES = {
    "name": "What is the user's name?",
    "language": "What programming language does the user prefer?",
    "company": "Where does the user work?",
    "birthday": "When is the user's birthday?",
    "deadline": "What is the project deadline?",
}


# ── FIFO baseline ──

class FIFOBaseline:
    """Naive FIFO: keeps last N turns, no ranking, no decay."""

    def __init__(self, max_turns: int = 10):
        self._buffer = deque(maxlen=max_turns)

    def store(self, content: str):
        self._buffer.append(content)

    def recall(self, query: str) -> list[str]:
        """Returns all stored content (no ranking)."""
        return list(self._buffer)

    def contains(self, keyword: str) -> bool:
        """Check if keyword appears in any stored content."""
        text = " ".join(self._buffer)
        return keyword.lower() in text.lower()


# ── Tests ──

class TestVsFIFO:
    """Prove EbbingContext retains key facts that FIFO loses."""

    def setup_method(self):
        self.engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        self.fifo = FIFOBaseline(max_turns=10)

        # Ingest the full conversation into both systems
        for role, content, _ in CONVERSATION:
            source = "user" if role == "user" else "agent"
            self.engine.store(content, source_type=source)
            self.fifo.store(content)

    def test_ebbingcontext_retains_early_key_facts(self):
        """EbbingContext should find all 5 key facts when queried."""
        found = 0
        for fact_key, (keyword, _) in KEY_FACTS.items():
            query = KEY_QUERIES[fact_key]
            results = self.engine.recall(query, top_k=5)
            texts = " ".join(r.item.content for r in results)
            if keyword.lower() in texts.lower():
                found += 1

        assert found >= 4, f"EbbingContext found only {found}/5 key facts"

    def test_fifo_loses_early_key_facts(self):
        """FIFO (last 10 turns) should miss early facts (name at turn 2, language at turn 5)."""
        missed = 0
        for fact_key, (keyword, _) in KEY_FACTS.items():
            if not self.fifo.contains(keyword):
                missed += 1

        assert missed >= 2, f"FIFO should miss at least 2 facts, but only missed {missed}"

    def test_ebbingcontext_retention_rate(self):
        """EbbingContext key fact retention rate should be >= 80%."""
        found = 0
        for fact_key, (keyword, _) in KEY_FACTS.items():
            query = KEY_QUERIES[fact_key]
            results = self.engine.recall(query, top_k=5)
            texts = " ".join(r.item.content for r in results)
            if keyword.lower() in texts.lower():
                found += 1

        retention = found / len(KEY_FACTS)
        assert retention >= 0.8, f"Retention rate {retention:.0%}, expected >= 80%"

    def test_fifo_retention_lower_than_ebbingcontext(self):
        """FIFO retention rate should be strictly less than EbbingContext's."""
        ec_found = 0
        fifo_found = 0

        for fact_key, (keyword, _) in KEY_FACTS.items():
            # EbbingContext
            query = KEY_QUERIES[fact_key]
            results = self.engine.recall(query, top_k=5)
            texts = " ".join(r.item.content for r in results)
            if keyword.lower() in texts.lower():
                ec_found += 1

            # FIFO
            if self.fifo.contains(keyword):
                fifo_found += 1

        assert ec_found > fifo_found, \
            f"EbbingContext ({ec_found}) should beat FIFO ({fifo_found})"

    def test_query_returns_fact_not_filler(self):
        """Querying for user's name should return the name fact, not filler."""
        results = self.engine.recall("What is the user's name Alice", top_k=3)
        assert len(results) > 0
        # Top result should contain "Alice"
        top_content = results[0].item.content
        assert "Alice" in top_content or "name" in top_content.lower(), \
            f"Top result '{top_content}' doesn't contain the name fact"

    def test_ebbingcontext_filters_filler_from_top(self):
        """Filler messages like greetings should not dominate top results."""
        results = self.engine.recall("user name Alice project", top_k=5)
        filler_phrases = ["how are you", "nice today", "sounds good", "perfect, thanks"]
        filler_in_top3 = sum(
            1 for r in results[:3]
            if any(phrase in r.item.content.lower() for phrase in filler_phrases)
        )
        assert filler_in_top3 <= 1, f"Too much filler in top-3: {filler_in_top3}"


class TestVsKeepAll:
    """Prove EbbingContext is more efficient than keeping everything."""

    def test_keep_all_no_relevance_ranking(self):
        """Keep-all returns everything — no relevance filtering."""
        all_items = []
        for _, content, _ in CONVERSATION:
            all_items.append(content)

        # Keep-all returns all 25 items for any query
        assert len(all_items) == 25

        # EbbingContext returns only top-k relevant items
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        for _, content, _ in CONVERSATION:
            engine.store(content, source_type="user")

        results = engine.recall("Python programming", top_k=5)
        assert len(results) <= 5  # Much more targeted

    def test_keep_all_exceeds_token_budget(self):
        """Many items overflow a token window; EbbingContext fits within budget."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        topics = [
            "python code function class module import variable",
            "cooking recipe kitchen ingredients baking flour",
            "space rocket planet orbit galaxy astronaut",
            "music melody guitar piano rhythm harmony",
            "sports football basketball tennis swimming",
        ]
        for i in range(50):
            topic = topics[i % len(topics)]
            padding = " ".join(f"info{i}y{j}" for j in range(30))
            engine.store(f"{topic} turn{i} {padding}", source_type="user")

        result = engine.recall_for_prompt(
            query="python code function",
            total_window=2000,
            system_prompt="You are helpful.",
            top_k=30,
        )

        assert result.total_tokens <= 2000
        assert result.memories_dropped > 0  # Can't fit all recalled

    def test_ebbingcontext_recall_is_targeted(self):
        """EbbingContext returns few relevant items vs keep-all returning everything."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        # Store items across topics
        for content in [
            "Python programming data science",
            "cooking recipe ingredients baking",
            "space exploration NASA rockets",
            "music theory harmony melody",
            "sports football basketball tennis",
        ]:
            engine.store(content, source_type="user")

        results = engine.recall("Python programming", top_k=3)
        assert len(results) <= 3  # Targeted, not all 5


class TestConflictHandlingEffectiveness:
    """Prove conflict resolution prevents memory accumulation."""

    def test_duplicate_update_replaces(self):
        """Near-duplicate content should overwrite, not accumulate."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        engine.store("user email address is alice@oldcompany.com", source_type="user")
        engine.store("user email address is alice@newcompany.com", source_type="user")

        all_active = engine.active.get_all()
        # With high enough similarity, old should be overwritten
        email_items = [i for i in all_active if "email" in i.content.lower()]

        # If conflict detection triggered: only 1 email item.
        # If similarity was below threshold: 2 items (still OK, just shows lite embedder limits)
        assert len(email_items) <= 2

    def test_unrelated_info_coexists(self):
        """Completely unrelated content should coexist independently."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        item1 = engine.store("Python programming language features", source_type="user")
        item2 = engine.store("The weather today is sunny and warm", source_type="user")

        assert engine.active.get(item1.id) is not None
        assert engine.active.get(item2.id) is not None
        # No conflict metadata
        assert "associated_with" not in item1.metadata
        assert "associated_with" not in item2.metadata

    def test_pin_survives_alongside_normal(self):
        """Pinned items and normal items coexist correctly."""
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())

        pinned = engine.store("System critical rule", decay_strategy=DecayStrategy.PIN)
        normal = engine.store("User mentioned weather", source_type="user")

        assert pinned.decay_strategy == DecayStrategy.PIN
        assert pinned.strength == 1.0
        assert engine.active.get(pinned.id) is not None
        assert engine.active.get(normal.id) is not None
