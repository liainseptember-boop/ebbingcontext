"""LoCoMo benchmark evaluation framework for EbbingContext.

LoCoMo (Long Conversation Memory) tests multi-hop reasoning
over extended dialogue histories.

Usage:
    python -m benchmarks.locomo_eval --data path/to/locomo.json --top-k 20
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ebbingcontext import MemoryEngine
from ebbingcontext.embedding.lite import LiteEmbeddingProvider


@dataclass
class EvalResult:
    """Single evaluation result."""

    question: str
    expected: str
    predicted: str
    retrieved_count: int
    recall_time_ms: float
    hit: bool


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""

    total_questions: int = 0
    hits: int = 0
    total_recall_ms: float = 0.0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.hits / self.total_questions if self.total_questions > 0 else 0.0

    @property
    def avg_recall_ms(self) -> float:
        return self.total_recall_ms / self.total_questions if self.total_questions > 0 else 0.0

    def summary(self) -> str:
        return (
            f"LoCoMo Evaluation Report\n"
            f"{'='*40}\n"
            f"Questions: {self.total_questions}\n"
            f"Hits:      {self.hits}\n"
            f"Accuracy:  {self.accuracy:.1%}\n"
            f"Avg recall: {self.avg_recall_ms:.1f}ms\n"
        )


def load_locomo_data(path: str) -> list[dict]:
    """Load LoCoMo dataset.

    Expected format: list of objects with:
        - conversation: list of {role, content} turns
        - questions: list of {question, answer} objects
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("LoCoMo data must be a JSON array")
    return data


def evaluate_session(
    engine: MemoryEngine,
    conversation: list[dict],
    questions: list[dict],
    agent_id: str,
    top_k: int = 20,
) -> list[EvalResult]:
    """Evaluate one conversation session."""
    # Ingest conversation
    for turn in conversation:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content:
            source_type = "user" if role == "user" else "agent"
            engine.store(content=content, agent_id=agent_id, source_type=source_type)

    # Evaluate questions
    results: list[EvalResult] = []
    for qa in questions:
        question = qa.get("question", "")
        expected = qa.get("answer", "")

        t0 = time.perf_counter()
        recalled = engine.recall(query=question, agent_id=agent_id, top_k=top_k)
        recall_ms = (time.perf_counter() - t0) * 1000

        # Check if expected answer appears in any recalled content
        retrieved_texts = " ".join(r.item.content for r in recalled)
        hit = expected.lower() in retrieved_texts.lower()

        results.append(EvalResult(
            question=question,
            expected=expected,
            predicted=retrieved_texts[:200],
            retrieved_count=len(recalled),
            recall_time_ms=recall_ms,
            hit=hit,
        ))

    return results


def run_benchmark(data_path: str, top_k: int = 20) -> BenchmarkReport:
    """Run full LoCoMo benchmark."""
    data = load_locomo_data(data_path)
    report = BenchmarkReport()

    for i, session in enumerate(data):
        conversation = session.get("conversation", [])
        questions = session.get("questions", [])

        if not conversation or not questions:
            continue

        # Fresh engine per session
        engine = MemoryEngine(embedding_provider=LiteEmbeddingProvider())
        agent_id = f"session_{i}"

        results = evaluate_session(engine, conversation, questions, agent_id, top_k)

        for r in results:
            report.total_questions += 1
            report.total_recall_ms += r.recall_time_ms
            if r.hit:
                report.hits += 1
            report.results.append(r)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for EbbingContext")
    parser.add_argument("--data", required=True, help="Path to LoCoMo JSON dataset")
    parser.add_argument("--top-k", type=int, default=20, help="Number of memories to recall")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()

    report = run_benchmark(args.data, args.top_k)
    print(report.summary())

    if args.output:
        output = {
            "accuracy": report.accuracy,
            "avg_recall_ms": report.avg_recall_ms,
            "total_questions": report.total_questions,
            "results": [
                {
                    "question": r.question,
                    "expected": r.expected,
                    "hit": r.hit,
                    "recall_time_ms": r.recall_time_ms,
                }
                for r in report.results
            ],
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
