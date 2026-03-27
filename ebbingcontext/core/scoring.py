"""Importance evaluation and retrieval scoring.

Final retrieval score:
    score_final = similarity(query, memory) × R̃(t)

Where R̃(t) is the effective retention from the decay engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from ebbingcontext.core.decay import DecayEngine
from ebbingcontext.models import MemoryItem


@dataclass
class ScoredMemory:
    """A memory item with its computed retrieval score."""

    item: MemoryItem
    similarity: float
    strength: float
    final_score: float


class ScoringEngine:
    """Handles importance evaluation and retrieval ranking."""

    def __init__(self, decay_engine: DecayEngine) -> None:
        self.decay_engine = decay_engine

    def compute_final_score(self, similarity: float, strength: float) -> float:
        """Compute final retrieval score = similarity × strength."""
        return similarity * strength

    def rank_memories(
        self,
        candidates: list[tuple[MemoryItem, float]],
        current_time: float,
        current_token_pos: int | None = None,
        top_k: int | None = None,
    ) -> list[ScoredMemory]:
        """Rank candidate memories by final score.

        Args:
            candidates: list of (MemoryItem, similarity_score) tuples.
            current_time: current timestamp for decay computation.
            current_token_pos: current token position (intra-session).
            top_k: return only top-k results. None = return all.

        Returns:
            Sorted list of ScoredMemory, highest score first.
        """
        scored: list[ScoredMemory] = []

        for item, similarity in candidates:
            strength = self.decay_engine.compute_strength(
                item, current_time, current_token_pos
            )
            final_score = self.compute_final_score(similarity, strength)
            scored.append(
                ScoredMemory(
                    item=item,
                    similarity=similarity,
                    strength=strength,
                    final_score=final_score,
                )
            )

        scored.sort(key=lambda x: x.final_score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def arrange_by_position(self, scored: list[ScoredMemory]) -> list[ScoredMemory]:
        """Arrange memories for prompt position orchestration.

        Strategy: high-strength at beginning and end, medium in the middle.
        This counters the "Lost in the Middle" phenomenon.
        """
        if len(scored) <= 2:
            return scored

        # Split into high/medium groups by strength
        high = [s for s in scored if s.strength >= 0.7]
        medium = [s for s in scored if s.strength < 0.7]

        if not high:
            return scored
        if not medium:
            return scored

        # High-strength: split between beginning and end
        mid_point = (len(high) + 1) // 2
        head = high[:mid_point]
        tail = high[mid_point:]

        return head + medium + tail
