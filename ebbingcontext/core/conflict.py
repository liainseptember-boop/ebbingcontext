"""Conflict detection and resolution.

Thresholds:
    - similarity >= 0.9: auto-overwrite, old → Archive
    - association_threshold <= similarity < 0.9: mark as associated, return together
    - similarity < association_threshold: treat as unrelated
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ebbingcontext.models import MemoryItem


class ConflictType(str, Enum):
    """Type of conflict between two memory items."""

    OVERWRITE = "overwrite"  # Near-duplicate, auto-replace
    ASSOCIATE = "associate"  # Related, link together
    UNRELATED = "unrelated"  # No conflict


@dataclass
class ConflictResult:
    """Result of conflict detection between a new item and existing items."""

    conflict_type: ConflictType
    existing_item: MemoryItem | None = None
    similarity: float = 0.0


class ConflictResolver:
    """Detects and resolves conflicts between memory items."""

    def __init__(
        self,
        auto_overwrite_threshold: float = 0.9,
        association_threshold: float = 0.7,
    ) -> None:
        self.auto_overwrite_threshold = auto_overwrite_threshold
        self.association_threshold = association_threshold

    def detect(
        self,
        new_item: MemoryItem,
        candidates: list[tuple[MemoryItem, float]],
    ) -> ConflictResult:
        """Detect conflict between a new item and existing candidates.

        Args:
            new_item: the incoming memory item.
            candidates: list of (existing_item, similarity) tuples,
                        sorted by similarity descending.

        Returns:
            ConflictResult indicating the type and the conflicting item.
        """
        if not candidates:
            return ConflictResult(conflict_type=ConflictType.UNRELATED)

        best_item, best_sim = candidates[0]

        if best_sim >= self.auto_overwrite_threshold:
            return ConflictResult(
                conflict_type=ConflictType.OVERWRITE,
                existing_item=best_item,
                similarity=best_sim,
            )

        if best_sim >= self.association_threshold:
            return ConflictResult(
                conflict_type=ConflictType.ASSOCIATE,
                existing_item=best_item,
                similarity=best_sim,
            )

        return ConflictResult(
            conflict_type=ConflictType.UNRELATED,
            similarity=best_sim,
        )
