"""Storage layer migration strategy.

Migration is strength-threshold-driven:
    - Active (strength > θ₁=0.6): participates in prompt assembly
    - Warm (θ₂ < strength ≤ θ₁): vector DB, retrieved when needed
    - Archive (strength ≤ θ₂=0.15): audit trail, preserved for history

Key design: "last chance" retrieval before migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ebbingcontext.models import MemoryItem, StorageLayer


class MigrationDirection(str, Enum):
    """Direction of migration."""

    DEMOTE = "demote"  # Active → Warm → Archive
    PROMOTE = "promote"  # Warm → Active (on retrieval)


@dataclass
class MigrationAction:
    """A pending migration action for a memory item."""

    item: MemoryItem
    direction: MigrationDirection
    from_layer: StorageLayer
    to_layer: StorageLayer


class MigrationEngine:
    """Determines which memories should migrate between storage layers."""

    def __init__(
        self,
        theta_active: float = 0.6,
        theta_archive: float = 0.15,
    ) -> None:
        self.theta_active = theta_active
        self.theta_archive = theta_archive

    def evaluate(self, item: MemoryItem) -> MigrationAction | None:
        """Evaluate whether a memory item should migrate.

        Returns a MigrationAction if migration is needed, None otherwise.
        """
        strength = item.strength

        if item.layer == StorageLayer.ACTIVE:
            if strength <= self.theta_archive:
                # Skip Warm, go directly to Archive
                return MigrationAction(
                    item=item,
                    direction=MigrationDirection.DEMOTE,
                    from_layer=StorageLayer.ACTIVE,
                    to_layer=StorageLayer.ARCHIVE,
                )
            if strength <= self.theta_active:
                return MigrationAction(
                    item=item,
                    direction=MigrationDirection.DEMOTE,
                    from_layer=StorageLayer.ACTIVE,
                    to_layer=StorageLayer.WARM,
                )

        elif item.layer == StorageLayer.WARM:
            if strength <= self.theta_archive:
                return MigrationAction(
                    item=item,
                    direction=MigrationDirection.DEMOTE,
                    from_layer=StorageLayer.WARM,
                    to_layer=StorageLayer.ARCHIVE,
                )
            if strength > self.theta_active:
                # Promoted back to Active (e.g., after retrieval boosted strength)
                return MigrationAction(
                    item=item,
                    direction=MigrationDirection.PROMOTE,
                    from_layer=StorageLayer.WARM,
                    to_layer=StorageLayer.ACTIVE,
                )

        return None

    def evaluate_batch(self, items: list[MemoryItem]) -> list[MigrationAction]:
        """Evaluate migration for a batch of items."""
        actions: list[MigrationAction] = []
        for item in items:
            action = self.evaluate(item)
            if action is not None:
                actions.append(action)
        return actions
