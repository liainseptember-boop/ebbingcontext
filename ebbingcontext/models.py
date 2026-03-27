"""Core data models for EbbingContext."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DecayStrategy(str, Enum):
    """Decay strategy for a memory item."""

    PIN = "pin"
    DECAY_RECOVERABLE = "decay_recoverable"
    DECAY_IRREVERSIBLE = "decay_irreversible"


class SensitivityLevel(str, Enum):
    """Sensitivity level for a memory item."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"


class StorageLayer(str, Enum):
    """Which storage layer a memory resides in."""

    ACTIVE = "active"
    WARM = "warm"
    ARCHIVE = "archive"


class MemoryItem(BaseModel):
    """A single memory item with metadata and decay state."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    agent_id: str = "default"

    # Dual labels
    decay_strategy: DecayStrategy = DecayStrategy.DECAY_RECOVERABLE
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL

    # Scoring
    importance: float = 0.5
    strength: float = 1.0

    # Decay tracking
    access_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)
    token_position: int = 0  # Token distance tracking (intra-session)

    # Storage
    layer: StorageLayer = StorageLayer.ACTIVE
    embedding: list[float] | None = None

    # Metadata
    source_type: str = "user"  # user / agent / tool
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Record an access (retrieval/reference)."""
        self.access_count += 1
        self.last_accessed_at = time.time()


class AuditRecord(BaseModel):
    """Audit trail entry for memory lifecycle events."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    memory_id: str
    event: str  # created / migrated / updated / archived / transferred
    from_layer: StorageLayer | None = None
    to_layer: StorageLayer | None = None
    strength_at_event: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    details: dict[str, Any] = Field(default_factory=dict)
