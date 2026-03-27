"""MemoryEngine — the central orchestrator.

Coordinates all subsystems: classification, decay, storage, migration, conflict resolution.
This is the main entry point for both the Python API and the MCP tools.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import uuid as _uuid

from ebbingcontext.core.chunker import MessageChunker
from ebbingcontext.core.classifier import Classifier

if TYPE_CHECKING:
    from ebbingcontext.config import EbbingConfig
    from ebbingcontext.embedding.base import EmbeddingProvider
    from ebbingcontext.prompt.assembler import AssembledPrompt
from ebbingcontext.core.conflict import ConflictResolver, ConflictType
from ebbingcontext.core.decay import DecayEngine
from ebbingcontext.core.migration import MigrationEngine, MigrationDirection
from ebbingcontext.core.scoring import ScoredMemory, ScoringEngine
from ebbingcontext.models import (
    AuditRecord,
    DecayStrategy,
    MemoryItem,
    SensitivityLevel,
    StorageLayer,
)
from ebbingcontext.storage.active import ActiveStore
from ebbingcontext.storage.archive import ArchiveStore
from ebbingcontext.storage.warm import WarmStore


class MemoryEngine:
    """Central memory management engine."""

    def __init__(
        self,
        # Decay params
        s_base: float = 1.0,
        alpha: float = 0.3,
        beta_active: float = 1.2,
        beta_warm: float = 0.8,
        rho: float = 0.1,
        # Storage thresholds
        theta_active: float = 0.6,
        theta_archive: float = 0.15,
        # Conflict thresholds
        auto_overwrite_threshold: float = 0.9,
        association_threshold: float = 0.7,
        # Pin constraints
        pin_max_ratio: float = 0.3,
        # Embedding provider
        embedding_provider: "EmbeddingProvider | None" = None,
        # Warm retrieval threshold
        warm_retrieval_threshold: float = 0.5,
    ) -> None:
        self.decay_engine = DecayEngine(
            s_base=s_base, alpha=alpha,
            beta_active=beta_active, beta_warm=beta_warm, rho=rho,
        )
        self.scoring_engine = ScoringEngine(self.decay_engine)
        self.classifier = Classifier()
        self.migration_engine = MigrationEngine(theta_active, theta_archive)
        self.conflict_resolver = ConflictResolver(auto_overwrite_threshold, association_threshold)

        self.active = ActiveStore()
        self.warm = WarmStore()
        self.archive = ArchiveStore()

        self.pin_max_ratio = pin_max_ratio
        self.embedding_provider = embedding_provider
        self.warm_retrieval_threshold = warm_retrieval_threshold
        self.chunker = MessageChunker()

    @classmethod
    def from_config(cls, config: "EbbingConfig") -> "MemoryEngine":
        """Create a MemoryEngine from an EbbingConfig."""
        from ebbingcontext.embedding import create_embedding_provider
        from ebbingcontext.storage import create_stores

        embedding_provider = create_embedding_provider(config.embedding)
        active, warm, archive = create_stores(config)

        engine = cls(
            s_base=config.decay.s_base,
            alpha=config.decay.alpha,
            beta_active=config.decay.beta_active,
            beta_warm=config.decay.beta_warm,
            rho=config.decay.rho,
            theta_active=config.storage.theta_active,
            theta_archive=config.storage.theta_archive,
            auto_overwrite_threshold=config.conflict.auto_overwrite_threshold,
            association_threshold=config.conflict.association_threshold,
            pin_max_ratio=config.pin.max_ratio,
            embedding_provider=embedding_provider,
            warm_retrieval_threshold=config.prompt.warm_retrieval_threshold,
        )
        engine.active = active
        engine.warm = warm
        engine.archive = archive
        return engine

    def store(
        self,
        content: str,
        agent_id: str = "default",
        source_type: str = "user",
        importance: float | None = None,
        decay_strategy: DecayStrategy | None = None,
        sensitivity: SensitivityLevel | None = None,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> MemoryItem:
        """Store a new memory item.

        LLM decides *whether* to call this. System decides *how* to store.
        Override parameters (importance, decay_strategy, sensitivity) are for C-mode.
        """
        # Step 1: Classify
        classification = self.classifier.classify(content, source_type)

        # Step 2: Build memory item (allow C-mode overrides)
        item = MemoryItem(
            content=content,
            agent_id=agent_id,
            decay_strategy=decay_strategy or classification.decay_strategy,
            sensitivity=sensitivity or classification.sensitivity,
            importance=importance if importance is not None else classification.importance,
            source_type=source_type,
            embedding=embedding,
            metadata=metadata or {},
        )

        # Step 2.5: Auto-embed if provider available and no embedding given
        if item.embedding is None and self.embedding_provider is not None:
            item.embedding = self.embedding_provider.embed(content)
            embedding = item.embedding

        # Step 3: Pin check
        if item.decay_strategy == DecayStrategy.PIN:
            if self.active.get_pin_ratio(agent_id) >= self.pin_max_ratio:
                raise ValueError(
                    f"Pin ratio limit ({self.pin_max_ratio:.0%}) reached. "
                    "Unpin existing memories before pinning new ones."
                )

        # Step 4: Conflict detection (if embedding available)
        if embedding is not None:
            candidates = self.warm.search(embedding, top_k=1, agent_id=agent_id)
            # Also check active items with embeddings
            for active_item in self.active.get_all(agent_id):
                if active_item.embedding is not None:
                    from numpy import array, dot
                    from numpy.linalg import norm

                    a = array(embedding)
                    b = array(active_item.embedding)
                    na, nb = norm(a), norm(b)
                    if na > 0 and nb > 0:
                        sim = float(dot(a / na, b / nb))
                        candidates.append((active_item, sim))

            candidates.sort(key=lambda x: x[1], reverse=True)
            conflict = self.conflict_resolver.detect(item, candidates)

            if conflict.conflict_type == ConflictType.OVERWRITE and conflict.existing_item:
                # Archive the old item
                old = conflict.existing_item
                self._archive_item(old, event="overwritten")
                self.active.remove(old.id)
                self.warm.remove(old.id)
            elif conflict.conflict_type == ConflictType.ASSOCIATE and conflict.existing_item:
                # Link them
                item.metadata["associated_with"] = conflict.existing_item.id
                conflict.existing_item.metadata.setdefault("associated_with_list", [])
                conflict.existing_item.metadata["associated_with_list"].append(item.id)

        # Step 5: Store in Active layer
        self.active.add(item)

        # Step 6: Audit
        self.archive.add_audit_record(
            AuditRecord(
                memory_id=item.id,
                event="created",
                to_layer=StorageLayer.ACTIVE,
                strength_at_event=item.strength,
                details={"source_type": source_type, "classification_confidence": classification.confidence},
            )
        )

        return item

    def store_message(
        self,
        content: str,
        agent_id: str = "default",
        source_type: str = "user",
        **kwargs,
    ) -> list[MemoryItem]:
        """Parse, chunk, and store a full message. Returns all stored items.

        Uses MessageChunker to split by source type, then stores each chunk.
        Chunks in the same message share a chunk_group ID.
        """
        chunks = self.chunker.chunk(content, source_type)
        if not chunks:
            return []

        group_id = _uuid.uuid4().hex
        items: list[MemoryItem] = []

        for chunk in chunks:
            merged_meta = {
                **kwargs.pop("metadata", {}),
                "chunk_group": group_id,
                **chunk.metadata,
            }
            item = self.store(
                content=chunk.content,
                agent_id=agent_id,
                source_type=chunk.source_type,
                metadata=merged_meta,
                **kwargs,
            )
            items.append(item)

        return items

    def recall(
        self,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        agent_id: str = "default",
        top_k: int = 10,
        current_token_pos: int | None = None,
    ) -> list[ScoredMemory]:
        """Retrieve relevant memories.

        Active memories first, then search Warm if needed.
        """
        now = time.time()
        candidates: list[tuple[MemoryItem, float]] = []

        # Auto-embed query if provider available
        if query_embedding is None and query is not None and self.embedding_provider is not None:
            query_embedding = self.embedding_provider.embed(query)

        # Get all active items for this agent
        active_items = self.active.get_all(agent_id)

        # Update strengths
        self.decay_engine.batch_update(active_items, now, current_token_pos)

        if query_embedding is not None:
            # Score active items by similarity
            for item in active_items:
                if item.embedding is not None:
                    from numpy import array, dot
                    from numpy.linalg import norm

                    a = array(query_embedding)
                    b = array(item.embedding)
                    na, nb = norm(a), norm(b)
                    if na > 0 and nb > 0:
                        sim = float(dot(a / na, b / nb))
                        candidates.append((item, sim))
                else:
                    # No embedding, give a baseline similarity
                    candidates.append((item, 0.5))

            # Check if we need Warm layer retrieval
            best_active_sim = max((s for _, s in candidates), default=0.0)
            if best_active_sim < self.warm_retrieval_threshold:
                warm_results = self.warm.search(query_embedding, top_k=top_k, agent_id=agent_id)
                for warm_item, sim in warm_results:
                    self.decay_engine.update_strength(warm_item, now, current_token_pos)
                    candidates.append((warm_item, sim))
        else:
            # No embedding, return all active items with default similarity
            candidates = [(item, 0.5) for item in active_items]

        # Rank
        scored = self.scoring_engine.rank_memories(candidates, now, current_token_pos, top_k)

        # Touch retrieved items (boost access count for decay recovery)
        for sm in scored:
            sm.item.touch()

        return scored

    def recall_for_prompt(
        self,
        query: str | None = None,
        agent_id: str = "default",
        total_window: int = 128000,
        system_prompt: str = "",
        tool_definitions: list[dict] | None = None,
        recent_messages: list[dict] | None = None,
        top_k: int = 20,
    ) -> "AssembledPrompt":
        """Recall memories and assemble them into a prompt-ready format."""
        from ebbingcontext.prompt.assembler import PromptAssembler

        pinned = self.active.get_pinned(agent_id)
        recalled = self.recall(query=query, agent_id=agent_id, top_k=top_k)
        # Exclude pinned from recalled (they're already included separately)
        recalled = [r for r in recalled if r.item.decay_strategy != DecayStrategy.PIN]

        assembler = PromptAssembler()
        return assembler.assemble(
            total_window=total_window,
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
            recent_messages=recent_messages,
            pinned_memories=pinned,
            recalled_memories=recalled,
        )

    def pin(self, memory_id: str, agent_id: str = "default") -> MemoryItem:
        """Pin a memory (mark as unforgettable)."""
        item = self._find_item(memory_id)
        if item is None:
            raise KeyError(f"Memory {memory_id} not found")

        if item.decay_strategy == DecayStrategy.PIN:
            return item  # Already pinned

        if self.active.get_pin_ratio(agent_id) >= self.pin_max_ratio:
            raise ValueError(
                f"Pin ratio limit ({self.pin_max_ratio:.0%}) reached."
            )

        item.decay_strategy = DecayStrategy.PIN
        item.strength = 1.0

        # If in Warm, promote to Active
        if item.layer == StorageLayer.WARM:
            self.warm.remove(item.id)
            self.active.add(item)

        self.archive.add_audit_record(
            AuditRecord(
                memory_id=item.id,
                event="pinned",
                strength_at_event=1.0,
            )
        )
        return item

    def forget(self, memory_id: str) -> MemoryItem:
        """Explicitly forget a memory (move to Archive, not delete)."""
        item = self._find_item(memory_id)
        if item is None:
            raise KeyError(f"Memory {memory_id} not found")

        self.active.remove(item.id)
        self.warm.remove(item.id)
        self._archive_item(item, event="forgotten")
        return item

    def inspect(self, memory_id: str) -> dict:
        """Inspect a memory's current state."""
        item = self._find_item(memory_id)
        if item is None:
            raise KeyError(f"Memory {memory_id} not found")

        now = time.time()
        current_strength = self.decay_engine.compute_strength(item, now)
        audit_trail = self.archive.get_audit_trail(memory_id)

        return {
            "item": item.model_dump(exclude={"embedding"}),
            "current_strength": current_strength,
            "stability": self.decay_engine.compute_stability(item),
            "audit_trail": [r.model_dump() for r in audit_trail],
        }

    def transfer(
        self,
        memory_id: str,
        from_agent: str,
        to_agent: str,
        max_sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL,
    ) -> MemoryItem | None:
        """Transfer a memory between agents with sensitivity filtering."""
        item = self._find_item(memory_id)
        if item is None:
            raise KeyError(f"Memory {memory_id} not found")

        if item.agent_id != from_agent:
            raise PermissionError(f"Memory {memory_id} does not belong to agent {from_agent}")

        # Sensitivity gate
        sensitivity_order = {
            SensitivityLevel.PUBLIC: 0,
            SensitivityLevel.INTERNAL: 1,
            SensitivityLevel.SENSITIVE: 2,
        }
        if sensitivity_order[item.sensitivity] > sensitivity_order[max_sensitivity]:
            return None  # Filtered out

        # Create a copy for the target agent
        transferred = item.model_copy(
            update={
                "id": None,  # Will get new ID from default_factory
                "agent_id": to_agent,
                "metadata": {**item.metadata, "transferred_from": from_agent, "original_id": item.id},
            }
        )
        # Force new ID
        import uuid
        transferred.id = uuid.uuid4().hex

        self.active.add(transferred)

        self.archive.add_audit_record(
            AuditRecord(
                memory_id=item.id,
                event="transferred",
                strength_at_event=item.strength,
                details={"from_agent": from_agent, "to_agent": to_agent, "new_id": transferred.id},
            )
        )
        return transferred

    def run_migration(self, agent_id: str = "default") -> int:
        """Run migration cycle: update strengths and migrate items between layers.

        Returns the number of items migrated.
        """
        now = time.time()
        migrated = 0

        # Update active items and check for demotion
        active_items = self.active.get_all(agent_id)
        self.decay_engine.batch_update(active_items, now)

        actions = self.migration_engine.evaluate_batch(active_items)
        for action in actions:
            if action.direction == MigrationDirection.DEMOTE:
                self.active.remove(action.item.id)
                if action.to_layer == StorageLayer.WARM:
                    self.warm.add(action.item)
                else:
                    self._archive_item(action.item, event="migrated")
                migrated += 1

        # Update warm items and check for demotion to archive
        warm_items = self.warm.get_all(agent_id)
        self.decay_engine.batch_update(warm_items, now)

        actions = self.migration_engine.evaluate_batch(warm_items)
        for action in actions:
            if action.direction == MigrationDirection.DEMOTE:
                self.warm.remove(action.item.id)
                self._archive_item(action.item, event="migrated")
                migrated += 1

        return migrated

    def _find_item(self, memory_id: str) -> MemoryItem | None:
        """Find an item across all layers."""
        item = self.active.get(memory_id)
        if item:
            return item
        item = self.warm.get(memory_id)
        if item:
            return item
        return self.archive.get(memory_id)

    def _archive_item(self, item: MemoryItem, event: str = "archived") -> None:
        """Move an item to archive with audit record."""
        from_layer = item.layer
        self.archive.add(item)
        self.archive.add_audit_record(
            AuditRecord(
                memory_id=item.id,
                event=event,
                from_layer=from_layer,
                to_layer=StorageLayer.ARCHIVE,
                strength_at_event=item.strength,
            )
        )
