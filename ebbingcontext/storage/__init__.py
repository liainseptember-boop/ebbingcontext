"""Storage layer implementations and factory.

Provides in-memory stores (default) and persistent stores (ChromaDB + SQLite + JSON).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebbingcontext.config import EbbingConfig


def create_stores(config: "EbbingConfig"):
    """Create storage layer instances based on config.

    Returns (active, warm, archive) tuple.
    When persist=True, uses JSON/ChromaDB/SQLite backends.
    """
    if config.storage.persist:
        from ebbingcontext.storage.active import ActiveStore
        from ebbingcontext.storage.archive_sqlite import SQLiteArchiveStore
        from ebbingcontext.storage.warm_chroma import ChromaWarmStore

        active = ActiveStore(persist_path=config.storage.active_persist_path)
        warm = ChromaWarmStore(persist_dir=config.vector_store.persist_dir)
        archive = SQLiteArchiveStore(db_path=config.storage.archive_db_path)
    else:
        from ebbingcontext.storage.active import ActiveStore
        from ebbingcontext.storage.archive import ArchiveStore
        from ebbingcontext.storage.warm import WarmStore

        active = ActiveStore()
        warm = WarmStore()
        archive = ArchiveStore()

    return active, warm, archive
