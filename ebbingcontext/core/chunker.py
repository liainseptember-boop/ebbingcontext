"""Message parsing and chunking.

Strategy per source type (from spec):
    - Tool output: split by JSON fields (fallback: by line)
    - User messages: split by sentence
    - Agent replies: split by paragraph
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


MIN_CHUNK_LENGTH = 10


@dataclass
class Chunk:
    """A single chunk of a parsed message."""

    content: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageChunker:
    """Splits messages into chunks based on source type."""

    def __init__(self, min_chunk_length: int = MIN_CHUNK_LENGTH) -> None:
        self.min_chunk_length = min_chunk_length

    def chunk(self, content: str, source_type: str) -> list[Chunk]:
        """Split content based on source type.

        Returns at least one chunk. Short content is never split.
        """
        content = content.strip()
        if not content:
            return []

        # Short content: don't split
        if len(content) < self.min_chunk_length * 2:
            return [Chunk(content=content, source_type=source_type)]

        if source_type == "tool":
            return self._chunk_tool_output(content)
        elif source_type == "agent":
            return self._chunk_by_paragraph(content)
        else:
            # "user" and others: by sentence
            return self._chunk_by_sentence(content)

    def _chunk_tool_output(self, content: str) -> list[Chunk]:
        """Parse JSON tool output and split by top-level fields.

        Falls back to line-based splitting if not valid JSON.
        """
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return self._chunk_by_line(content)

        if not isinstance(data, dict):
            return [Chunk(content=content, source_type="tool")]

        chunks: list[Chunk] = []
        for key, value in data.items():
            value_str = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
            chunk_content = f"{key}: {value_str}"
            chunks.append(Chunk(
                content=chunk_content,
                source_type="tool",
                metadata={"field_name": key},
            ))

        return chunks if chunks else [Chunk(content=content, source_type="tool")]

    def _chunk_by_sentence(self, content: str) -> list[Chunk]:
        """Split on sentence boundaries."""
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r"(?<=[.!?。！？])\s+", content)
        return self._merge_short_chunks(parts, "user")

    def _chunk_by_paragraph(self, content: str) -> list[Chunk]:
        """Split on double newlines or paragraph breaks."""
        parts = re.split(r"\n\s*\n", content)
        return self._merge_short_chunks(parts, "agent")

    def _chunk_by_line(self, content: str) -> list[Chunk]:
        """Split by non-empty lines (fallback for non-JSON tool output)."""
        parts = [line.strip() for line in content.split("\n") if line.strip()]
        return self._merge_short_chunks(parts, "tool")

    def _merge_short_chunks(self, parts: list[str], source_type: str) -> list[Chunk]:
        """Merge fragments shorter than min_chunk_length into adjacent chunks."""
        if not parts:
            return []

        merged: list[str] = []
        buffer = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if buffer:
                buffer = buffer + " " + part
            else:
                buffer = part

            if len(buffer) >= self.min_chunk_length:
                merged.append(buffer)
                buffer = ""

        # Flush remaining buffer
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        return [Chunk(content=text, source_type=source_type) for text in merged]
