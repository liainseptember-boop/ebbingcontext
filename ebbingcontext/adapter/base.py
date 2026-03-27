"""Abstract base for LLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM completion."""

    content: str
    usage: dict[str, int]  # {"prompt_tokens": ..., "completion_tokens": ...}


class LLMAdapter(ABC):
    """Abstract LLM provider for classification fallback and future features."""

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        ...
