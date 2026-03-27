"""OpenAI-compatible LLM adapter.

Supports any OpenAI-compatible API endpoint.
Requires: pip install ebbingcontext[openai]
"""

from __future__ import annotations

from ebbingcontext.adapter.base import LLMAdapter, LLMResponse
from ebbingcontext.adapter.token_counter import TokenCounter


class OpenAIAdapter(LLMAdapter):
    """LLM adapter using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install ebbingcontext[openai]"
            )

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._token_counter = TokenCounter()

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    def count_tokens(self, text: str) -> int:
        return self._token_counter.count(text)
