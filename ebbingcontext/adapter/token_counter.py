"""Standalone token counter — no external dependencies required.

Heuristic: ~4 characters per token for English/code, ~2 characters per token for CJK.
Conservative (slight overcount) to avoid context window overflow.
"""

from __future__ import annotations

import re

# CJK Unicode ranges
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


class TokenCounter:
    """Estimates token counts without requiring tiktoken or an LLM API."""

    def count(self, text: str) -> int:
        """Estimate token count.

        Uses character-based heuristic, slightly overestimates for safety.
        """
        if not text:
            return 0

        cjk_chars = len(_CJK_PATTERN.findall(text))
        other_chars = len(text) - cjk_chars

        # CJK: ~1.5 chars per token (conservative)
        # Latin/code: ~3.5 chars per token (conservative)
        cjk_tokens = cjk_chars / 1.5
        other_tokens = other_chars / 3.5

        return max(1, int(cjk_tokens + other_tokens + 0.5))
