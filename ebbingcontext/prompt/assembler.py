"""Prompt assembler — token budget management and position orchestration.

Token Budget = Total Window - System Prompt - Pin - Tool Defs - Recent N Turns - Output Reserve

Position strategy (Lost-in-Middle mitigation):
    High-strength memories at beginning and end, medium in the middle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ebbingcontext.adapter.token_counter import TokenCounter
from ebbingcontext.core.scoring import ScoredMemory
from ebbingcontext.models import MemoryItem


@dataclass
class PromptSection:
    """A section of the assembled prompt."""

    role: str  # "system", "memory_pin", "memory_context", "recent_turns"
    content: str
    token_count: int
    source_ids: list[str] = field(default_factory=list)


@dataclass
class AssembledPrompt:
    """Result of prompt assembly."""

    sections: list[PromptSection]
    total_tokens: int
    budget_remaining: int
    memories_included: int
    memories_dropped: int


class PromptAssembler:
    """Assembles memories into a prompt that fits within a token budget."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        output_reserve: int = 1024,
        recent_turns: int = 3,
    ) -> None:
        self.token_counter = token_counter or TokenCounter()
        self.output_reserve = output_reserve
        self.recent_turns = recent_turns

    def assemble(
        self,
        total_window: int,
        system_prompt: str = "",
        tool_definitions: list[dict] | None = None,
        recent_messages: list[dict[str, str]] | None = None,
        pinned_memories: list[MemoryItem] | None = None,
        recalled_memories: list[ScoredMemory] | None = None,
    ) -> AssembledPrompt:
        """Assemble memories into a prompt that fits the token budget.

        Priority order (non-negotiable allocations first):
        1. System prompt
        2. Pinned memories
        3. Tool definitions
        4. Recent N turns
        5. Output reserve
        6. Remaining budget -> recalled memories (position-orchestrated)
        """
        sections: list[PromptSection] = []
        tool_definitions = tool_definitions or []
        recent_messages = recent_messages or []
        pinned_memories = pinned_memories or []
        recalled_memories = recalled_memories or []

        # 1. System prompt
        system_tokens = self.token_counter.count(system_prompt) if system_prompt else 0
        if system_prompt:
            sections.append(PromptSection(
                role="system",
                content=system_prompt,
                token_count=system_tokens,
            ))

        # 2. Pinned memories
        pin_contents = []
        pin_ids = []
        pin_tokens = 0
        for mem in pinned_memories:
            t = self.token_counter.count(mem.content)
            pin_contents.append(mem.content)
            pin_ids.append(mem.id)
            pin_tokens += t

        if pin_contents:
            sections.append(PromptSection(
                role="memory_pin",
                content="\n".join(pin_contents),
                token_count=pin_tokens,
                source_ids=pin_ids,
            ))

        # 3. Tool definitions
        tool_str = json.dumps(tool_definitions, ensure_ascii=False) if tool_definitions else ""
        tool_tokens = self.token_counter.count(tool_str) if tool_str else 0
        if tool_str:
            sections.append(PromptSection(
                role="tool_definitions",
                content=tool_str,
                token_count=tool_tokens,
            ))

        # 4. Recent turns
        recent_tokens = 0
        recent_contents = []
        for msg in recent_messages[-self.recent_turns:]:
            content = msg.get("content", "")
            t = self.token_counter.count(content)
            recent_tokens += t
            recent_contents.append(content)

        if recent_contents:
            sections.append(PromptSection(
                role="recent_turns",
                content="\n".join(recent_contents),
                token_count=recent_tokens,
            ))

        # 5. Calculate remaining budget
        fixed_tokens = system_tokens + pin_tokens + tool_tokens + recent_tokens + self.output_reserve
        memory_budget = max(0, total_window - fixed_tokens)

        # 6. Fill with recalled memories (position-orchestrated)
        arranged = self._apply_position_orchestration(recalled_memories)
        included, dropped = self._fill_budget(arranged, memory_budget)

        if included:
            mem_contents = []
            mem_ids = []
            mem_tokens = 0
            for sm in included:
                mem_contents.append(sm.item.content)
                mem_ids.append(sm.item.id)
                mem_tokens += self.token_counter.count(sm.item.content)

            sections.append(PromptSection(
                role="memory_context",
                content="\n".join(mem_contents),
                token_count=mem_tokens,
                source_ids=mem_ids,
            ))

        total = sum(s.token_count for s in sections) + self.output_reserve
        remaining = total_window - total

        return AssembledPrompt(
            sections=sections,
            total_tokens=total,
            budget_remaining=remaining,
            memories_included=len(included),
            memories_dropped=dropped,
        )

    def _apply_position_orchestration(
        self, memories: list[ScoredMemory]
    ) -> list[ScoredMemory]:
        """Apply Lost-in-Middle position orchestration.

        High-strength memories at beginning and end, medium in the middle.
        """
        if len(memories) <= 2:
            return list(memories)

        high = [m for m in memories if m.strength >= 0.7]
        medium = [m for m in memories if m.strength < 0.7]

        if not high or not medium:
            return list(memories)

        mid_point = (len(high) + 1) // 2
        head = high[:mid_point]
        tail = high[mid_point:]

        return head + medium + tail

    def _fill_budget(
        self, memories: list[ScoredMemory], budget: int
    ) -> tuple[list[ScoredMemory], int]:
        """Greedily fill the token budget with memories.

        Skips items that exceed remaining budget (doesn't break —
        later smaller items may still fit).
        """
        included: list[ScoredMemory] = []
        used = 0

        for mem in memories:
            cost = self.token_counter.count(mem.item.content)
            if used + cost <= budget:
                included.append(mem)
                used += cost

        dropped = len(memories) - len(included)
        return included, dropped
