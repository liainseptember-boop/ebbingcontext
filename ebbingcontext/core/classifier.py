"""Dual-label classification: decay strategy + sensitivity level.

Design:
    - Rule-first: deterministic rules cover ~60-70% of cases
    - LLM fallback: rules can't cover → delegate to LLM
    - Separation of concerns: LLM decides "whether to store", system decides "how to store"

Evolution path: V1 rules+LLM → V2 rules+dedicated_classifier+LLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebbingcontext.adapter.base import LLMAdapter

from ebbingcontext.models import DecayStrategy, SensitivityLevel


@dataclass
class ClassificationResult:
    """Result of dual-label classification."""

    decay_strategy: DecayStrategy
    sensitivity: SensitivityLevel
    importance: float  # 0.0 ~ 1.0
    confidence: float  # How confident the classifier is (0.0 ~ 1.0)
    rule_matched: str | None = None  # Which rule matched, None if LLM fallback


# Patterns for sensitivity detection
_SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_\s]?key|secret|password|token|credential|private[_\s]?key)"),
    re.compile(r"(?i)(ssn|social\s*security|credit\s*card|bank\s*account)"),
    re.compile(r"(?i)(bearer\s+[a-zA-Z0-9\-._~+/]+=*)"),
    re.compile(r"[a-zA-Z0-9]{32,}"),  # Long alphanumeric strings (potential keys)
]

_INTERNAL_PATTERNS = [
    re.compile(r"(?i)(internal|confidential|proprietary|do\s*not\s*share)"),
    re.compile(r"(?i)(user\s*(name|id|email|phone|address))"),
]

# Patterns for decay strategy detection
_PIN_PATTERNS = [
    re.compile(r"(?i)(system\s*prompt|system\s*instruction|role\s*definition)"),
    re.compile(r"(?i)(must\s*remember|never\s*forget|always\s*remember|critical\s*info)"),
    re.compile(r"(?i)(identity|persona|objective|constraint|rule)"),
]

_IRREVERSIBLE_PATTERNS = [
    re.compile(r"(?i)(debug|trace|verbose|log\s*output)"),
    re.compile(r"(?i)(intermediate\s*result|step\s*\d+\s*output)"),
    re.compile(r"(?i)(raw\s*response|full\s*output|dump)"),
]


class RuleClassifier:
    """Deterministic rule-based classifier for the common cases."""

    def classify_sensitivity(self, content: str) -> tuple[SensitivityLevel, str | None]:
        """Classify sensitivity level based on content patterns."""
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(content):
                return SensitivityLevel.SENSITIVE, f"sensitive:{pattern.pattern}"

        for pattern in _INTERNAL_PATTERNS:
            if pattern.search(content):
                return SensitivityLevel.INTERNAL, f"internal:{pattern.pattern}"

        return SensitivityLevel.PUBLIC, None

    def classify_decay_strategy(
        self, content: str, source_type: str
    ) -> tuple[DecayStrategy, str | None]:
        """Classify decay strategy based on content and source."""
        # System prompts are always pinned
        if source_type == "system":
            return DecayStrategy.PIN, "source:system"

        for pattern in _PIN_PATTERNS:
            if pattern.search(content):
                return DecayStrategy.PIN, f"pin:{pattern.pattern}"

        for pattern in _IRREVERSIBLE_PATTERNS:
            if pattern.search(content):
                return DecayStrategy.DECAY_IRREVERSIBLE, f"irreversible:{pattern.pattern}"

        # Default: recoverable decay
        return DecayStrategy.DECAY_RECOVERABLE, None

    def estimate_importance(self, content: str, source_type: str) -> float:
        """Estimate importance score based on heuristics.

        Returns a value between 0.0 and 1.0.
        """
        score = 0.5  # baseline

        # Source type adjustments
        if source_type == "system":
            score += 0.3
        elif source_type == "user":
            score += 0.1
        elif source_type == "tool":
            score -= 0.1

        # Length heuristic: very short or very long content is less important
        length = len(content)
        if length < 20:
            score -= 0.1
        elif length > 2000:
            score -= 0.1
        elif 50 <= length <= 500:
            score += 0.1

        # Question marks suggest queries (slightly higher)
        if "?" in content:
            score += 0.05

        return max(0.0, min(1.0, score))


class LLMClassifier:
    """LLM-based classification for cases where rules are not confident."""

    SYSTEM_PROMPT = (
        "You are a memory classification assistant. Given a piece of text and its source type, "
        "classify it on two dimensions:\n"
        '1. decay_strategy: "pin" (critical, must not forget), "decay_recoverable" (useful, '
        'can be re-strengthened), "decay_irreversible" (low-value, monotonic decay)\n'
        '2. sensitivity: "public", "internal", "sensitive"\n'
        "3. importance: 0.0 to 1.0\n\n"
        'Respond in JSON only: {"decay_strategy": "...", "sensitivity": "...", "importance": 0.X}'
    )

    def __init__(self, adapter: "LLMAdapter") -> None:
        self.adapter = adapter

    async def classify(self, content: str, source_type: str) -> ClassificationResult | None:
        """Classify using LLM. Returns None on failure (never propagates errors)."""
        try:
            import json as _json

            response = await self.adapter.complete(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Source: {source_type}\nContent: {content}"},
                ],
                temperature=0,
            )
            data = _json.loads(response.content)
            return ClassificationResult(
                decay_strategy=DecayStrategy(data["decay_strategy"]),
                sensitivity=SensitivityLevel(data["sensitivity"]),
                importance=float(data["importance"]),
                confidence=0.8,
                rule_matched="llm",
            )
        except Exception:
            return None


class Classifier:
    """Main classifier combining rules and LLM fallback.

    V1: Rule-based classification with confidence scores.
    When confidence is low, marks the result for LLM fallback.
    """

    def __init__(self, llm_adapter: "LLMAdapter | None" = None, confidence_threshold: float = 0.6) -> None:
        self.rule_classifier = RuleClassifier()
        self.llm_classifier = LLMClassifier(llm_adapter) if llm_adapter else None
        self.confidence_threshold = confidence_threshold

    def classify(
        self,
        content: str,
        source_type: str = "user",
    ) -> ClassificationResult:
        """Classify a piece of content with dual labels.

        Args:
            content: the text content to classify.
            source_type: "user" / "agent" / "tool" / "system".

        Returns:
            ClassificationResult with decay strategy, sensitivity, and importance.
        """
        decay_strategy, decay_rule = self.rule_classifier.classify_decay_strategy(
            content, source_type
        )
        sensitivity, sens_rule = self.rule_classifier.classify_sensitivity(content)
        importance = self.rule_classifier.estimate_importance(content, source_type)

        # Confidence: higher when rules matched explicitly
        rule_matched_parts = []
        confidence = 0.5  # baseline for no explicit rule match

        if decay_rule:
            rule_matched_parts.append(decay_rule)
            confidence += 0.2
        if sens_rule:
            rule_matched_parts.append(sens_rule)
            confidence += 0.15

        confidence = min(1.0, confidence)
        rule_matched = " | ".join(rule_matched_parts) if rule_matched_parts else None

        # Adjust importance for pinned items
        if decay_strategy == DecayStrategy.PIN:
            importance = max(importance, 0.8)

        return ClassificationResult(
            decay_strategy=decay_strategy,
            sensitivity=sensitivity,
            importance=importance,
            confidence=confidence,
            rule_matched=rule_matched,
        )

    async def classify_with_fallback(
        self,
        content: str,
        source_type: str = "user",
    ) -> ClassificationResult:
        """Classify with LLM fallback for low-confidence results.

        Error policy: LLM failure → return rule-based result. Never propagate errors.
        """
        result = self.classify(content, source_type)

        if result.confidence >= self.confidence_threshold or self.llm_classifier is None:
            return result

        llm_result = await self.llm_classifier.classify(content, source_type)
        if llm_result is not None:
            return llm_result

        # LLM failed, return rule-based result
        return result
