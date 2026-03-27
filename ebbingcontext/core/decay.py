"""Ebbinghaus forgetting curve decay algorithm.

Core formula:
    R(t) = exp(-(Δt / S)^β)

    S = S₀ × (1 + α × ln(1 + n))
      - S₀ = importance × S_base
      - α = 0.3 (configurable)
      - n = access_count

    score_final = sim(query, memory) × R̃(t)
    R̃(t) = ρ + (1 - ρ) × R(t)    # ρ=0.1 floor retention

Dual-dimension decay:
    - Intra-session: Δt measured in token distance
    - Cross-session: Δt measured in physical time (seconds)

Layer-specific β:
    - Active: β=1.2 (super-linear, fast decay)
    - Warm:   β=0.8 (sub-linear, slow decay)
"""

from __future__ import annotations

import math

from ebbingcontext.models import DecayStrategy, MemoryItem, StorageLayer


# Default parameters
DEFAULT_S_BASE = 1.0
DEFAULT_ALPHA = 0.3
DEFAULT_BETA_ACTIVE = 1.2
DEFAULT_BETA_WARM = 0.8
DEFAULT_RHO = 0.1


class DecayEngine:
    """Computes memory strength decay based on the Ebbinghaus forgetting curve."""

    def __init__(
        self,
        s_base: float = DEFAULT_S_BASE,
        alpha: float = DEFAULT_ALPHA,
        beta_active: float = DEFAULT_BETA_ACTIVE,
        beta_warm: float = DEFAULT_BETA_WARM,
        rho: float = DEFAULT_RHO,
    ) -> None:
        self.s_base = s_base
        self.alpha = alpha
        self.beta_active = beta_active
        self.beta_warm = beta_warm
        self.rho = rho

    def _get_beta(self, layer: StorageLayer) -> float:
        """Get the β parameter for a storage layer."""
        if layer == StorageLayer.ACTIVE:
            return self.beta_active
        return self.beta_warm

    def compute_stability(self, item: MemoryItem) -> float:
        """Compute the stability S for a memory item.

        S = S₀ × (1 + α × ln(1 + n))
        S₀ = importance × S_base
        """
        s0 = item.importance * self.s_base
        return s0 * (1.0 + self.alpha * math.log(1.0 + item.access_count))

    def compute_retention(
        self,
        delta_t: float,
        stability: float,
        beta: float,
    ) -> float:
        """Compute raw retention R(t) = exp(-(Δt / S)^β).

        Returns 1.0 if stability is zero (avoid division by zero).
        """
        if stability <= 0:
            return 0.0
        if delta_t <= 0:
            return 1.0
        ratio = delta_t / stability
        return math.exp(-(ratio**beta))

    def compute_effective_retention(self, raw_retention: float) -> float:
        """Apply floor retention: R̃(t) = ρ + (1 - ρ) × R(t)."""
        return self.rho + (1.0 - self.rho) * raw_retention

    def compute_strength(
        self,
        item: MemoryItem,
        current_time: float,
        current_token_pos: int | None = None,
    ) -> float:
        """Compute the current strength of a memory item.

        Pin items always return strength 1.0.

        For decaying items, uses the appropriate Δt:
        - Intra-session (token distance) if current_token_pos is provided
        - Cross-session (physical time) otherwise
        """
        if item.decay_strategy == DecayStrategy.PIN:
            return 1.0

        beta = self._get_beta(item.layer)
        stability = self.compute_stability(item)

        # Choose delta based on whether we're doing intra-session or cross-session
        if current_token_pos is not None:
            delta_t = max(0, current_token_pos - item.token_position)
        else:
            delta_t = max(0.0, current_time - item.last_accessed_at)

        raw_r = self.compute_retention(delta_t, stability, beta)
        return self.compute_effective_retention(raw_r)

    def update_strength(
        self,
        item: MemoryItem,
        current_time: float,
        current_token_pos: int | None = None,
    ) -> float:
        """Compute and update the strength field on a memory item. Returns new strength."""
        strength = self.compute_strength(item, current_time, current_token_pos)
        item.strength = strength
        return strength

    def batch_update(
        self,
        items: list[MemoryItem],
        current_time: float,
        current_token_pos: int | None = None,
    ) -> list[MemoryItem]:
        """Update strengths for a batch of memory items. Returns items sorted by strength desc."""
        for item in items:
            self.update_strength(item, current_time, current_token_pos)
        return sorted(items, key=lambda x: x.strength, reverse=True)
