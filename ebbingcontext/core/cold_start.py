"""Cold start — template loading and batch import for new agents.

Provides seed memories from YAML templates so agents start with
useful context instead of an empty memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebbingcontext.engine import MemoryEngine


TEMPLATES_DIR = Path(__file__).parent.parent / "prompt" / "templates"


def list_templates() -> list[str]:
    """List available template names."""
    if not TEMPLATES_DIR.exists():
        return []
    return [p.stem for p in TEMPLATES_DIR.glob("*.yaml")]


def load_template(name: str) -> dict:
    """Load a template by name. Returns the parsed YAML dict."""
    import yaml

    path = TEMPLATES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Template '{name}' not found at {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Template '{name}' must be a YAML mapping")
    return data


def apply_template(engine: "MemoryEngine", name: str, agent_id: str = "default") -> int:
    """Apply a template to seed an agent's memory.

    Returns the number of memories stored.
    """
    template = load_template(name)
    memories = template.get("memories", [])
    count = 0

    for entry in memories:
        content = entry.get("content")
        if not content:
            continue
        engine.store(
            content=content,
            agent_id=agent_id,
            source_type=entry.get("source_type", "system"),
            importance=entry.get("importance"),
            decay_strategy=None,
            metadata=entry.get("metadata", {}),
        )
        count += 1

    return count
