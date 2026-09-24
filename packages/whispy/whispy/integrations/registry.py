"""Integration registry — declared external context sources/sinks.

Integrations are declared, not hardcoded: each entry describes a
provider, its direction (source/sink), and its config schema. Concrete
drivers plug in later; the registry keeps the surface stable.
"""

from __future__ import annotations

from typing import Any, Dict, List


class IntegrationRegistry:
    """Declared integrations: external sources and action providers."""

    def __init__(self):
        self._integrations: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, kind: str,
                 config_schema: Dict[str, Any] | None = None,
                 metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if kind not in ("source", "sink", "both"):
            raise ValueError("kind must be source|sink|both")
        entry = {"name": name, "kind": kind,
                 "config_schema": config_schema or {},
                 "enabled": False, "metadata": metadata or {}}
        self._integrations[name] = entry
        return entry

    def enable(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._integrations.get(name)
        if entry is None:
            raise KeyError(f"unknown integration {name!r}")
        entry["config"] = dict(config)
        entry["enabled"] = True
        return entry

    def disable(self, name: str) -> Dict[str, Any]:
        entry = self._integrations.get(name)
        if entry is None:
            raise KeyError(f"unknown integration {name!r}")
        entry["enabled"] = False
        return entry

    def list(self) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._integrations.values()]
