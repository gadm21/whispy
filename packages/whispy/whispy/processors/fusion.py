"""FusionProcessor — combines multiple modalities/processor outputs.

Fusion receives a SensorWindow whose ``modalities`` map carries explicit
missing/stale markers (§47). A fusion model must declare in its config
how to treat missing modalities::

    {
        "processor": "fusion",
        "name": "hvac-fusion",
        "inputs": [{"sensor": "radar"}, {"sensor": "env"}],
        "on_missing": "abstain"        # abstain | zero | ignore
    }

``abstain`` (default) returns an explicit low-confidence "unknown"
prediction when a required modality is missing or stale. ``zero`` is only
permitted when the manifest explicitly declares it — never silently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..contracts import Prediction, SensorWindow
from .base import Processor, ProcessorMeta
from .rules import RuleProcessor

logger = logging.getLogger(__name__)

_ON_MISSING = ("abstain", "zero", "ignore")


class FusionProcessor(Processor):
    """Fuses features across modalities, then applies a rule stage."""

    def __init__(self, config: Dict[str, Any],
                 meta: Optional[ProcessorMeta] = None):
        self._config = config
        self._inputs = list(config.get("inputs") or [])
        self._on_missing = str(config.get("on_missing") or "abstain")
        if self._on_missing not in _ON_MISSING:
            raise ValueError(
                f"on_missing must be one of {_ON_MISSING}, got {self._on_missing!r}")
        self._required = [i.get("sensor") for i in self._inputs
                          if isinstance(i, dict) and i.get("required", True)]
        self._rule = RuleProcessor(config) if (config.get("rules") or config.get("rule")) else None
        self._meta = meta or ProcessorMeta(
            name=config.get("name", "fusion-processor"),
            processor_type="fusion",
            sensor="fusion",
            task=config.get("task", "fusion"),
            inputs=tuple(i.get("sensor") for i in self._inputs
                         if isinstance(i, dict)),
        )

    def metadata(self) -> ProcessorMeta:
        return self._meta

    @staticmethod
    def _resolve_sensor(window: SensorWindow, name: str) -> Optional[str]:
        from ..windows import resolve_sensor_id
        return resolve_sensor_id(window, name)

    def _available(self, window: SensorWindow, resolved: Optional[str]) -> bool:
        """A required input is available only with live, non-empty data."""
        if resolved is None:
            return False
        marker = window.modalities.get(resolved)
        if marker is not None:
            return marker.state == "ok"
        # No explicit marker: fall back to actual sample presence so a
        # sensor that never opened (no marker, no samples) is unavailable.
        return bool(window.samples.get(resolved))

    def predict(self, window: SensorWindow) -> Prediction:
        blocked = [r for r in self._required
                   if not self._available(window, self._resolve_sensor(window, r))]
        if blocked and self._on_missing == "abstain":
            return Prediction(
                label="unknown", confidence=0.0,
                metadata={"reason": "required modalities unavailable",
                          "unavailable": blocked})
        if self._rule is not None:
            pred = self._rule.predict(window)
            pred.metadata.setdefault("fusion", True)
            if blocked:
                pred.metadata["degraded"] = blocked
            return pred
        # No rule stage: report modality completeness as the prediction.
        complete = window.is_complete() and not blocked
        return Prediction(
            label="complete" if complete else "degraded",
            confidence=1.0 if complete else 0.5,
            metadata={"unavailable": sorted(blocked)},
        )


__all__ = ["FusionProcessor"]
