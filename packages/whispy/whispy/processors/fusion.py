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

    def predict(self, window: SensorWindow) -> Prediction:
        unavailable = set(window.missing()) | set(window.stale())
        blocked = [s for s in self._required if s in unavailable]
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
        complete = window.is_complete()
        return Prediction(
            label="complete" if complete else "degraded",
            confidence=1.0 if complete else 0.5,
            metadata={"unavailable": sorted(unavailable)},
        )


__all__ = ["FusionProcessor"]
