"""Processor contract — consumes real SensorWindows, emits Predictions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..contracts import PROCESSOR_TYPES, Prediction, SensorWindow


@dataclass
class ProcessorMeta:
    """Registry/discovery metadata for a processor."""

    name: str
    version: str = "1.0.0"
    processor_type: str = "rule"               # one of PROCESSOR_TYPES
    sensor: str = "any"                        # radar | csi | camera | fusion | any
    task: str = "occupancy"
    inputs: tuple = ()                         # required sensor modalities
    outputs: tuple = ("label", "confidence")
    hardware_reqs: Dict[str, Any] = field(default_factory=dict)
    config_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "processor_type": self.processor_type,
            "sensor": self.sensor,
            "task": self.task,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "hardware_reqs": self.hardware_reqs,
            "config_schema": self.config_schema,
        }


class Processor(ABC):
    """Base class for all deployable processors."""

    @abstractmethod
    def metadata(self) -> ProcessorMeta:
        """Registry metadata: name, type, inputs, outputs, config schema."""

    @abstractmethod
    def predict(self, window: SensorWindow) -> Prediction:
        """Map a sensor window to a prediction."""

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply per-device tunables. Default no-op."""

    def health(self) -> Dict[str, Any]:
        return {"status": "ok"}


def create_processor(manifest_or_config: Dict[str, Any],
                     artifact: Optional[bytes] = None) -> Processor:
    """Instantiate the processor named by a manifest/config.

    ``processor`` selects the implementation: ``rule`` → RuleProcessor,
    ``torchscript`` → TorchScriptProcessor, ``fusion`` → FusionProcessor.
    """
    from .rules import RuleProcessor
    from .torchscript import TorchScriptProcessor
    from .fusion import FusionProcessor

    kind = str(manifest_or_config.get("processor")
               or manifest_or_config.get("processor_type") or "rule")
    if kind == "rule":
        return RuleProcessor(manifest_or_config)
    if kind == "torchscript":
        return TorchScriptProcessor(manifest_or_config, artifact=artifact)
    if kind == "fusion":
        return FusionProcessor(manifest_or_config)
    # Plugin processors registered under the whispy.models entry-point
    # group are valid deployment targets too.
    try:
        from ..models.registry import installed_models
        cls = installed_models().get(kind)
        if cls is not None:
            return cls(manifest_or_config)
    except Exception:
        pass
    raise ValueError(
        f"unknown processor type {kind!r}; expected one of {PROCESSOR_TYPES} "
        f"or an installed whispy.models plugin")


__all__ = [
    "PROCESSOR_TYPES",
    "Processor",
    "ProcessorMeta",
    "create_processor",
]
