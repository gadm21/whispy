"""Standard processor interface for the ThothCraft model ecosystem.

Processor types
---------------
rule        Declarative threshold/logic evaluated by ``RuleProcessor`` —
            no artifact, deploys as pure config.
classical   Python DSP/feature pipeline (DBSCAN, PCA, sklearn).
torchscript Serialized ``.pt`` model executed by the device runtime.
fusion      Combines other processors' ``Prediction`` outputs.

A processor consumes a :class:`SensorWindow` (per-sensor numpy views of a
capture window) and returns a :class:`Prediction` — the same shape the
spatial-state engine and ROS2 bridge consume.
"""

from __future__ import annotations

import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np

PROCESSOR_TYPES = ("rule", "classical", "torchscript", "fusion")


@dataclass
class ProcessorMeta:
    """Registry/discovery metadata for a processor."""

    name: str
    version: str = "1.0.0"
    processor_type: str = "classical"          # one of PROCESSOR_TYPES
    sensor: str = "any"                        # radar | csi | camera | fusion | any
    task: str = "occupancy"                    # occupancy | har | localization | environmental
    inputs: tuple = ()                         # required sensor modalities
    outputs: tuple = ("label", "confidence")
    hardware_reqs: Dict[str, Any] = field(default_factory=dict)
    config_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema
    accuracy: Optional[float] = None
    dataset_provenance: Optional[str] = None

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
            "accuracy": self.accuracy,
            "dataset_provenance": self.dataset_provenance,
        }


@dataclass
class Prediction:
    """Standard processor output — consumed by spatial state, ROS2, HA."""

    label: str
    confidence: float = 1.0
    people_count: Optional[int] = None
    xy: Optional[np.ndarray] = None           # (N, 2) points in meters
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "label": self.label,
            "confidence": self.confidence,
            "people_count": self.people_count,
            "extras": self.extras,
        }
        if self.xy is not None:
            out["xy"] = np.asarray(self.xy).tolist()
        return out


class SensorWindow:
    """Per-sensor numpy views over a capture window (minute or chunk).

    Wraps a mapping of sensor name → samples (lists, ``SensorData``, or
    arrays) and exposes uniform ``.to_numpy()`` access plus simple
    derived features (``snr_mean`` etc.) that rule processors reference.
    """

    def __init__(self, sensors: Mapping[str, Any]):
        self._sensors = dict(sensors)

    @property
    def sensors(self) -> tuple:
        return tuple(self._sensors.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._sensors

    def __getitem__(self, name: str) -> np.ndarray:
        return self.to_numpy(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._sensors)

    def to_numpy(self, name: str, dtype=None) -> np.ndarray:
        data = self._sensors[name]
        if hasattr(data, "to_numpy"):
            return data.to_numpy(dtype=dtype)
        return np.asarray(data, dtype=dtype)

    def feature(self, name: str) -> float:
        """Derived scalar features usable in rule expressions.

        Supported: ``<sensor>_mean``, ``<sensor>_std``, ``<sensor>_max``,
        ``<sensor>_min``, ``<sensor>_energy`` (mean of squares),
        ``snr_mean`` (radar SNR column mean when present).
        """
        if name == "snr_mean":
            for key in ("radar_snr", "snr", "radar"):
                if key in self._sensors:
                    arr = np.asarray(self.to_numpy(key), dtype=float)
                    return float(np.nanmean(arr))
            raise KeyError("no SNR-bearing sensor in window")
        m = re.fullmatch(r"(\w+)_(mean|std|max|min|energy)", name)
        if m and m.group(1) in self._sensors:
            arr = np.asarray(self.to_numpy(m.group(1)), dtype=float)
            op = m.group(2)
            if op == "mean":
                return float(np.nanmean(arr))
            if op == "std":
                return float(np.nanstd(arr))
            if op == "max":
                return float(np.nanmax(arr))
            if op == "min":
                return float(np.nanmin(arr))
            return float(np.nanmean(arr * arr))
        raise KeyError(f"unknown feature: {name}")


class Processor(ABC):
    """Base class for all deployable processors."""

    @abstractmethod
    def metadata(self) -> ProcessorMeta:
        """Registry metadata: name, type, inputs, outputs, config schema."""

    @abstractmethod
    def predict(self, window: SensorWindow) -> Prediction:
        """Map a sensor window to a prediction."""

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply per-device tunables (e.g. ``snr_threshold``). Default no-op."""

    def health(self) -> Dict[str, Any]:
        return {"status": "ok"}


_OPS = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt,
    "<=": operator.le, "==": operator.eq, "!=": operator.ne,
}
_EXPR_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?|[A-Za-z_]\w*)\s*$")


class RuleProcessor(Processor):
    """Declarative threshold processor — the simplest deployable model.

    Config example::

        {
          "rules": [{"when": "snr_mean > snr_threshold", "label": "occupied"}],
          "else": "empty",
          "params": {"snr_threshold": 12.0}
        }

    ``params`` are defaults; per-device ``configure()`` overrides them.
    """

    def __init__(self, config: Dict[str, Any], meta: Optional[ProcessorMeta] = None):
        self._rules = config.get("rules") or []
        self._else = config.get("else", "unknown")
        self._params = dict(config.get("params") or {})
        self._meta = meta or ProcessorMeta(
            name=config.get("name", "rule-processor"),
            processor_type="rule",
            sensor=config.get("sensor", "any"),
            task=config.get("task", "occupancy"),
            config_schema=config.get("config_schema") or {},
        )

    def metadata(self) -> ProcessorMeta:
        return self._meta

    def configure(self, config: Dict[str, Any]) -> None:
        params = config.get("params", config)
        if isinstance(params, dict):
            self._params.update(params)

    def _resolve(self, token: str, window: SensorWindow) -> float:
        if token in self._params:
            return float(self._params[token])
        return window.feature(token)

    def predict(self, window: SensorWindow) -> Prediction:
        for rule in self._rules:
            expr = rule.get("when", "")
            m = _EXPR_RE.match(expr)
            if not m:
                continue
            lhs, op, rhs = m.groups()
            try:
                left = self._resolve(lhs, window)
            except KeyError:
                continue
            try:
                right = float(rhs)
            except ValueError:
                try:
                    right = self._resolve(rhs, window)
                except KeyError:
                    continue
            if _OPS[op](left, right):
                return Prediction(
                    label=rule.get("label", "positive"),
                    confidence=float(rule.get("confidence", 1.0)),
                    extras={"rule": expr, "value": left},
                )
        return Prediction(label=self._else, confidence=1.0)
