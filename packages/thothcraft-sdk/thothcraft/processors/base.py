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

import logging
import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

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
        ``snr_mean`` (radar SNR column mean when present), and direct scalar
        fields like ``temperature_c``, ``pressure_mbar``, ``snr_db``, etc.
        """
        # Direct scalar in sensors mapping
        if name in self._sensors and isinstance(self._sensors[name], (int, float)):
            return float(self._sensors[name])
        # Check subdictionaries in sensors
        for s_val in self._sensors.values():
            if isinstance(s_val, dict) and name in s_val and isinstance(s_val[name], (int, float)):
                return float(s_val[name])
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
    """Declarative threshold and rule processor — supports sensor thresholds,

    computer-vision face detection, and downstream actuator triggering.
    """

    def __init__(self, config: Dict[str, Any], meta: Optional[ProcessorMeta] = None):
        self._config = config
        self._rules = config.get("rules") or []
        self._else = config.get("else", "unknown")
        self._params = dict(config.get("params") or {})
        self.rule_type = str(config.get("rule_type") or ("face_detection" if config.get("sensor") == "camera" else "threshold"))
        self._meta = meta or ProcessorMeta(
            name=config.get("name", "rule-processor"),
            processor_type="rule",
            sensor=config.get("sensor", "any"),
            task=config.get("task", "occupancy"),
            config_schema=config.get("config_schema") or {},
        )
        self._face_cascade = None
        self._actuator = None
        if config.get("actuator"):
            try:
                from ..actuators import create_actuator
                self._actuator = create_actuator(config["actuator"])
            except Exception as exc:
                logger.warning("Could not initialize actuator: %s", exc)

    def metadata(self) -> ProcessorMeta:
        return self._meta

    def configure(self, config: Dict[str, Any]) -> None:
        params = config.get("params", config)
        if isinstance(params, dict):
            self._params.update(params)
        if config.get("actuator"):
            try:
                from ..actuators import create_actuator
                self._actuator = create_actuator(config["actuator"])
            except Exception as exc:
                logger.warning("Could not update actuator: %s", exc)

    def _resolve(self, token: str, window: SensorWindow) -> float:
        try:
            return window.feature(token)
        except KeyError:
            if token in self._params:
                return float(self._params[token])
            raise

    def _get_face_cascade(self):
        if self._face_cascade is None:
            try:
                import cv2  # type: ignore
                if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as exc:
                logger.warning("OpenCV face cascade unavailable: %s", exc)
        return self._face_cascade

    def _predict_face(self, window: SensorWindow) -> Prediction:
        """Run OpenCV face detection on camera frame."""
        frame_data = None
        for k in ("camera", "video", "frame"):
            if k in window:
                frame_data = window._sensors[k]
                break
        if frame_data is None:
            return Prediction(label=self._else, confidence=0.0, extras={"error": "no camera frame"})

        try:
            import cv2  # type: ignore
        except ImportError:
            return Prediction(label=self._else, confidence=0.0, extras={"error": "opencv not installed"})

        # Decode JPEG bytes or use numpy array
        if isinstance(frame_data, bytes):
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif hasattr(frame_data, "shape"):
            img = np.asarray(frame_data)
        else:
            return Prediction(label=self._else, confidence=0.0, extras={"error": "unsupported frame format"})

        if img is None:
            return Prediction(label=self._else, confidence=0.0, extras={"error": "failed to decode frame"})

        cascade = self._get_face_cascade()
        boxes = []
        if cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                boxes = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]
        else:
            # Fallback: YCrCb skin-color & facial contour heuristic for environments without cascade binaries
            if len(img.shape) == 3 and img.shape[2] == 3:
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8),
                                   np.array([255, 173, 127], dtype=np.uint8))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h > 1200 and 0.65 <= (h / max(1, w)) <= 1.8:
                        boxes.append([int(x), int(y), int(w), int(h)])

        has_face = len(boxes) > 0
        pos_label = "face_detected"
        if self._rules:
            pos_label = self._rules[0].get("label", "face_detected")

        if has_face:
            return Prediction(
                label=pos_label,
                confidence=0.95,
                people_count=len(boxes),
                extras={"faces_count": len(boxes), "bounding_boxes": boxes},
            )
        return Prediction(label=self._else, confidence=1.0, people_count=0)

    def predict(self, window: SensorWindow) -> Prediction:
        if self.rule_type in ("face_detection", "cv_face") or any(r.get("rule_type") == "face_detection" for r in self._rules):
            pred = self._predict_face(window)
        else:
            pred = None
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
                    pred = Prediction(
                        label=rule.get("label", "positive"),
                        confidence=float(rule.get("confidence", 1.0)),
                        extras={"rule": expr, "value": left},
                    )
                    break
            if pred is None:
                pred = Prediction(label=self._else, confidence=1.0)

        # Trigger attached actuator plugin if present
        if self._actuator is not None:
            try:
                self._actuator.trigger(pred)
            except Exception as exc:
                logger.warning("Actuator trigger failed: %s", exc)

        return pred
