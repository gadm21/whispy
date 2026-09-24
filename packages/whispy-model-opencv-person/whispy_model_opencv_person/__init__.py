"""opencv-haar-person — installable person-presence model for Whispy.

A ``Processor`` plugin that consumes camera frames (JPEG/RGB payloads)
from a :class:`SensorWindow` and emits ``person``/``no_person``
predictions with detection boxes in ``attributes``.

The model knows nothing about which camera produced the frame — a
Windows webcam, a Pi camera, or a recorded fixture all arrive as the
same normalized ``SensorSample`` payloads.

Manifest equivalent::

    name: opencv-haar-person
    inputs:
      - name: video
        modality: camera
        capabilities: [rgb8_or_jpeg]
    outputs:
      task: person_presence
      labels: [person, no_person]
    resources: {cpu: true, gpu: false, memory_mb: 200}
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

from whispy.contracts import Prediction, SensorWindow
from whispy.models.runner import bound_samples
from whispy.processors.base import Processor, ProcessorMeta

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - dependency of this package
    cv2 = None  # type: ignore
    np = None  # type: ignore

DEFAULT_CASCADES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_upperbody.xml",
)


class HaarPersonModel(Processor):
    """Haar-cascade person detector over camera frames.

    Config::

        {
            "cascades": ["haarcascade_frontalface_default.xml"],
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": [48, 48],
            "input": "video"          # bound input name
        }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self._input = str(self._config.get("input") or "video")
        self._scale = float(self._config.get("scale_factor") or 1.1)
        self._neighbors = int(self._config.get("min_neighbors") or 5)
        self._min_size = tuple(self._config.get("min_size") or (48, 48))
        self._cascades = self._load_cascades()

    # -- setup ------------------------------------------------------------------
    def _load_cascades(self) -> List[Any]:
        if cv2 is None or not hasattr(cv2, "CascadeClassifier"):
            return []
        names = self._config.get("cascades") or list(DEFAULT_CASCADES)
        out = []
        for name in names:
            path = name if "\\" in name or "/" in name else \
                cv2.data.haarcascades + name
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                out.append(cascade)
            else:
                logger.warning("cascade %s failed to load", path)
        return out

    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="opencv-haar-person",
            version="0.1.0",
            processor_type="opencv-haar-person",
            sensor="camera",
            task="person_presence",
            inputs=("camera",),
            outputs=("person", "no_person"),
            hardware_reqs={"cpu": True, "gpu": False, "memory_mb": 200},
            config_schema={
                "type": "object",
                "properties": {
                    "cascades": {"type": "array", "items": {"type": "string"}},
                    "scale_factor": {"type": "number"},
                    "min_neighbors": {"type": "integer"},
                    "min_size": {"type": "array"},
                    "input": {"type": "string"},
                },
            },
        )

    def health(self) -> Dict[str, Any]:
        ok = cv2 is not None and bool(self._cascades)
        return {"status": "ok" if ok else "error",
                "cascades": len(self._cascades),
                "detail": "" if ok else "opencv or cascades unavailable"}

    # -- frame decoding -------------------------------------------------------------
    @staticmethod
    def _decode_frame(sample) -> Optional[Any]:
        """SensorSample payload → BGR ndarray (jpeg/png/rgb8/gray)."""
        if cv2 is None or np is None:
            return None
        payload = sample.payload
        if isinstance(payload, dict):
            encoding = str(payload.get("encoding") or "").lower()
            data = payload.get("data")
            if data is None:
                return None
            if isinstance(data, str):
                try:
                    data = base64.b64decode(data)
                except Exception:
                    return None
            if encoding in ("jpeg", "jpg", "png"):
                arr = np.frombuffer(bytes(data), dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if encoding in ("rgb8", "bgr8", "gray8"):
                w = int(payload.get("width") or 0)
                h = int(payload.get("height") or 0)
                channels = {"rgb8": 3, "bgr8": 3, "gray8": 1}.get(encoding, 3)
                arr = np.frombuffer(bytes(data), dtype=np.uint8)
                try:
                    img = arr.reshape(h, w, channels)
                except Exception:
                    return None
                if encoding == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding == "gray8":
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return img
            return None
        # Raw ndarray-like payload
        try:
            arr = np.asarray(payload)
            if arr.ndim == 2:
                return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if arr.ndim == 3:
                return arr.astype(np.uint8)
        except Exception:
            return None
        return None

    # -- inference ------------------------------------------------------------------
    def predict(self, window: SensorWindow) -> Prediction:
        if cv2 is None or not self._cascades:
            return Prediction(
                label="error", confidence=0.0, task="person_presence",
                metadata={"error": "opencv or cascade unavailable"})

        samples = bound_samples(window, self._input)
        if not samples:
            # Fall back: any camera-modality samples in the window.
            for sid, chunk in window.samples.items():
                if chunk and chunk[-1].sensor_type == "camera":
                    samples = chunk
                    break
        if not samples:
            return Prediction(
                label="no_person", confidence=0.0, task="person_presence",
                metadata={"reason": "no camera frame in window",
                          "count": 0, "boxes": []})

        frame = self._decode_frame(samples[-1])
        if frame is None:
            return Prediction(
                label="no_person", confidence=0.0, task="person_presence",
                metadata={"reason": "undecodable frame",
                          "count": 0, "boxes": []})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes: List[List[int]] = []
        for cascade in self._cascades:
            try:
                found = cascade.detectMultiScale(
                    gray, scaleFactor=self._scale,
                    minNeighbors=self._neighbors, minSize=self._min_size)
                boxes.extend([list(map(int, b)) for b in found])
            except Exception as exc:
                logger.warning("detectMultiScale failed: %s", exc)

        # De-duplicate overlapping boxes across cascades.
        boxes = self._dedupe(boxes)
        count = len(boxes)
        label = "person" if count > 0 else "no_person"
        confidence = min(0.99, 0.5 + 0.1 * count) if count else 0.9
        return Prediction(
            label=label, confidence=confidence, task="person_presence",
            people_count=count,
            scores={"person": confidence if count else 1.0 - confidence,
                    "no_person": 1.0 - confidence if count else confidence},
            metadata={"count": count, "boxes": boxes,
                      "frame_sensor": samples[-1].sensor_id,
                      "frame_device": samples[-1].device_id})

    @staticmethod
    def _dedupe(boxes: List[List[int]], iou: float = 0.4) -> List[List[int]]:
        """Drop boxes largely contained in another box."""
        def area(b):
            return max(0, b[2]) * max(0, b[3])

        def overlap(a, b):
            x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
            inter = x * y
            denom = min(area(a), area(b))
            return inter / denom if denom else 0.0

        out: List[List[int]] = []
        for box in sorted(boxes, key=area, reverse=True):
            if all(overlap(box, kept) < iou for kept in out):
                out.append(box)
        return out


__all__ = ["HaarPersonModel"]
