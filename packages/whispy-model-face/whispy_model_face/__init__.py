"""whispy-model-face — face detection + eigenface recognition plugins.

Two ``Processor`` plugins:

- ``opencv-haar-face`` (:class:`HaarFaceModel`) — builtin OpenCV
  frontal-face detector. Emits ``face``/``no_face`` with boxes and
  base64 JPEG crops in ``detections`` so downstream models can consume
  the crop without re-detecting.

- ``pca-face-recognizer`` (:class:`EigenfaceRecognizer`) — eigenface
  recognition ported from gadm21/Face-recognition-using-PCA-and-SVD:
  crop → normalize → project into a PCA basis → nearest gallery
  projection by Euclidean distance → ``person:<name>`` when the
  distance is under ``max_distance`` ("very close"), else
  ``person:unknown``.

  The basis (mean + eigenvectors) and the gallery (name → projections)
  come from Brain's face asset store — ``GET /v1/faces/basis`` and
  ``GET /v1/faces/gallery`` — typically via
  :class:`whispy.cloud.faces.FaceGallery`. A basis can also be fitted
  locally from a face dataset (e.g. Olivetti) with
  :func:`whispy_model_face.basis.fit_basis` and pushed to Brain.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

from whispy.contracts import Prediction, SensorWindow
from whispy.models.runner import bound_samples
from whispy.processors.base import Processor, ProcessorMeta

from . import basis as _basis

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - hard dependency
    np = None  # type: ignore


def _decode_frame(sample) -> Optional[Any]:
    """SensorSample payload → ndarray (jpeg/png/rgb8/gray8/ndarray)."""
    if np is None:
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
            if cv2 is None:
                return None
            arr = np.frombuffer(bytes(data), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if encoding in ("rgb8", "bgr8", "gray8"):
            w = int(payload.get("width") or 0)
            h = int(payload.get("height") or 0)
            ch = {"rgb8": 3, "bgr8": 3, "gray8": 1}.get(encoding, 3)
            arr = np.frombuffer(bytes(data), dtype=np.uint8)
            try:
                return arr.reshape(h, w, ch)
            except Exception:
                return None
        return None
    try:
        arr = np.asarray(payload)
        return arr if arr.ndim in (2, 3) else None
    except Exception:
        return None


def _latest_frame(window: SensorWindow, input_name: str):
    """Bound input → latest decodable frame, else any camera sample."""
    samples = bound_samples(window, input_name)
    if not samples:
        for _, chunk in window.samples.items():
            if chunk and chunk[-1].sensor_type == "camera":
                samples = chunk
                break
    if not samples:
        return None, None
    return _decode_frame(samples[-1]), samples[-1]


class HaarFaceModel(Processor):
    """Builtin OpenCV frontal-face detector.

    Emits ``face``/``no_face``; ``metadata.detections`` carries
    ``{box, crop_jpeg}`` per face so a downstream recognizer can consume
    crops directly (the reference repo's detect→crop→recognize chain).

    Config::

        {"scale_factor": 1.1, "min_neighbors": 5, "min_size": [50, 50],
         "input": "video", "emit_crops": true}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self._input = str(self._config.get("input") or "video")
        self._scale = float(self._config.get("scale_factor") or 1.1)
        self._neighbors = int(self._config.get("min_neighbors") or 5)
        self._min_size = tuple(self._config.get("min_size") or (50, 50))
        self._emit_crops = bool(self._config.get("emit_crops", True))
        self._cascade = None
        if cv2 is not None and hasattr(cv2, "CascadeClassifier"):
            path = str(self._config.get("cascade")
                       or cv2.data.haarcascades
                       + "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(path)
            self._cascade = None if cascade.empty() else cascade

    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="opencv-haar-face", version="0.1.0",
            processor_type="opencv-haar-face", sensor="camera",
            task="face_detection", inputs=("camera",),
            outputs=("face", "no_face"),
            hardware_reqs={"cpu": True, "gpu": False, "memory_mb": 150},
            config_schema={
                "type": "object",
                "properties": {
                    "cascade": {"type": "string"},
                    "scale_factor": {"type": "number"},
                    "min_neighbors": {"type": "integer"},
                    "min_size": {"type": "array"},
                    "input": {"type": "string"},
                    "emit_crops": {"type": "boolean"},
                },
            })

    def health(self) -> Dict[str, Any]:
        ok = self._cascade is not None
        return {"status": "ok" if ok else "error",
                "detail": "" if ok else "opencv/cascade unavailable"}

    def predict(self, window: SensorWindow) -> Prediction:
        if self._cascade is None:
            return Prediction(label="no_face", confidence=0.0,
                              task="face_detection",
                              metadata={"error": "opencv/cascade unavailable",
                                        "detections": []})
        frame, sample = _latest_frame(window, self._input)
        if frame is None:
            return Prediction(label="no_face", confidence=0.0,
                              task="face_detection",
                              metadata={"reason": "no decodable frame",
                                        "detections": []})
        gray = frame if frame.ndim == 2 else cv2.cvtColor(
            frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        try:
            found = self._cascade.detectMultiScale(
                gray, scaleFactor=self._scale,
                minNeighbors=self._neighbors, minSize=self._min_size)
        except Exception as exc:
            logger.warning("detectMultiScale failed: %s", exc)
            found = []

        detections: List[Dict[str, Any]] = []
        for (x, y, w, h) in [list(map(int, b)) for b in found]:
            det: Dict[str, Any] = {"box": [x, y, w, h]}
            if self._emit_crops:
                crop = frame[y:y + h, x:x + w]
                ok, buf = cv2.imencode(".jpg", crop)
                if ok:
                    det["crop_jpeg"] = base64.b64encode(
                        buf.tobytes()).decode("ascii")
            detections.append(det)

        count = len(detections)
        return Prediction(
            label="face" if count else "no_face",
            confidence=min(0.99, 0.6 + 0.1 * count) if count else 0.9,
            task="face_detection", people_count=count,
            metadata={"detections": detections, "count": count,
                      "frame_sensor": getattr(sample, "sensor_id", ""),
                      "frame_device": getattr(sample, "device_id", "")})


class EigenfaceRecognizer(Processor):
    """Eigenface recognizer — PCA projection + nearest-gallery match.

    Config::

        {
            "basis": <npz bytes | dict>,        # mean + eigenvectors
            "gallery": {"name": [[weights]]},   # enrolled projections
            "max_distance": 12.0,               # "very close" cutoff;
                                                # 0 → calibrate_threshold
            "input": "video",
            "detect": true                      # haar-detect when no
                                                # upstream crop is given
        }

    Crop sources, in order: ``window.metadata["face_crop"]`` (upstream
    detector), ``window.metadata["detections"][0]["box"]``, own Haar
    detection (``detect: true``), else the whole frame.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self._input = str(self._config.get("input") or "video")
        self._detect = bool(self._config.get("detect", True))
        raw = self._config.get("basis")
        if isinstance(raw, (bytes, bytearray)):
            self._basis = _basis.deserialize_basis(bytes(raw))
        elif isinstance(raw, dict):
            self._basis = raw
        else:
            self._basis = None
        self._gallery: Dict[str, List[List[float]]] = {
            str(k): [list(p) for p in v]
            for k, v in dict(self._config.get("gallery") or {}).items()}
        self._max_distance = float(self._config.get("max_distance") or 0.0)
        if not self._max_distance and self._gallery:
            self._max_distance = _basis.calibrate_threshold(self._gallery)
        self._detector = None
        if self._detect and cv2 is not None \
                and hasattr(cv2, "CascadeClassifier"):
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml")
            self._detector = None if cascade.empty() else cascade

    def set_gallery(self, gallery: Dict[str, List[List[float]]],
                    max_distance: Optional[float] = None) -> None:
        """Hot-swap the enrolled gallery (e.g. after FaceGallery pull)."""
        self._gallery = {str(k): [list(p) for p in v]
                         for k, v in gallery.items()}
        if max_distance:
            self._max_distance = float(max_distance)
        elif not self._max_distance:
            self._max_distance = _basis.calibrate_threshold(self._gallery)

    def set_basis(self, basis: Dict[str, Any]) -> None:
        self._basis = basis

    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="pca-face-recognizer", version="0.1.0",
            processor_type="pca-face-recognizer", sensor="camera",
            task="face_recognition", inputs=("camera",),
            outputs=("person:<name>", "person:unknown", "no_face"),
            hardware_reqs={"cpu": True, "gpu": False, "memory_mb": 250},
            config_schema={
                "type": "object",
                "properties": {
                    "basis": {}, "gallery": {"type": "object"},
                    "max_distance": {"type": "number"},
                    "input": {"type": "string"},
                    "detect": {"type": "boolean"},
                },
            })

    def health(self) -> Dict[str, Any]:
        ok = self._basis is not None
        return {"status": "ok" if ok else "error",
                "persons": len(self._gallery),
                "detail": "" if ok else "no basis configured"}

    # -- crop resolution ---------------------------------------------------
    def _crop(self, window: SensorWindow, frame) -> Optional[Any]:
        meta = dict(getattr(window, "metadata", {}) or {})
        crop = meta.get("face_crop")
        if crop is not None:
            return crop
        detections = meta.get("detections") or []
        if detections and isinstance(detections[0], dict):
            box = detections[0].get("box")
            if box:
                x, y, w, h = [int(v) for v in box]
                return frame[y:y + h, x:x + w]
        if self._detector is not None:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(
                frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            found = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            if len(found):
                x, y, w, h = max(found, key=lambda b: b[2] * b[3])
                return frame[int(y):int(y + h), int(x):int(x + w)]
            return None
        return frame                      # assume the frame is the face

    # -- inference ---------------------------------------------------------
    def predict(self, window: SensorWindow) -> Prediction:
        if np is None or self._basis is None:
            return Prediction(label="person:unknown", confidence=0.0,
                              task="face_recognition",
                              metadata={"error": "no basis configured"})
        frame, sample = _latest_frame(window, self._input)
        if frame is None:
            return Prediction(label="no_face", confidence=0.0,
                              task="face_recognition",
                              metadata={"reason": "no decodable frame"})
        crop = self._crop(window, frame)
        if crop is None:
            return Prediction(label="no_face", confidence=0.0,
                              task="face_recognition",
                              metadata={"reason": "no face detected"})

        projection = _basis.project(self._basis, crop)
        if projection is None:
            return Prediction(label="no_face", confidence=0.0,
                              task="face_recognition",
                              metadata={"reason": "undecodable crop"})
        name, dist = _basis.nearest(self._gallery, projection)
        threshold = self._max_distance
        matched = name is not None and (not threshold or dist <= threshold)
        label = f"person:{name}" if matched else (
            "person:unknown" if name is not None else "no_face")
        # Confidence: distance mapped into (0,1] — 1.0 at d=0, ~0 at the
        # threshold. Unknown faces report low confidence by construction.
        confidence = (max(0.05, 1.0 - dist / threshold)
                      if matched and threshold else
                      (0.5 if matched else 0.2))
        return Prediction(
            label=label, confidence=round(confidence, 4),
            task="face_recognition",
            metadata={"person": name if matched else None,
                      "distance": round(dist, 4),
                      "max_distance": threshold,
                      "matched": bool(matched),
                      "frame_sensor": getattr(sample, "sensor_id", ""),
                      "frame_device": getattr(sample, "device_id", "")})


__all__ = ["HaarFaceModel", "EigenfaceRecognizer"]
