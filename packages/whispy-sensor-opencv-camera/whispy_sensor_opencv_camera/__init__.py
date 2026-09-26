"""OpenCV camera sensor adapter — physical camera discovery for Whispy.

Enumerates real capture devices via ``cv2.VideoCapture`` and exports each
as a :class:`SensorDescriptor` with a stable id derived from the device's
persistent hardware identity where the OS exposes one (Windows PnP
instance id via WMI; otherwise a backend+index fallback).

The adapter knows nothing about what consumes the frames — person
detectors, recorders, and remote streamers all receive the same
normalized ``SensorSample``::

    payload = {"encoding": "jpeg", "width": 1280, "height": 720,
               "data": "<base64 jpeg>"}
"""

from __future__ import annotations

import base64
import itertools
import logging
import platform
import subprocess
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from whispy.contracts import SensorDescriptor, SensorSample
from whispy.devices.base import SensorHandle
from whispy.sensors.base import HealthReport, SensorAdapter, SensorMeta

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - dependency of this package
    cv2 = None  # type: ignore


def _windows_camera_devices() -> List[Dict[str, str]]:
    """PnP camera instances on Windows: name + persistent InstanceId."""
    if platform.system() != "Windows":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -Class CAMERA -Status OK -ErrorAction "
             "SilentlyContinue | Select-Object FriendlyName,InstanceId | "
             "ConvertTo-Json"],
            capture_output=True, text=True, timeout=10)
        import json
        data = json.loads(out.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
        return [{"name": d.get("FriendlyName", ""),
                 "instance_id": d.get("InstanceId", "")}
                for d in data if d.get("InstanceId")]
    except Exception:
        return []


def _backend_flag() -> int:
    if cv2 is None:
        return 0
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW          # DirectShow: fastest reliable default
    if platform.system() == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class _CameraHandle(SensorHandle):
    """Streams normalized JPEG frames from one camera descriptor."""

    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        self._config = dict(config or {})
        self._cap = None
        self._seq = itertools.count()
        self._lock = threading.Lock()

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _open(self):
        if self._cap is not None:
            return
        index = int(self._desc.metadata.get("capture_index", 0))
        cap = cv2.VideoCapture(index, _backend_flag())
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"cannot open camera index {index} ({self._desc.id})")
        width = int(self._config.get("width") or 0)
        height = int(self._config.get("height") or 0)
        fps = float(self._config.get("fps") or 0)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap = cap

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        self._open()
        cap = self._cap
        assert cap is not None
        fps = float(self._config.get("fps")
                    or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        period = 1.0 / fps if fps > 0 else 0.0
        quality = int(self._config.get("jpeg_quality") or 80)
        count = 0
        try:
            while True:
                with self._lock:
                    ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    continue
                h, w = frame.shape[:2]
                yield SensorSample(
                    device_id="",
                    sensor_id=self._desc.id,
                    sensor_type="camera",
                    timestamp=time.time(),
                    sequence=next(self._seq),
                    payload_type="jpeg",
                    payload={
                        "encoding": "jpeg",
                        "width": int(w),
                        "height": int(h),
                        "data": base64.b64encode(buf.tobytes()).decode("ascii"),
                    },
                    sample_rate=fps,
                    metadata={"adapter": self._desc.adapter,
                              "hardware_id": self._desc.hardware_id},
                )
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
                if period:
                    time.sleep(period)
        finally:
            # Release the sensor so the camera's privacy LED turns off the
            # moment nobody is streaming — and re-opens on the next tail.
            self.close()

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


class OpenCvCameraAdapter(SensorAdapter):
    """Discovers UVC/integrated cameras via OpenCV.

    ``discover()`` probes capture indices ``0..max_index`` and pairs them
    with OS-reported camera names/ids when available (Windows PnP). Each
    camera gets a stable id ``camera-<sha1(hardware_id)[:4]>``.
    """

    def __init__(self, max_index: int = 4):
        self._max_index = max_index
        self._handles: List[_CameraHandle] = []

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="opencv-camera",
            version="0.1.0",
            modalities=("camera",),
            description="UVC/integrated cameras via OpenCV VideoCapture",
            config_schema={
                "type": "object",
                "properties": {
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "fps": {"type": "number"},
                    "jpeg_quality": {"type": "integer"},
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[SensorDescriptor]:
        if cv2 is None:
            return []
        pnp = _windows_camera_devices()
        out: List[SensorDescriptor] = []
        for index in range(self._max_index):
            cap = None
            try:
                cap = cv2.VideoCapture(index, _backend_flag())
                if not cap.isOpened():
                    cap.release()
                    continue
                # isOpened() alone isn't enough: Linux exposes non-capture
                # V4L2 nodes (codec/M2M/ISP) that "open" but never produce
                # frames — the Pi reported a phantom camera. Require a real
                # frame (up to ~1s) before advertising the device.
                deadline = time.time() + 1.0
                ok, frame = False, None
                while time.time() < deadline:
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        break
                    time.sleep(0.05)
                if not ok or frame is None or not getattr(frame, "size", 0):
                    cap.release()
                    continue
                width = int(frame.shape[1] if getattr(frame, "size", 0) else 0)
                height = int(frame.shape[0] if getattr(frame, "size", 0) else 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            except Exception:
                if cap is not None:
                    cap.release()
                continue
            finally:
                if cap is not None:
                    cap.release()

            dev = pnp[index] if index < len(pnp) else {}
            hw = dev.get("instance_id") or \
                f"{platform.system()}-camera-index-{index}"
            name = dev.get("name") or f"Camera {index}"
            out.append(SensorDescriptor(
                id=SensorDescriptor.make_id("camera", hw),
                modality="camera",
                adapter="opencv-camera",
                name=name,
                hardware_id=hw,
                capabilities=["rgb8", "jpeg"],
                config_schema=self.metadata().config_schema,
                stable=bool(dev.get("instance_id")),
                metadata={
                    "capture_index": index,
                    "backend": "opencv",
                    "width": width, "height": height,
                    "fps_max": fps,
                },
            ))
        return out

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _CameraHandle:
        handle = _CameraHandle(descriptor, config)
        self._handles.append(handle)
        return handle

    def health(self) -> HealthReport:
        return HealthReport(
            status="ok" if cv2 is not None else "error",
            detail="" if cv2 is not None else "opencv-python not installed")

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


__all__ = ["OpenCvCameraAdapter"]
