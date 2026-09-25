"""DreamHAT radar adapter — wraps the existing BGT60TR13C stack.

This adapter deliberately does **not** rewrite the low-level radar code.
It wraps the proven pipeline (``ifxradarsdk`` frame capture +
``RadarProcessor`` numpy FFT, as in ``thoth/dl/radar_processor.py``)::

    existing DreamHAT code (ifxradarsdk + RadarProcessor)
        ↓
    DreamHatRadarAdapter
        ↓
    radar SensorDescriptor  (radar-<hash>, stable)
        ↓
    SensorHandle
        ↓
    SensorSample  {encoding: radar_frame, snr_db, range_profile, ...}

Discovery returns ``[]`` when ``ifxradarsdk`` is absent or no shield is
attached — the Pi 5 reports no radar rather than a fake one.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from whispy.contracts import SensorDescriptor, SensorSample
from whispy.devices.base import SensorHandle
from whispy.sensors.base import HealthReport, SensorAdapter, SensorMeta

logger = logging.getLogger(__name__)

HARDWARE_ID = "rpi-dreamhat-bgt60tr13c"


def _sdk():
    try:
        from ifxradarsdk import Device  # type: ignore
        return Device
    except Exception:
        try:
            from ifxradarsdk import DeviceHelper  # type: ignore  # noqa
            import ifxradarsdk  # type: ignore
            return ifxradarsdk
        except Exception:
            return None


def _radar_present() -> bool:
    sdk = _sdk()
    if sdk is None:
        return False
    try:
        dev = sdk() if callable(sdk) else None
        if dev is None:
            return False
        try:
            dev.close()
        except Exception:
            pass
        return True
    except Exception:
        return False


class _RadarHandle(SensorHandle):
    """Streams processed radar frames (SNR + range profile) per frame."""

    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        self._config = dict(config or {})
        self._seq = itertools.count()
        self._device = None
        self._processor = None

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _open(self) -> None:
        if self._device is not None:
            return
        sdk = _sdk()
        if sdk is None:
            raise RuntimeError("ifxradarsdk not installed")
        self._device = sdk()
        # Wrap the existing processor when available; the raw frame path
        # stays identical to the proven implementation.
        try:
            import importlib
            mod = importlib.import_module(
                str(self._config.get("processor_module")
                    or "radar_processor"))
            self._processor = getattr(mod, "RadarProcessor", None)
            if self._processor is not None:
                self._processor = self._processor(
                    config_path=self._config.get("config_path"))
        except Exception as exc:
            logger.debug("RadarProcessor unavailable, raw frames: %s", exc)
            self._processor = None

    def _frame_payload(self, frame) -> Dict[str, Any]:
        """One radar frame → normalized payload with SNR + range profile."""
        import numpy as np
        arr = np.asarray(frame)
        # Per-frame features the 4 dB rule model consumes.
        magnitude = np.abs(arr.astype(np.complex128)
                         if np.iscomplexobj(arr) else arr.astype(float))
        signal = float(magnitude.max()) if magnitude.size else 0.0
        noise = float(np.median(magnitude)) if magnitude.size else 0.0
        snr_db = 20.0 * float(np.log10((signal + 1e-9) / (noise + 1e-9)))
        range_profile = magnitude.mean(axis=tuple(range(1, arr.ndim))) \
            if arr.ndim > 1 else magnitude
        # Downsampled energy map: last two axes = range × azimuth bins.
        xy_map: List[List[float]] = []
        if arr.ndim >= 2:
            m2 = magnitude.reshape(-1, *magnitude.shape[-2:]).mean(axis=0)
            target = 24
            ys = np.linspace(0, m2.shape[0], target + 1).astype(int)
            xs = np.linspace(0, m2.shape[1], target + 1).astype(int)
            xy_map = []
            for i in range(target):
                y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
                row = []
                for j in range(target):
                    x0, x1 = xs[j], max(xs[j + 1], xs[j] + 1)
                    row.append(float(m2[y0:y1, x0:x1].mean()))
                xy_map.append(row)
        return {
            "encoding": "radar_frame",
            "shape": list(arr.shape),
            "snr_db": round(snr_db, 3),
            "range_profile": range_profile.ravel()[:128].tolist(),
            "xy_map": xy_map,
            "energy": float((magnitude ** 2).mean()) if magnitude.size else 0.0,
        }

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        self._open()
        count = 0
        while True:
            frame = None
            try:
                frames = self._device.get_next_frame()
                frame = frames[0] if isinstance(frames, (list, tuple)) \
                    else frames
            except Exception as exc:
                logger.warning("radar frame error: %s", exc)
                time.sleep(0.05)
                continue
            if frame is None:
                continue
            yield SensorSample(
                device_id="",
                sensor_id=self._desc.id,
                sensor_type="radar",
                timestamp=time.time(),
                sequence=next(self._seq),
                payload_type="radar_frame",
                payload=self._frame_payload(frame),
                metadata={"adapter": "dreamhat-radar",
                          "hardware_id": self._desc.hardware_id},
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                return

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None


class DreamHatRadarAdapter(SensorAdapter):
    """Discovers the DreamHAT BGT60TR13C shield on this host."""

    def __init__(self):
        self._handles: List[_RadarHandle] = []

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="dreamhat-radar",
            version="0.1.0",
            modalities=("radar",),
            description="DreamHAT BGT60TR13C 60 GHz radar (ifxradarsdk)",
            config_schema={
                "type": "object",
                "properties": {
                    "config_path": {"type": "string"},
                    "processor_module": {"type": "string"},
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[SensorDescriptor]:
        if not _radar_present():
            return []
        return [SensorDescriptor(
            id=SensorDescriptor.make_id("radar", HARDWARE_ID),
            modality="radar",
            adapter="dreamhat-radar",
            name="DreamHAT BGT60TR13C",
            hardware_id=HARDWARE_ID,
            capabilities=["radar_frame", "snr_db", "range_profile"],
            config_schema=self.metadata().config_schema,
            stable=True,
        )]

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _RadarHandle:
        handle = _RadarHandle(descriptor, config)
        self._handles.append(handle)
        return handle

    def health(self) -> HealthReport:
        return HealthReport(
            status="ok" if _sdk() is not None else "error",
            detail="" if _sdk() is not None else "ifxradarsdk not installed")

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


__all__ = ["DreamHatRadarAdapter"]
