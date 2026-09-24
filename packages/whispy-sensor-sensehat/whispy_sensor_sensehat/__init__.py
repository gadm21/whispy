"""Sense HAT sensor adapter — one HAT, five logical sensors.

A single Sense HAT exposes multiple physical/logical streams::

    imu          orientation + accel/gyro vectors
    temperature  °C
    humidity     %RH
    pressure     mbar
    joystick     direction events

Each is a separate :class:`SensorDescriptor` sharing the HAT's stable
hardware id (``rpi-sensehat``), so inventory shows physical instances::

    imu-a1b2  temperature-a1b2  humidity-a1b2  pressure-a1b2  joystick-a1b2

The Pi 3B+ stays lightweight: no torch, no whisper, no OpenCV.
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

HARDWARE_ID = "rpi-sensehat"

_MODALITIES = ("imu", "temperature", "humidity", "pressure", "joystick")


def _sense_hat():
    try:
        from sense_hat import SenseHat  # type: ignore
        return SenseHat()
    except Exception:
        try:
            from sense_emu import SenseHat  # type: ignore
            return SenseHat()
        except Exception:
            return None


class _SenseHatHandle(SensorHandle):
    """Streams one modality from the shared Sense HAT."""

    def __init__(self, hat, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._hat = hat
        self._desc = descriptor
        self._config = dict(config or {})
        self._seq = itertools.count()

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _read(self) -> Any:
        hat = self._hat
        modality = self._desc.modality
        if modality == "imu":
            accel = hat.get_accelerometer_raw()
            gyro = hat.get_gyroscope_raw()
            orient = hat.get_orientation_degrees()
            return {
                "accel": {k: round(float(accel[k]), 5) for k in ("x", "y", "z")},
                "gyro": {k: round(float(gyro[k]), 5) for k in ("x", "y", "z")},
                "orientation_deg": {k: round(float(orient[k]), 2)
                                    for k in ("pitch", "roll", "yaw")},
            }
        if modality == "temperature":
            return float(hat.get_temperature())
        if modality == "humidity":
            return float(hat.get_humidity())
        if modality == "pressure":
            return float(hat.get_pressure())
        if modality == "joystick":
            events = hat.stick.get_events()
            return [{"direction": e.direction, "action": e.action}
                    for e in events]
        raise KeyError(f"unknown sensehat modality {modality!r}")

    _UNITS = {
        "temperature": {"value": "°C"},
        "humidity": {"value": "%RH"},
        "pressure": {"value": "mbar"},
        "imu": {"accel": "g", "gyro": "rad/s", "orientation_deg": "deg"},
    }

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        rate = float(self._config.get("sample_rate") or 2.0)
        period = 1.0 / rate if rate > 0 else 0.1
        count = 0
        while True:
            payload = self._read()
            yield SensorSample(
                device_id="",
                sensor_id=self._desc.id,
                sensor_type=self._desc.modality,
                timestamp=time.time(),
                sequence=next(self._seq),
                payload_type="json",
                payload=payload,
                sample_rate=rate,
                units=self._UNITS.get(self._desc.modality, {}),
                metadata={"adapter": "sensehat",
                          "hardware_id": self._desc.hardware_id},
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                return
            time.sleep(period)

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        pass


class SenseHatSensorAdapter(SensorAdapter):
    """Discovers a Sense HAT and exposes its five sensor streams."""

    def __init__(self):
        self._hat = None
        self._hat_tried = False
        self._handles: List[_SenseHatHandle] = []

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="sensehat",
            version="0.1.0",
            modalities=_MODALITIES,
            description="Raspberry Pi Sense HAT (IMU/env/joystick)",
            config_schema={
                "type": "object",
                "properties": {"sample_rate": {"type": "number"}},
            },
            maintainer="thothcraft",
        )

    def _get_hat(self):
        if not self._hat_tried:
            self._hat = _sense_hat()
            self._hat_tried = True
        return self._hat

    def discover(self) -> List[SensorDescriptor]:
        if self._get_hat() is None:
            return []
        return [SensorDescriptor(
            id=SensorDescriptor.make_id(modality, HARDWARE_ID),
            modality=modality,
            adapter="sensehat",
            name=f"Sense HAT {modality}",
            hardware_id=HARDWARE_ID,
            capabilities=[modality],
            config_schema=self.metadata().config_schema,
            stable=True,
        ) for modality in _MODALITIES]

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _SenseHatHandle:
        hat = self._get_hat()
        if hat is None:
            raise RuntimeError("Sense HAT not available")
        handle = _SenseHatHandle(hat, descriptor, config)
        self._handles.append(handle)
        return handle

    def health(self) -> HealthReport:
        return HealthReport(
            status="ok" if self._get_hat() is not None else "error",
            detail="" if self._get_hat() is not None
            else "sense-hat/sense-emu not installed")

    def close(self) -> None:
        self._handles.clear()


__all__ = ["SenseHatSensorAdapter"]
