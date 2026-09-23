"""Deterministic recorded-fixture driver.

Architecture requirement (§30 exit gate): "unavailable physical sensors
have deterministic recorded fixtures." A ``FixtureDriver`` replays a
recorded sample sequence for any modality so tests and development work
without hardware.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Iterator, List, Optional

from ..contracts import SensorSample
from .base import HealthReport, SensorDriver, SensorMeta


class FixtureDriver(SensorDriver):
    """Replays a deterministic sample sequence.

    Config::

        {
            "sensor_type": "radar",
            "sensor_id": "radar-0",
            "device_id": "fixture-device",
            "sample_rate": 10.0,
            "payloads": [[1, 2, 3], [4, 5, 6]],   # cycled forever
            "realtime": false                     # sleep between samples
        }
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._opened = False
        self._seq = itertools.count()

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="fixture",
            version="1.0.0",
            modalities=("radar", "csi", "camera", "imu", "env", "system"),
            description="Deterministic recorded-fixture sensor",
            config_schema={
                "type": "object",
                "properties": {
                    "sensor_type": {"type": "string"},
                    "payloads": {"type": "array"},
                    "sample_rate": {"type": "number"},
                    "realtime": {"type": "boolean"},
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[Dict[str, Any]]:
        return [{"id": "fixture-0", "kind": "recorded-fixture"}]

    def open(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = dict(config or {})
        self._opened = True
        self._seq = itertools.count()

    def stream(self) -> Iterator[SensorSample]:
        if not self._opened:
            raise RuntimeError("FixtureDriver.stream() before open()")
        sensor_type = self._config.get("sensor_type", "fixture")
        sensor_id = self._config.get("sensor_id", f"{sensor_type}-0")
        device_id = self._config.get("device_id", "fixture-device")
        rate = float(self._config.get("sample_rate") or 10.0)
        realtime = bool(self._config.get("realtime"))
        payloads = self._config.get("payloads") or [[0.0]]
        period = 1.0 / rate if rate > 0 else 0.0
        for payload in itertools.cycle(payloads):
            if not self._opened:
                return
            seq = next(self._seq)
            yield SensorSample(
                device_id=device_id,
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                timestamp=time.time(),
                sequence=seq,
                payload_type="fixture",
                payload=payload,
                sample_rate=rate,
                metadata={"fixture": True},
            )
            if realtime and period > 0:
                time.sleep(period)

    def health(self) -> HealthReport:
        return HealthReport(status="ok" if self._opened else "error",
                            detail="fixture driver")

    def close(self) -> None:
        self._opened = False
