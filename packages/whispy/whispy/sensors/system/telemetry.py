"""Host system telemetry driver — CPU, memory, battery.

Works on any computer (Windows/macOS/Linux/Pi). Uses ``psutil`` when
installed; otherwise emits a minimal ``os``-based load sample so the
contract still produces real measurements.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterator, List, Optional

from ...contracts import SensorSample
from ..base import HealthReport, SensorDriver, SensorMeta


class SystemTelemetryDriver(SensorDriver):
    """Streams host telemetry samples (cpu_percent, mem_percent, ...)."""

    def __init__(self) -> None:
        self._opened = False
        self._seq = 0
        self._rate = 1.0

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="system-telemetry",
            version="1.0.0",
            modalities=("system",),
            description="Host CPU/memory/battery telemetry",
            config_schema={
                "type": "object",
                "properties": {"sample_rate": {"type": "number"}},
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[Dict[str, Any]]:
        return [{"id": "system-0", "kind": "host-telemetry"}]

    def open(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self._rate = float(config.get("sample_rate") or 1.0)
        self._opened = True
        self._seq = 0

    def _read(self) -> Dict[str, Any]:
        try:
            import psutil  # type: ignore
            out: Dict[str, Any] = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_percent": psutil.virtual_memory().percent,
                "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            }
            try:
                batt = psutil.sensors_battery()
                if batt is not None:
                    out["battery_percent"] = batt.percent
                    out["battery_plugged"] = batt.power_plugged
            except Exception:
                pass
            return out
        except ImportError:
            load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
            return {"cpu_percent": None, "mem_percent": None, "load_avg": [load]}

    def stream(self) -> Iterator[SensorSample]:
        if not self._opened:
            raise RuntimeError("SystemTelemetryDriver.stream() before open()")
        period = 1.0 / self._rate if self._rate > 0 else 1.0
        while self._opened:
            yield SensorSample(
                device_id="local",
                sensor_id="system-0",
                sensor_type="system",
                timestamp=time.time(),
                sequence=self._seq,
                payload_type="json",
                payload=self._read(),
                sample_rate=self._rate,
                units={"cpu_percent": "%", "mem_percent": "%",
                       "battery_percent": "%"},
            )
            self._seq += 1
            time.sleep(period)

    def health(self) -> HealthReport:
        return HealthReport(status="ok" if self._opened else "error")

    def close(self) -> None:
        self._opened = False
