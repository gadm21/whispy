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

    _APPS_INTERVAL_S = 15.0
    _APPS_MAX = 12

    def _read_apps(self, psutil: Any) -> List[Dict[str, Any]]:
        """Top processes by CPU — process NAMES only (no cmdline, which
        can carry args/secrets). This is how 'is Windsurf open' reaches
        evidence: {name, cpu, count} rows, refreshed slowly."""
        now = time.time()
        if (now - getattr(self, "_apps_at", 0.0)) < self._APPS_INTERVAL_S:
            return getattr(self, "_apps_cache", [])  # type: ignore
        try:
            # cpu_times deltas between polls — a fresh process_iter
            # always reports cpu_percent 0.0, so keep previous counters
            # and diff them against the 15 s refresh window.
            prev: Dict[int, float] = getattr(self, "_apps_prev", {})
            prev_at = float(getattr(self, "_apps_prev_at", 0.0))
            dt = max(now - prev_at, 1.0)
            curr: Dict[int, float] = {}
            rows: Dict[str, Dict[str, Any]] = {}
            for p in psutil.process_iter(["name"]):
                try:
                    name = (p.info.get("name") or "").lower()
                except Exception:
                    continue
                if not name or name in ("system idle process", "system",
                                        "registry", "memory compression"):
                    continue
                row = rows.setdefault(name, {"name": name,
                                           "cpu": 0.0, "count": 0})
                row["count"] += 1
                try:
                    t = p.cpu_times()
                    total = float(t.user) + float(t.system)
                    pid_key = p.pid
                    curr[pid_key] = total
                    # CPU% = busy_seconds / wall_seconds * 100 (can
                    # exceed 100 on multi-core — cap later).
                    row["cpu"] += max(0.0, total - prev.get(
                        pid_key, total)) / dt * 100.0
                except Exception:
                    pass
            apps = sorted(rows.values(), key=lambda r: -r["cpu"])
            out = [{"name": r["name"], "cpu": round(r["cpu"], 1),
                    "count": r["count"]}
                   for r in apps[: self._APPS_MAX]]
            self._apps_prev = curr
            self._apps_prev_at = now
        except Exception:
            out = getattr(self, "_apps_cache", [])  # type: ignore
        self._apps_at = now
        self._apps_cache = out  # type: ignore
        return out

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
            try:
                apps = self._read_apps(psutil)
                out["apps"] = apps
                out["apps_count"] = len(apps)
                out["top_app"] = apps[0]["name"] if apps else None
                out["top_app_cpu"] = apps[0]["cpu"] if apps else None
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
