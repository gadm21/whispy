"""Sensor plugin interface — the "Works with ThothCraft" contract.

Any hardware becomes a Thoth node by shipping a ``SensorDriver`` in a
pip package that registers under the ``thothcraft.sensors`` entry-point
group::

    # pyproject.toml of a driver package
    [project.entry-points."thothcraft.sensors"]
    realsense = "thothcraft_sensor_realsense:RealSenseDriver"

``thothcraftd`` and ``thothcraft sensors list`` discover every installed
driver via :func:`installed_drivers`. ``thothcraft sensors test`` runs
the conformance suite in :func:`check_driver`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type

import numpy as np

ENTRY_POINT_GROUP = "thothcraft.sensors"


@dataclass
class SensorMeta:
    """Driver metadata — powers registry listings and config UIs."""

    name: str
    version: str = "1.0.0"
    modalities: tuple = ()                    # e.g. ("radar",), ("camera","depth")
    description: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema
    maintainer: Optional[str] = None


@dataclass
class SensorFrame:
    """One timestamped sample in standard units."""

    sensor_type: str                          # radar | csi | camera | env | ...
    timestamp_ns: int
    data: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, sensor_type: str, data: Any, **meta) -> "SensorFrame":
        return cls(sensor_type=sensor_type,
                   timestamp_ns=time.time_ns(),
                   data=np.asarray(data), meta=meta)


@dataclass
class HealthReport:
    status: str = "ok"                        # ok | degraded | error
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class SensorDriver(ABC):
    """Base class for ThothCraft sensor drivers.

    Lifecycle: ``discover()`` → ``open(config)`` → ``stream()`` →
    ``close()``. ``health()`` may be polled any time after ``open``.
    """

    @abstractmethod
    def metadata(self) -> SensorMeta:
        """Static driver metadata."""

    @abstractmethod
    def discover(self) -> List[Dict[str, Any]]:
        """Return attached devices this driver can handle (may be empty)."""

    @abstractmethod
    def open(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Acquire the hardware; ``config`` validated against config_schema."""

    @abstractmethod
    def stream(self) -> Iterator[SensorFrame]:
        """Yield frames until ``close()``; must be interruptible."""

    @abstractmethod
    def close(self) -> None:
        """Release the hardware; safe to call twice."""

    def health(self) -> HealthReport:
        return HealthReport()

    def calibrate(self) -> Dict[str, Any]:
        """Optional driver-specific calibration; default unsupported."""
        raise NotImplementedError(f"{type(self).__name__} has no calibration")


def installed_drivers() -> Dict[str, Type[SensorDriver]]:
    """Discover all drivers registered under the entry-point group."""
    from importlib.metadata import entry_points

    drivers: Dict[str, Type[SensorDriver]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
        except Exception:
            continue
        if isinstance(cls, type) and issubclass(cls, SensorDriver):
            drivers[ep.name] = cls
    return drivers


def check_driver(driver: SensorDriver, max_frames: int = 5,
                 timeout_s: float = 10.0) -> Dict[str, Any]:
    """Conformance check: discover → open → stream → health → close.

    Returns a report dict; ``report["passed"]`` is True when the driver
    satisfies the contract (valid frames, monotonic timestamps, clean
    shutdown). Used by ``thothcraft sensors test``.
    """
    report: Dict[str, Any] = {"driver": type(driver).__name__,
                              "checks": [], "passed": False}

    def check(name: str, ok: bool, detail: str = "") -> bool:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        return ok

    try:
        meta = driver.metadata()
        check("metadata", bool(meta.name and meta.modalities),
              f"{meta.name} {meta.version}")
    except Exception as exc:
        check("metadata", False, str(exc))
        return report

    try:
        found = driver.discover()
        check("discover", isinstance(found, list), f"{len(found)} device(s)")
    except Exception as exc:
        check("discover", False, str(exc))
        return report

    try:
        driver.open({})
        check("open", True)
    except Exception as exc:
        check("open", False, str(exc))
        return report

    frames, last_ts, ok, err = 0, -1, True, ""
    deadline = time.monotonic() + timeout_s
    try:
        for frame in driver.stream():
            if not isinstance(frame, SensorFrame) or frame.timestamp_ns <= 0:
                ok, err = False, "invalid SensorFrame"
                break
            if frame.timestamp_ns < last_ts:
                ok, err = False, "non-monotonic timestamps"
                break
            last_ts, frames = frame.timestamp_ns, frames + 1
            if frames >= max_frames or time.monotonic() > deadline:
                break
    except Exception as exc:
        ok, err = False, str(exc)
    check("stream", ok and frames > 0, err or f"{frames} frame(s)")

    try:
        health = driver.health()
        check("health", isinstance(health, HealthReport), health.status)
    except Exception as exc:
        check("health", False, str(exc))

    try:
        driver.close()
        check("close", True)
    except Exception as exc:
        check("close", False, str(exc))

    report["passed"] = all(c["ok"] for c in report["checks"])
    return report
