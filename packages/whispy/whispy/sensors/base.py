"""Whispy sensor driver contract.

A driver produces real :class:`~whispy.contracts.SensorSample` streams —
never availability booleans. Health is reported separately via
:meth:`SensorDriver.health` and ``Sensor.online``.

Third-party drivers register under the ``whispy.sensors`` entry-point
group::

    [project.entry-points."whispy.sensors"]
    realsense = "whispy_sensor_realsense:RealSenseDriver"
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type

from ..contracts import SensorSample

ENTRY_POINT_GROUP = "whispy.sensors"


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
class HealthReport:
    status: str = "ok"                        # ok | degraded | error
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class SensorDriver(ABC):
    """Base class for Whispy sensor drivers.

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
    def stream(self) -> Iterator[SensorSample]:
        """Yield real measurement samples until ``close()``; interruptible."""

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


def builtin_drivers() -> Dict[str, Type[SensorDriver]]:
    """Drivers shipped inside whispy itself and auto-detected on ``local()``.

    ``FixtureDriver`` is intentionally excluded — recorded fixtures are
    instantiated explicitly in tests, never auto-discovered as hardware.
    """
    from .system.telemetry import SystemTelemetryDriver

    return {
        "system": SystemTelemetryDriver,
    }


def all_drivers() -> Dict[str, Type[SensorDriver]]:
    """Built-in drivers merged with entry-point drivers (entry points win)."""
    drivers = builtin_drivers()
    drivers.update(installed_drivers())
    return drivers


def check_driver(driver: SensorDriver, max_samples: int = 5,
                 timeout_s: float = 10.0) -> Dict[str, Any]:
    """Conformance check: discover → open → stream → health → close.

    Returns a report dict; ``report["passed"]`` is True when the driver
    satisfies the contract (valid samples, monotonic timestamps, clean
    shutdown).
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

    samples, last_ts, last_seq, ok, err = 0, -1.0, -1, True, ""
    deadline = time.monotonic() + timeout_s
    try:
        for sample in driver.stream():
            if not isinstance(sample, SensorSample) or sample.timestamp <= 0:
                ok, err = False, "invalid SensorSample"
                break
            if sample.timestamp < last_ts:
                ok, err = False, "non-monotonic timestamps"
                break
            if sample.sequence < last_seq:
                ok, err = False, "non-monotonic sequence"
                break
            if sample.payload is None:
                ok, err = False, "empty payload — availability is not a measurement"
                break
            last_ts, last_seq, samples = sample.timestamp, sample.sequence, samples + 1
            if samples >= max_samples or time.monotonic() > deadline:
                break
    except Exception as exc:
        ok, err = False, str(exc)
    check("stream", ok and samples > 0, err or f"{samples} sample(s)")

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
