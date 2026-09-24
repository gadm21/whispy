"""Whispy sensor driver/adapter contracts.

A driver produces real :class:`~whispy.contracts.SensorSample` streams —
never availability booleans. Health is reported separately via
:meth:`SensorDriver.health` and ``Sensor.online``.

Two plugin interfaces live here:

- **SensorDriver** (legacy) — one driver ↔ one sensor:
  ``discover()`` → ``open(config)`` → ``stream()`` → ``close()``.
- **SensorAdapter** (current) — one adapter ↔ zero or more physical
  sensors: ``discover()`` → ``list[SensorDescriptor]`` then
  ``connect(descriptor, config)`` → ``SensorHandle``.

:class:`SensorDriverAdapter` wraps any legacy driver into the adapter
interface so existing drivers keep working during the migration.

Third-party plugins register under the ``whispy.sensors`` entry-point
group::

    [project.entry-points."whispy.sensors"]
    realsense = "whispy_sensor_realsense:RealSenseAdapter"
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from ..contracts import SensorDescriptor, SensorSample

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


class SensorAdapter(ABC):
    """New-style source adapter — discovers observation-source instances.

    Lifecycle::

        adapter.discover()          → list[SourceDescriptor] (0..n)
        adapter.connect(desc, cfg)  → SourceHandle (opened lazily)
        adapter.close()             → release adapter-level resources

    Unlike :class:`SensorDriver`, an adapter is *not* bound to a single
    source: a camera adapter discovers every attached camera, a Sense
    HAT adapter exposes imu/temperature/humidity/pressure descriptors.

    ``source_class`` is ``"sensor"`` for physical hardware. Context
    sources (battery, foreground app, calendar…) subclass
    :class:`ContextAdapter` instead — they are never fake sensors.
    """

    source_class: str = "sensor"

    @abstractmethod
    def metadata(self) -> SensorMeta:
        """Static adapter metadata."""

    @abstractmethod
    def discover(self) -> List[SensorDescriptor]:
        """Enumerate physical sensors this adapter can serve (may be [])."""

    @abstractmethod
    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None):
        """Open one discovered sensor; returns a SensorHandle."""

    def health(self) -> HealthReport:
        return HealthReport()

    def close(self) -> None:
        """Release adapter-level resources; safe to call twice."""


# Canonical names (§4): ``ObservationAdapter`` is the general contract;
# ``SensorAdapter`` remains the physical-sensor specialization.
ObservationAdapter = SensorAdapter


class ContextAdapter(SensorAdapter):
    """Adapter for non-physical observation sources (§4).

    Context sources produce :class:`~whispy.contracts.Observation`-shaped
    data (battery state, foreground application, location fixes, calendar
    entries) whose descriptors carry ``source_class="context"``.
    """

    source_class: str = "context"

    @staticmethod
    def _mark_context(descriptors: List[SensorDescriptor]
                      ) -> List[SensorDescriptor]:
        """Tag discovered descriptors as context sources."""
        for desc in descriptors:
            desc.source_class = "context"
        return descriptors


class _DriverSensorHandle:
    """SensorHandle over a legacy driver's open()/stream() pair.

    Defined here (not devices.base) to avoid an import cycle; satisfies
    the SensorHandle interface structurally.
    """

    def __init__(self, driver: "SensorDriver", descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._driver = driver
        self._descriptor = descriptor
        self._config = dict(config or {})
        self._opened = False

    @property
    def info(self):
        return self._descriptor.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._descriptor

    def _ensure_open(self) -> None:
        if not self._opened:
            self._driver.open(self._config)
            self._opened = True

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        self._ensure_open()
        count = 0
        for sample in self._driver.stream():
            if not sample.sensor_id or sample.sensor_id.endswith("-0") \
                    and sample.sensor_id != self._descriptor.id:
                sample.sensor_id = self._descriptor.id
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                return

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        if self._opened:
            try:
                self._driver.close()
            finally:
                self._opened = False


class SensorDriverAdapter(SensorAdapter):
    """Wraps a legacy :class:`SensorDriver` into the adapter interface.

    ``discover()`` maps the driver's ``discover()`` dicts × advertised
    modalities to :class:`SensorDescriptor` objects. When a discovered
    device reports a ``hardware_id`` the descriptor id is stable
    (``<modality>-<hash>``); otherwise it falls back to the historical
    ``<modality>-<index>`` form so existing inventory ids are preserved.
    """

    def __init__(self, driver: Union[SensorDriver, Type[SensorDriver]],
                 name: str = ""):
        self._driver = driver() if isinstance(driver, type) else driver
        self._name = name or self._safe_meta().name or \
            type(self._driver).__name__

    def _safe_meta(self) -> SensorMeta:
        try:
            return self._driver.metadata()
        except Exception:
            return SensorMeta(name=type(self._driver).__name__)

    @property
    def driver(self) -> SensorDriver:
        return self._driver

    def metadata(self) -> SensorMeta:
        return self._safe_meta()

    def discover(self) -> List[SensorDescriptor]:
        meta = self._safe_meta()
        modalities = list(meta.modalities) or [self._name]
        try:
            found = self._driver.discover()
        except Exception:
            found = []
        out: List[SensorDescriptor] = []
        for index, dev in enumerate(found or [{}]):
            dev = dict(dev or {})
            hw = str(dev.get("hardware_id") or dev.get("serial") or "")
            dev_id = str(dev.get("id") or "")
            for modality in modalities:
                if hw:
                    sid = SensorDescriptor.make_id(modality, hw)
                elif dev_id and len(modalities) == 1:
                    sid = dev_id                      # preserve legacy id
                else:
                    sid = SensorDescriptor.make_id(
                        modality, index=index)
                out.append(SensorDescriptor(
                    id=sid, modality=modality, adapter=self._name,
                    name=str(dev.get("name") or meta.name),
                    hardware_id=hw,
                    capabilities=list(meta.modalities),
                    config_schema=dict(meta.config_schema),
                    stable=bool(hw),
                    source_class=self.source_class,
                    metadata={"legacy_driver": True,
                              "discovered": dev},
                ))
        return out

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _DriverSensorHandle:
        return _DriverSensorHandle(self._driver, descriptor, config)

    def health(self) -> HealthReport:
        try:
            return self._driver.health()
        except Exception as exc:
            return HealthReport(status="error", detail=str(exc))

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass


def installed_drivers() -> Dict[str, Type[SensorDriver]]:
    """Discover legacy drivers registered under the entry-point group."""
    from ..plugins import registry

    drivers: Dict[str, Type[SensorDriver]] = {}
    for name, info in registry().discover_sensors().items():
        if info.available and info.kind == "driver" and info.cls is not None:
            drivers[name] = info.cls
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


def installed_adapters() -> Dict[str, SensorAdapter]:
    """Instantiate every sensor plugin as a :class:`SensorAdapter`.

    Entry points that export a legacy ``SensorDriver`` are wrapped in
    :class:`SensorDriverAdapter`; plugins that fail to load are skipped
    (their error is visible via ``PluginRegistry.discover_sensors()``).
    """
    from ..plugins import registry

    out: Dict[str, SensorAdapter] = {}
    for name, info in registry().discover_sensors().items():
        if not info.available or info.cls is None:
            continue
        try:
            if info.kind == "adapter":
                out[name] = info.cls()
            elif info.kind == "driver":
                out[name] = SensorDriverAdapter(info.cls, name=name)
        except Exception:
            continue
    return out


def all_adapters() -> Dict[str, SensorAdapter]:
    """Built-in drivers (wrapped) merged with entry-point adapters."""
    out: Dict[str, SensorAdapter] = {
        name: SensorDriverAdapter(cls, name=name)
        for name, cls in builtin_drivers().items()
    }
    out.update(installed_adapters())
    return out


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
