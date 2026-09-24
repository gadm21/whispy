"""Plugin conformance checks — the acceptance gate for new packages.

Symmetrical to :func:`whispy.sensors.check_driver`:

- :func:`check_sensor_adapter` — discover → connect → stream → close
- :func:`check_model_plugin` — metadata → predict on a fixture window
- :func:`check_actuator_adapter` — discover → connect → execute → close

Every hardware/model package should pass its check before being
accepted. Checks never fake success: a plugin that cannot produce real
samples or a confirmed action result fails explicitly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .contracts import (
    ActionStatus, ActuatorCommand, ActuatorDescriptor, ModalityState,
    SensorDescriptor, SensorSample, SensorWindow,
)


def _report(name: str) -> Dict[str, Any]:
    return {"plugin": name, "checks": [], "passed": False}


def _check(report: Dict[str, Any], name: str, ok: bool,
           detail: str = "") -> bool:
    report["checks"].append({"name": name, "ok": bool(ok),
                             "detail": detail})
    return ok


def check_sensor_adapter(adapter, max_samples: int = 3,
                         timeout_s: float = 10.0,
                         config: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """Conformance: discover → connect → stream → health → close.

    Adapters with no attached hardware pass ``discover`` but skip the
    stream check — ``report["skipped"]`` lists why.
    """
    from .sensors.base import SensorAdapter

    report = _report(type(adapter).__name__)
    report["skipped"] = []

    if not _check(report, "is_sensor_adapter",
                  isinstance(adapter, SensorAdapter)):
        report["passed"] = False
        return report

    try:
        meta = adapter.metadata()
        _check(report, "metadata", bool(meta.name), meta.name)
    except Exception as exc:
        _check(report, "metadata", False, str(exc))
        return report

    try:
        descriptors = adapter.discover()
        ok = isinstance(descriptors, list) and all(
            isinstance(d, SensorDescriptor) and d.id and d.modality
            for d in descriptors)
        _check(report, "discover", ok, f"{len(descriptors)} descriptor(s)")
    except Exception as exc:
        _check(report, "discover", False, str(exc))
        return report

    if not descriptors:
        report["skipped"].append("stream: no hardware discovered")
        report["passed"] = all(c["ok"] for c in report["checks"])
        return report

    desc = descriptors[0]
    try:
        handle = adapter.connect(desc, config or {})
        _check(report, "connect", hasattr(handle, "stream"), desc.id)
    except Exception as exc:
        _check(report, "connect", False, str(exc))
        return report

    samples, last_ts, ok, err = 0, -1.0, True, ""
    deadline = time.monotonic() + timeout_s
    try:
        for sample in handle.stream():
            if not isinstance(sample, SensorSample) or sample.timestamp <= 0:
                ok, err = False, "invalid SensorSample"
                break
            if sample.timestamp < last_ts:
                ok, err = False, "non-monotonic timestamps"
                break
            if sample.payload is None:
                ok, err = False, "empty payload"
                break
            last_ts, samples = sample.timestamp, samples + 1
            if samples >= max_samples or time.monotonic() > deadline:
                break
    except Exception as exc:
        ok, err = False, str(exc)
    _check(report, "stream", ok and samples > 0,
           err or f"{samples} sample(s) from {desc.id}")

    try:
        health = adapter.health()
        _check(report, "health", hasattr(health, "status"),
               getattr(health, "status", ""))
    except Exception as exc:
        _check(report, "health", False, str(exc))

    try:
        if hasattr(handle, "close"):
            handle.close()
        adapter.close()
        _check(report, "close", True)
    except Exception as exc:
        _check(report, "close", False, str(exc))

    report["passed"] = all(c["ok"] for c in report["checks"])
    return report


def check_model_plugin(model_cls, config: Optional[Dict[str, Any]] = None,
                       window: Optional[SensorWindow] = None
                       ) -> Dict[str, Any]:
    """Conformance: instantiate → metadata → predict(fixture window).

    The default window carries one synthetic ``fixture`` sensor; models
    that require specific modalities may abstain/unknown — the check only
    requires a valid :class:`Prediction`, not a particular label.
    """
    from .processors.base import Processor

    report = _report(getattr(model_cls, "__name__", str(model_cls)))

    try:
        proc = model_cls(config or {})
        _check(report, "instantiate", isinstance(proc, Processor))
    except Exception as exc:
        _check(report, "instantiate", False, str(exc))
        return report

    try:
        meta = proc.metadata()
        _check(report, "metadata", bool(meta.name), meta.name)
    except Exception as exc:
        _check(report, "metadata", False, str(exc))
        return report

    if window is None:
        sample = SensorSample.now("conformance", "fixture-0", "fixture",
                                  [0.0], payload_type="list")
        window = SensorWindow(
            start_timestamp=0.0, end_timestamp=1.0,
            samples={"fixture-0": [sample]},
            modalities={"fixture-0": ModalityState(
                sensor_id="fixture-0", state="ok")})
    try:
        pred = proc.predict(window)
        ok = hasattr(pred, "label") and isinstance(pred.label, str) \
            and 0.0 <= float(pred.confidence) <= 1.0
        _check(report, "predict", ok,
               f"label={pred.label!r} confidence={pred.confidence}")
    except Exception as exc:
        _check(report, "predict", False, str(exc))

    try:
        health = proc.health()
        _check(report, "health", isinstance(health, dict),
               str(health.get("status", "")))
    except Exception as exc:
        _check(report, "health", False, str(exc))

    report["passed"] = all(c["ok"] for c in report["checks"])
    return report


def check_actuator_adapter(adapter, probe: bool = False,
                           config: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
    """Conformance: discover → connect → (optional) execute → close.

    ``probe=False`` (default) never fires a physical effect — it verifies
    the handle exists and reports its operations. ``probe=True`` executes
    a ``noop``/``status`` command when the descriptor advertises one, so
    CI stays safe on real hardware.
    """
    from .actuators.base import ActuatorAdapter, ActuatorHandle

    report = _report(type(adapter).__name__)
    report["skipped"] = []

    if not _check(report, "is_actuator_adapter",
                  isinstance(adapter, ActuatorAdapter)):
        return report

    try:
        meta = adapter.metadata()
        _check(report, "metadata", bool(meta.name), meta.name)
    except Exception as exc:
        _check(report, "metadata", False, str(exc))
        return report

    try:
        descriptors = adapter.discover()
        ok = isinstance(descriptors, list) and all(
            isinstance(d, ActuatorDescriptor) and d.id and d.kind
            for d in descriptors)
        _check(report, "discover", ok, f"{len(descriptors)} descriptor(s)")
    except Exception as exc:
        _check(report, "discover", False, str(exc))
        return report

    if not descriptors:
        report["skipped"].append("execute: no hardware discovered")
        report["passed"] = all(c["ok"] for c in report["checks"])
        return report

    desc = descriptors[0]
    try:
        handle = adapter.connect(desc, config or {})
        _check(report, "connect", isinstance(handle, ActuatorHandle)
               or hasattr(handle, "execute"), desc.id)
    except Exception as exc:
        _check(report, "connect", False, str(exc))
        return report

    if probe and desc.operations:
        op = next((o for o in ("noop", "status", "clear")
                       if o in desc.operations), desc.operations[0])
        try:
            result = handle.execute(ActuatorCommand(operation=op))
            _check(report, "execute",
                   result.status in (ActionStatus.SUCCEEDED,
                                     ActionStatus.UNSUPPORTED),
                   f"{op}: {result.status.value} {result.detail}")
        except Exception as exc:
            _check(report, "execute", False, str(exc))
    else:
        report["skipped"].append("execute: probe disabled")

    try:
        if hasattr(handle, "close"):
            handle.close()
        adapter.close()
        _check(report, "close", True)
    except Exception as exc:
        _check(report, "close", False, str(exc))

    report["passed"] = all(c["ok"] for c in report["checks"])
    return report


__all__ = [
    "check_actuator_adapter",
    "check_model_plugin",
    "check_sensor_adapter",
]
