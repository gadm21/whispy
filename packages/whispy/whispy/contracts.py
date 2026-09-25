"""ThothCraft v1 shared contracts.

Single source of truth for the data structures exchanged between Whispy,
Thoth, Brain, thothHUB, and the Flutter mobile app (Architecture v3.0,
Part III — Core Contracts).

Every structure here is plain data: ``to_dict()``/``from_dict()`` round-trip
through JSON-safe types so the same contract can be validated on the edge,
in Brain, and in client SDKs.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionStatus(str, Enum):
    """Actuator execution result states (§7.6 / §17)."""

    QUEUED = "queued"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    EXPIRED = "expired"                      # never executed — outlived expiry

    @property
    def terminal(self) -> bool:
        return self in (
            ActionStatus.SUCCEEDED, ActionStatus.FAILED,
            ActionStatus.UNSUPPORTED, ActionStatus.EXPIRED)


class DeploymentState(str, Enum):
    """Deployment state machine (§19).

    queued → received → validated → installed → acknowledged → active
    Any stage may instead transition to ``failed``.
    """

    QUEUED = "queued"
    RECEIVED = "received"
    VALIDATED = "validated"
    INSTALLED = "installed"
    ACKNOWLEDGED = "acknowledged"
    ACTIVE = "active"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (DeploymentState.ACTIVE, DeploymentState.FAILED)


# Legal forward transitions for the deployment state machine.
DEPLOYMENT_TRANSITIONS: Dict[DeploymentState, frozenset] = {
    DeploymentState.QUEUED: frozenset({DeploymentState.RECEIVED, DeploymentState.FAILED}),
    DeploymentState.RECEIVED: frozenset({DeploymentState.VALIDATED, DeploymentState.FAILED}),
    DeploymentState.VALIDATED: frozenset({DeploymentState.INSTALLED, DeploymentState.FAILED}),
    DeploymentState.INSTALLED: frozenset({DeploymentState.ACKNOWLEDGED, DeploymentState.FAILED}),
    DeploymentState.ACKNOWLEDGED: frozenset({DeploymentState.ACTIVE, DeploymentState.FAILED}),
    DeploymentState.ACTIVE: frozenset({DeploymentState.FAILED}),
    DeploymentState.FAILED: frozenset(),
}

PROCESSOR_TYPES = ("rule", "torchscript", "fusion")
MODEL_MANIFEST_FORMAT = "whispy-model/v1"
MODEL_MANIFEST_FORMAT_V2 = "whispy-model/v2"
# ``thoth-model/v1`` is the pre-rename name for the same manifest schema.
# It is accepted on ingest and normalized to ``MODEL_MANIFEST_FORMAT`` so
# packages produced before the rename keep working; new packages should
# always emit ``whispy-model/v2`` (or ``whispy-model/v1``).
LEGACY_MODEL_MANIFEST_FORMAT = "thoth-model/v1"
SUPPORTED_MANIFEST_FORMATS = frozenset({
    MODEL_MANIFEST_FORMAT,
    MODEL_MANIFEST_FORMAT_V2,
    LEGACY_MODEL_MANIFEST_FORMAT,
})

# Canonical minute container schema (§9). ``thoth-minute/v1`` replaces the
# legacy chunk-oriented capture manifest as the domain temporal/storage
# abstraction. Timestamp is authoritative; a minute may hold heterogeneous
# sources and is never assumed to contain 60 samples.
MINUTE_MANIFEST_FORMAT = "thoth-minute/v1"

# Source classes (§4): physical sensors vs. logical/context sources.
SOURCE_CLASSES = frozenset({"sensor", "context"})

# Model lifecycle types (§7).
MODEL_LIFECYCLES = frozenset({"streaming", "windowed", "batch"})

# Execution classes (§7/§14). Vendor-neutral by design.
EXECUTION_CLASSES = frozenset({
    "local", "trusted_edge", "managed_cloud", "external_provider",
})


# ---------------------------------------------------------------------------
# Device / Sensor inventory
# ---------------------------------------------------------------------------

@dataclass
class Sensor:
    """Sensor contract (§13)."""

    id: str
    type: str                                  # radar | csi | camera | imu | env | system
    driver: str = ""
    driver_version: str = ""
    sample_rate: Optional[float] = None
    units: Dict[str, str] = field(default_factory=dict)
    online: bool = True
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sensor":
        return cls(
            id=str(data.get("id") or data.get("sensor_id") or ""),
            type=str(data.get("type") or data.get("sensor_type") or ""),
            driver=str(data.get("driver") or ""),
            driver_version=str(data.get("driver_version") or ""),
            sample_rate=data.get("sample_rate"),
            units=dict(data.get("units") or {}),
            online=bool(data.get("online", True)),
            capabilities=list(data.get("capabilities") or []),
            metadata=dict(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Hardware descriptors (adapter discovery)
# ---------------------------------------------------------------------------

def _short_hash(text: str, length: int = 4) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


@dataclass
class SensorDescriptor:
    """A physical sensor instance discovered by a :class:`SensorAdapter`.

    ``id`` is stable across reboots whenever the adapter can supply a
    persistent ``hardware_id`` (USB PnP id, serial, MAC …): it is derived
    as ``<modality>-<sha1(hardware_id)[:4]>``. Without hardware identity
    the id falls back to ``<modality>-<index>`` and ``stable`` is False.
    """

    id: str
    modality: str                              # camera | microphone | radar | …
    adapter: str = ""                          # adapter/plugin name
    name: str = ""                             # human name ("Integrated Camera")
    hardware_id: str = ""                      # persistent hardware identity
    capabilities: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    stable: bool = False
    source_class: str = "sensor"               # sensor | context (§4)
    health: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """Canonical ``type`` view of ``modality`` (schema field name)."""
        return self.modality

    @staticmethod
    def make_id(modality: str, hardware_id: str = "",
                index: int = 0) -> str:
        if hardware_id:
            return f"{modality}-{_short_hash(hardware_id)}"
        return f"{modality}-{index}"

    def to_sensor(self, online: bool = True) -> "Sensor":
        """Project this descriptor to the §13 Sensor inventory contract."""
        return Sensor(
            id=self.id, type=self.modality, driver=self.adapter,
            online=online, capabilities=list(self.capabilities),
            metadata={"hardware_id": self.hardware_id, "name": self.name,
                      "stable": self.stable, **self.metadata})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SensorDescriptor":
        return cls(
            id=str(data.get("id") or ""),
            modality=str(data.get("modality") or data.get("type") or ""),
            adapter=str(data.get("adapter") or data.get("driver") or ""),
            name=str(data.get("name") or ""),
            hardware_id=str(data.get("hardware_id") or ""),
            capabilities=list(data.get("capabilities") or []),
            config_schema=dict(data.get("config_schema") or {}),
            stable=bool(data.get("stable", False)),
            source_class=str(data.get("source_class") or "sensor"),
            health=dict(data.get("health") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


# ``SourceDescriptor`` is the canonical name (§4); ``SensorDescriptor`` is
# retained as the backward-compatible alias — same class, same wire shape.
SourceDescriptor = SensorDescriptor


@dataclass
class ActuatorDescriptor:
    """A physical actuator instance discovered by an ActuatorAdapter."""

    id: str
    kind: str                                  # speaker | matrix | gpio | …
    adapter: str = ""
    name: str = ""
    hardware_id: str = ""
    operations: List[str] = field(default_factory=list)   # speak|play|show|…
    capabilities: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    stable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(kind: str, hardware_id: str = "", index: int = 0) -> str:
        if hardware_id:
            return f"{kind}-{_short_hash(hardware_id)}"
        return f"{kind}-{index}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActuatorDescriptor":
        return cls(
            id=str(data.get("id") or ""),
            kind=str(data.get("kind") or data.get("type") or ""),
            adapter=str(data.get("adapter") or ""),
            name=str(data.get("name") or ""),
            hardware_id=str(data.get("hardware_id") or ""),
            operations=list(data.get("operations") or []),
            capabilities=list(data.get("capabilities") or []),
            config_schema=dict(data.get("config_schema") or {}),
            stable=bool(data.get("stable", False)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ActuatorCommand:
    """A single operation request sent to an actuator handle.

    Wire-safe: serializes to ``{"operation": ..., "params": {...}}`` for
    the LAN ``POST /api/actuators/{id}/actions`` endpoint.
    """

    operation: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return {"operation": self.operation, "params": self.params,
                "timeout_seconds": self.timeout_seconds}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActuatorCommand":
        if isinstance(data, ActuatorCommand):
            return data
        return cls(
            operation=str(data.get("operation") or data.get("op") or ""),
            params=dict(data.get("params") or {}),
            timeout_seconds=float(data.get("timeout_seconds") or 30.0),
        )


@dataclass
class ModelBinding:
    """Binds one named model input to a concrete sensor source.

    ``source`` selects where the SensorHandle comes from:
    ``local`` (this device), ``lan`` (another Thoth node), ``brain``
    (via Brain relay), or ``fixture``/``replay`` (recorded data).
    """

    input_name: str
    sensor_id: str
    source: str = "local"                      # local | lan | brain | fixture
    source_device: str = ""                    # device id / host for remote
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelBinding":
        return cls(
            input_name=str(data.get("input_name") or data.get("name") or ""),
            sensor_id=str(data.get("sensor_id") or data.get("sensor") or ""),
            source=str(data.get("source") or "local"),
            source_device=str(data.get("source_device") or ""),
            config=dict(data.get("config") or {}),
        )


@dataclass
class Device:
    """Device contract (§12). LAN IPs are diagnostics, never identity."""

    id: str
    stable_uuid: str
    name: str
    owner: Optional[str] = None
    platform: str = ""
    architecture: str = ""
    software_version: str = ""
    whispy_version: str = ""
    online: bool = False
    last_seen: Optional[float] = None
    capabilities: List[str] = field(default_factory=list)
    sensors: List[Sensor] = field(default_factory=list)
    health: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["sensors"] = [s.to_dict() if isinstance(s, Sensor) else s
                          for s in self.sensors]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Device":
        return cls(
            id=str(data.get("id") or data.get("device_id") or ""),
            stable_uuid=str(data.get("stable_uuid") or data.get("device_uuid") or ""),
            name=str(data.get("name") or data.get("device_name") or ""),
            owner=data.get("owner") or data.get("account"),
            platform=str(data.get("platform") or ""),
            architecture=str(data.get("architecture") or ""),
            software_version=str(data.get("software_version") or ""),
            whispy_version=str(data.get("whispy_version") or ""),
            online=bool(data.get("online", False)),
            last_seen=data.get("last_seen"),
            capabilities=list(data.get("capabilities") or []),
            sensors=[Sensor.from_dict(s) for s in (data.get("sensors") or [])],
            health=dict(data.get("health") or {}),
        )


# ---------------------------------------------------------------------------
# Sensor samples and windows
# ---------------------------------------------------------------------------

@dataclass
class SensorSample:
    """One timestamped measurement (§14).

    ``payload`` carries the real measurement — never an availability
    boolean. Health/availability lives in ``metadata``/``Sensor.online``.
    """

    device_id: str
    sensor_id: str
    sensor_type: str
    timestamp: float                           # seconds, monotonic-source or epoch
    sequence: int
    payload_type: str                          # e.g. "ndarray:float32", "jpeg", "scalar"
    payload: Any
    sample_rate: Optional[float] = None
    units: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, device_id: str, sensor_id: str, sensor_type: str,
            payload: Any, *, sequence: int = 0, payload_type: str = "auto",
            sample_rate: Optional[float] = None,
            units: Optional[Dict[str, str]] = None,
            **metadata: Any) -> "SensorSample":
        return cls(
            device_id=device_id, sensor_id=sensor_id, sensor_type=sensor_type,
            timestamp=time.time(), sequence=sequence,
            payload_type=payload_type, payload=payload,
            sample_rate=sample_rate, units=dict(units or {}),
            metadata=dict(metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = self.payload
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        return {
            "device_id": self.device_id,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "payload_type": self.payload_type,
            "payload": payload,
            "sample_rate": self.sample_rate,
            "units": self.units,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SensorSample":
        return cls(
            device_id=str(data.get("device_id") or ""),
            sensor_id=str(data.get("sensor_id") or ""),
            sensor_type=str(data.get("sensor_type") or ""),
            timestamp=float(data.get("timestamp") or 0.0),
            sequence=int(data.get("sequence") or 0),
            payload_type=str(data.get("payload_type") or "auto"),
            payload=data.get("payload"),
            sample_rate=data.get("sample_rate"),
            units=dict(data.get("units") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ModalityState:
    """Explicit availability marker for one modality inside a window (§15)."""

    sensor_id: str
    state: str                                 # "ok" | "missing" | "stale"
    last_sample_timestamp: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SensorWindow:
    """Synchronized multi-sensor window (§15).

    ``samples`` maps sensor_id → list[SensorSample]. ``modalities`` records
    explicit missing/stale state so fusion processors never silently treat
    absent data as valid zeros.
    """

    start_timestamp: float
    end_timestamp: float
    samples: Dict[str, List[SensorSample]] = field(default_factory=dict)
    modalities: Dict[str, ModalityState] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)

    @property
    def sensors(self) -> tuple:
        return tuple(self.samples.keys())

    @property
    def duration(self) -> float:
        return max(0.0, self.end_timestamp - self.start_timestamp)

    def missing(self) -> List[str]:
        return [sid for sid, m in self.modalities.items() if m.state == "missing"]

    def stale(self) -> List[str]:
        return [sid for sid, m in self.modalities.items() if m.state == "stale"]

    def is_complete(self) -> bool:
        return all(m.state == "ok" for m in self.modalities.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "sensors": list(self.samples.keys()),
            "samples": {sid: [s.to_dict() for s in chunk]
                        for sid, chunk in self.samples.items()},
            "modalities": {sid: m.to_dict() for sid, m in self.modalities.items()},
            "preprocessing": self.preprocessing,
            "timing": self.timing,
        }


# ---------------------------------------------------------------------------
# Predictions and actions
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Prediction contract (§16)."""

    label: str
    confidence: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    device_id: str = ""
    runtime_model_id: str = ""
    task: str = ""                               # e.g. person_presence
    timestamp: float = field(default_factory=time.time)
    scores: Dict[str, float] = field(default_factory=dict)
    source_window: Optional[Dict[str, Any]] = None
    people_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def attributes(self) -> Dict[str, Any]:
        """Alias for ``metadata`` — model-specific result attributes."""
        return self.metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "device_id": self.device_id,
            "runtime_model_id": self.runtime_model_id,
            "task": self.task,
            "timestamp": self.timestamp,
            "label": self.label,
            "confidence": self.confidence,
            "scores": self.scores,
            "source_window": self.source_window,
            "people_count": self.people_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Prediction":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            device_id=str(data.get("device_id") or ""),
            runtime_model_id=str(data.get("runtime_model_id") or ""),
            task=str(data.get("task") or ""),
            timestamp=float(data.get("timestamp") or time.time()),
            label=str(data.get("label") or ""),
            confidence=float(data.get("confidence") or 0.0),
            scores=dict(data.get("scores") or {}),
            source_window=data.get("source_window"),
            people_count=data.get("people_count"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class RetryPolicy:
    max_attempts: int = 1
    backoff_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "RetryPolicy":
        data = data or {}
        return cls(
            max_attempts=int(data.get("max_attempts") or 1),
            backoff_seconds=float(data.get("backoff_seconds") or 0.0),
        )


@dataclass
class ActionResult:
    """Explicit actuator execution outcome (§7.6).

    An actuator may never report ``SUCCEEDED`` merely because its config
    parsed — only after the downstream effect was confirmed or the
    provider acknowledged the call.
    """

    status: ActionStatus
    action_type: str = ""
    detail: str = ""
    attempts: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    response: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status is ActionStatus.SUCCEEDED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "action_type": self.action_type,
            "detail": self.detail,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "response": self.response,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionResult":
        return cls(
            status=ActionStatus(data.get("status", "failed")),
            action_type=str(data.get("action_type") or ""),
            detail=str(data.get("detail") or ""),
            attempts=int(data.get("attempts") or 0),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            response=data.get("response"),
        )


@dataclass
class Action:
    """Action contract (§17) — binds a prediction outcome to an actuator."""

    type: str                                  # home_assistant | device | webhook
    config: Dict[str, Any] = field(default_factory=dict)
    min_confidence: float = 0.0
    trigger_labels: List[str] = field(default_factory=list)
    delay_seconds: float = 0.0
    debounce_seconds: float = 0.0
    cooldown_seconds: float = 0.0
    timeout_seconds: float = 10.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    result: Optional[ActionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "config": self.config,
            "min_confidence": self.min_confidence,
            "trigger_labels": self.trigger_labels,
            "delay_seconds": self.delay_seconds,
            "debounce_seconds": self.debounce_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "timeout_seconds": self.timeout_seconds,
            "retry_policy": self.retry_policy.to_dict(),
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Action":
        result = data.get("result")
        return cls(
            type=str(data.get("type") or ""),
            config=dict(data.get("config") or {}),
            min_confidence=float(data.get("min_confidence") or 0.0),
            trigger_labels=list(data.get("trigger_labels") or []),
            delay_seconds=float(data.get("delay_seconds") or 0.0),
            debounce_seconds=float(data.get("debounce_seconds") or 0.0),
            cooldown_seconds=float(data.get("cooldown_seconds") or 0.0),
            timeout_seconds=float(data.get("timeout_seconds") or 10.0),
            retry_policy=RetryPolicy.from_dict(data.get("retry_policy")),
            result=ActionResult.from_dict(result) if isinstance(result, Mapping) else None,
        )


# ---------------------------------------------------------------------------
# Model artifacts and deployments
# ---------------------------------------------------------------------------

@dataclass
class ModelInput:
    """One named model input.

    Legacy form (still parsed)::

        {"sensor": "radar", "window_seconds": 2.0,
         "required_sample_rate": 10.0}

    Capability form::

        {"name": "audio", "modality": "microphone",
         "capabilities": ["pcm_audio"],
         "constraints": {"sample_rate": 16000},
         "window_seconds": 4.0}

    Compatibility becomes "does this sensor satisfy what the model
    needs" rather than "is this sensor called microphone".
    """

    sensor: str = ""                           # legacy: sensor id/modality
    window_seconds: float = 1.0
    required_sample_rate: Optional[float] = None
    name: str = ""                             # input port name ("audio")
    modality: str = ""                         # required modality
    capabilities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    required: bool = True

    def __post_init__(self) -> None:
        # Legacy manifests only set ``sensor``; treat it as the modality
        # and default the port name so bindings can reference it.
        if self.sensor and not self.modality:
            self.modality = self.sensor
        if not self.name:
            self.name = self.modality or self.sensor

    def matches(self, sensor: "Sensor") -> bool:
        """Whether a Sensor contract satisfies this input's needs."""
        if self.modality and sensor.type != self.modality:
            return False
        missing = [c for c in self.capabilities
                   if c not in sensor.capabilities]
        if missing:
            return False
        rate_req = self.constraints.get("sample_rate") or \
            self.required_sample_rate
        if rate_req and sensor.sample_rate and \
                float(sensor.sample_rate) != float(rate_req):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor": self.sensor,
            "window_seconds": self.window_seconds,
            "required_sample_rate": self.required_sample_rate,
            "name": self.name,
            "modality": self.modality,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInput":
        window = data.get("window_seconds", data.get("window"))
        if isinstance(window, Mapping):
            window = window.get("seconds")
        return cls(
            sensor=str(data.get("sensor") or ""),
            window_seconds=float(window or 1.0),
            required_sample_rate=data.get("required_sample_rate"),
            name=str(data.get("name") or ""),
            modality=str(data.get("modality") or ""),
            capabilities=list(data.get("capabilities")
                              or data.get("requires") or []),
            constraints=dict(data.get("constraints") or {}),
            required=bool(data.get("required", True)),
        )


@dataclass
class ModelManifest:
    """``whispy-model/v1`` + ``whispy-model/v2`` artifact manifest (§7/§18).

    v2 adds a stable ``id``/``version``/``task`` identity, a declared
    ``lifecycle`` (streaming|windowed|batch), vendor-neutral ``execution``
    classes, ``resources`` hints, ``privacy`` classification, and a
    ``config_schema``. Legacy ``thoth-model/v1`` and ``whispy-model/v1``
    manifests are accepted and normalized on ``from_dict``.
    """

    name: str
    processor: str                             # one of PROCESSOR_TYPES or plugin id
    inputs: List[ModelInput] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    format: str = MODEL_MANIFEST_FORMAT
    whispy_version: str = ""
    artifact_sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # -- v2 fields (optional; defaults keep v1 semantics) ---------------------
    id: str = ""                               # stable model id (defaults to name)
    version: str = ""                          # model/package version
    task: str = ""                             # e.g. person_presence, stt
    lifecycle: str = "windowed"                # streaming | windowed | batch
    execution: List[str] = field(default_factory=lambda: ["local"])
    resources: Dict[str, Any] = field(default_factory=dict)
    privacy: Dict[str, Any] = field(default_factory=dict)
    config_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.id or self.name

    def __post_init__(self) -> None:
        self.inputs = [
            i if isinstance(i, ModelInput) else ModelInput.from_dict(i)
            for i in self.inputs
        ]

    def validate(self) -> List[str]:
        """Return a list of validation errors (empty when valid)."""
        errors: List[str] = []
        if self.format not in SUPPORTED_MANIFEST_FORMATS:
            errors.append(
                f"format must be one of {sorted(SUPPORTED_MANIFEST_FORMATS)}, "
                f"got {self.format!r}")
        if self.lifecycle and self.lifecycle not in MODEL_LIFECYCLES:
            errors.append(
                f"lifecycle must be one of {sorted(MODEL_LIFECYCLES)}, "
                f"got {self.lifecycle!r}")
        bad_exec = [e for e in self.execution if e not in EXECUTION_CLASSES]
        if bad_exec:
            errors.append(
                f"execution entries must be in {sorted(EXECUTION_CLASSES)}, "
                f"got {bad_exec}")
        if not self.name:
            errors.append("name is required")
        known = set(PROCESSOR_TYPES)
        try:
            from .plugins import PluginRegistry
            known.update(PluginRegistry().discover_models().keys())
        except Exception:
            pass
        if self.processor not in known:
            errors.append(
                f"processor must be a built-in {PROCESSOR_TYPES} or an "
                f"installed whispy.models plugin, got {self.processor!r}")
        if not self.inputs:
            errors.append("at least one input is required")
        for i, inp in enumerate(self.inputs):
            if not (inp.sensor or inp.modality or inp.name):
                errors.append(
                    f"inputs[{i}] requires a sensor, modality, or name")
            if inp.window_seconds <= 0:
                errors.append(f"inputs[{i}].window_seconds must be > 0")
        if self.artifact_sha256 and len(self.artifact_sha256) != 64:
            errors.append("artifact_sha256 must be a 64-char hex digest")
        return errors

    def verify_artifact(self, blob: bytes) -> bool:
        """Hash-check a model artifact against the manifest."""
        if not self.artifact_sha256:
            return True
        return hashlib.sha256(blob).hexdigest() == self.artifact_sha256

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "format": self.format,
            "name": self.name,
            "processor": self.processor,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": self.outputs,
            "whispy_version": self.whispy_version,
            "artifact_sha256": self.artifact_sha256,
            "metadata": self.metadata,
        }
        if self.format == MODEL_MANIFEST_FORMAT_V2:
            out.update({
                "id": self.model_id,
                "version": self.version,
                "task": self.task,
                "lifecycle": self.lifecycle,
                "execution": list(self.execution),
                "resources": self.resources,
                "privacy": self.privacy,
                "config_schema": self.config_schema,
            })
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelManifest":
        fmt = str(data.get("format") or data.get("schema")
                  or MODEL_MANIFEST_FORMAT)
        if fmt == LEGACY_MODEL_MANIFEST_FORMAT:
            fmt = MODEL_MANIFEST_FORMAT
        return cls(
            format=fmt,
            name=str(data.get("name") or data.get("id") or ""),
            processor=str(data.get("processor") or ""),
            inputs=[ModelInput.from_dict(i) for i in (data.get("inputs") or [])],
            outputs=list(data.get("outputs") or []),
            whispy_version=str(data.get("whispy_version") or ""),
            artifact_sha256=str(data.get("artifact_sha256")
                                or data.get("artifact_hash") or ""),
            metadata=dict(data.get("metadata") or {}),
            id=str(data.get("id") or ""),
            version=str(data.get("version") or ""),
            task=str(data.get("task") or ""),
            lifecycle=str(data.get("lifecycle") or "windowed"),
            execution=list(data.get("execution")
                           or data.get("execution_classes") or ["local"]),
            resources=dict(data.get("resources") or {}),
            privacy=dict(data.get("privacy") or {}),
            config_schema=dict(data.get("config_schema") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "ModelManifest":
        return cls.from_dict(json.loads(text))


@dataclass
class DeploymentFailure:
    stage: str
    code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Deployment:
    """Deployment record with explicit state machine (§19).

    ``runtime_model_id`` is generated at install time and persisted by the
    node so it survives daemon restarts.
    """

    deployment_id: str
    device_id: str
    model_id: str
    state: DeploymentState = DeploymentState.QUEUED
    runtime_model_id: str = ""
    failure: Optional[DeploymentFailure] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def transition(self, target: DeploymentState, *,
                   failure: Optional[DeploymentFailure] = None,
                   runtime_model_id: Optional[str] = None) -> DeploymentState:
        """Advance the state machine; raises on illegal transitions."""
        allowed = DEPLOYMENT_TRANSITIONS[self.state]
        if target not in allowed:
            raise ValueError(
                f"illegal deployment transition {self.state.value} → {target.value}")
        self.state = target
        self.updated_at = time.time()
        if failure is not None:
            self.failure = failure
        if runtime_model_id is not None:
            self.runtime_model_id = runtime_model_id
        self.history.append({
            "state": target.value, "at": self.updated_at,
            **({"failure": failure.to_dict()} if failure else {}),
        })
        return self.state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "device_id": self.device_id,
            "model_id": self.model_id,
            "state": self.state.value,
            "runtime_model_id": self.runtime_model_id,
            "failure": self.failure.to_dict() if self.failure else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Deployment":
        failure = data.get("failure")
        dep = cls(
            deployment_id=str(data.get("deployment_id") or ""),
            device_id=str(data.get("device_id") or ""),
            model_id=str(data.get("model_id") or ""),
            state=DeploymentState(data.get("state", "queued")),
            runtime_model_id=str(data.get("runtime_model_id") or ""),
            failure=DeploymentFailure(**failure) if isinstance(failure, Mapping) else None,
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            history=list(data.get("history") or []),
        )
        return dep


@dataclass
class Capture:
    """Capture lifecycle object (§22) — a logical session, not loose files."""

    id: str
    device_id: str
    started_at: float
    stopped_at: Optional[float] = None
    state: str = "active"                      # active | stopped | uploaded | failed
    sensors: List[str] = field(default_factory=list)
    sample_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Capture":
        return cls(
            id=str(data.get("id") or ""),
            device_id=str(data.get("device_id") or ""),
            started_at=float(data.get("started_at") or 0.0),
            stopped_at=data.get("stopped_at"),
            state=str(data.get("state") or "active"),
            sensors=list(data.get("sensors") or []),
            sample_counts=dict(data.get("sample_counts") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Observations (§4) — generalization of SensorSample
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """One timestamped observation from any source class.

    Generalizes :class:`SensorSample`: a battery reading, calendar entry,
    or foreground-application record is an Observation whose source has
    ``source_class="context"`` — it never pretends to be a physical sensor.
    Optional fields stay optional.
    """

    source_id: str
    device_id: str
    timestamp: float                           # epoch seconds (authoritative)
    payload: Any
    observation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema: str = ""                           # e.g. "digital.foreground_application/v1"
    sequence: int = 0
    payload_type: str = "auto"
    sample_rate: Optional[float] = None
    units: Dict[str, str] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional — remain None when not applicable.
    confidence: Optional[float] = None
    accuracy: Optional[float] = None
    spatial_reference: Optional[Dict[str, Any]] = None
    privacy_classification: Optional[str] = None

    @classmethod
    def from_sample(cls, sample: "SensorSample",
                    schema: str = "") -> "Observation":
        """Project a legacy SensorSample into an Observation."""
        return cls(
            source_id=sample.sensor_id,
            device_id=sample.device_id,
            timestamp=sample.timestamp,
            payload=sample.payload,
            schema=schema or sample.sensor_type,
            sequence=sample.sequence,
            payload_type=sample.payload_type,
            sample_rate=sample.sample_rate,
            units=dict(sample.units),
            provenance={"adapter": "sensor", "sensor_type": sample.sensor_type},
            metadata=dict(sample.metadata),
        )

    def to_sample(self, sensor_type: str = "") -> "SensorSample":
        """Project to the legacy SensorSample contract."""
        return SensorSample(
            device_id=self.device_id, sensor_id=self.source_id,
            sensor_type=sensor_type or self.schema,
            timestamp=self.timestamp, sequence=self.sequence,
            payload_type=self.payload_type, payload=self.payload,
            sample_rate=self.sample_rate, units=dict(self.units),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = self.payload
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        return {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "schema": self.schema,
            "sequence": self.sequence,
            "payload_type": self.payload_type,
            "payload": payload,
            "sample_rate": self.sample_rate,
            "units": self.units,
            "quality": self.quality,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "accuracy": self.accuracy,
            "spatial_reference": self.spatial_reference,
            "privacy_classification": self.privacy_classification,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        return cls(
            observation_id=str(data.get("observation_id")
                               or data.get("id") or uuid.uuid4().hex),
            source_id=str(data.get("source_id") or data.get("sensor_id") or ""),
            device_id=str(data.get("device_id") or ""),
            timestamp=float(data.get("timestamp") or 0.0),
            schema=str(data.get("schema") or data.get("sensor_type") or ""),
            sequence=int(data.get("sequence") or 0),
            payload_type=str(data.get("payload_type") or "auto"),
            payload=data.get("payload"),
            sample_rate=data.get("sample_rate"),
            units=dict(data.get("units") or {}),
            quality=dict(data.get("quality") or {}),
            provenance=dict(data.get("provenance") or {}),
            metadata=dict(data.get("metadata") or {}),
            confidence=data.get("confidence"),
            accuracy=data.get("accuracy"),
            spatial_reference=data.get("spatial_reference"),
            privacy_classification=data.get("privacy_classification"),
        )


class ObservationWindow(SensorWindow):
    """Synchronized multi-source window (§4/§15).

    Structurally identical to :class:`SensorWindow`; ``observations`` is a
    semantic alias for ``samples`` (source_id → list[Observation]).
    """

    @property
    def observations(self) -> Dict[str, List[Any]]:
        return self.samples

    @property
    def sources(self) -> tuple:
        return self.sensors


# ---------------------------------------------------------------------------
# Compute capabilities (§13)
# ---------------------------------------------------------------------------

@dataclass
class ComputeCapability:
    """Device compute advertisement (§13).

    Unknown metrics stay ``None`` — never fabricate thermal/GPU data.
    Model routing matches on these capabilities, never on device names.
    """

    architecture: str = ""
    logical_cpu_count: Optional[int] = None
    memory_total_mb: Optional[int] = None
    memory_available_mb: Optional[int] = None
    gpu: List[Dict[str, Any]] = field(default_factory=list)
    accelerators: List[str] = field(default_factory=list)
    vram_mb: Optional[int] = None
    storage_available_mb: Optional[int] = None
    battery: Optional[Dict[str, Any]] = None
    charging: Optional[bool] = None
    thermal: Optional[Dict[str, Any]] = None
    network: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComputeCapability":
        return cls(
            architecture=str(data.get("architecture") or ""),
            logical_cpu_count=data.get("logical_cpu_count"),
            memory_total_mb=data.get("memory_total_mb"),
            memory_available_mb=data.get("memory_available_mb"),
            gpu=list(data.get("gpu") or []),
            accelerators=list(data.get("accelerators") or []),
            vram_mb=data.get("vram_mb"),
            storage_available_mb=data.get("storage_available_mb"),
            battery=data.get("battery"),
            charging=data.get("charging"),
            thermal=data.get("thermal"),
            network=dict(data.get("network") or {}),
        )


# ---------------------------------------------------------------------------
# Inference (§14/§15)
# ---------------------------------------------------------------------------

@dataclass
class InferencePolicy:
    """Normalized execution policy (§14).

    One canonical representation: ``allowed``/``preferred`` subsets of
    EXECUTION_CLASSES plus constraints. Convenience strings
    (``local_only``, ``prefer_local``, ``prefer_edge``, ``cloud_only``,
    ``automatic``) map into this via :meth:`from_string`.
    """

    allowed: List[str] = field(
        default_factory=lambda: ["local", "trusted_edge"])
    preferred: List[str] = field(default_factory=list)
    max_latency_ms: Optional[float] = None
    max_cost: Optional[float] = None
    queue_if_unavailable: bool = False
    privacy: Dict[str, Any] = field(default_factory=dict)

    _PRESETS = {
        "local_only": (["local"], ["local"]),
        "prefer_local": (["local", "trusted_edge"], ["local", "trusted_edge"]),
        "prefer_edge": (["local", "trusted_edge"], ["trusted_edge", "local"]),
        "cloud_only": (["managed_cloud"], ["managed_cloud"]),
        "automatic": (["local", "trusted_edge", "managed_cloud"],
                      ["local", "trusted_edge", "managed_cloud"]),
    }

    @classmethod
    def from_string(cls, preset: str) -> "InferencePolicy":
        allowed, preferred = cls._PRESETS.get(
            preset, cls._PRESETS["automatic"])
        return cls(allowed=list(allowed), preferred=list(preferred))

    def __post_init__(self) -> None:
        bad = [e for e in self.allowed + self.preferred
               if e not in EXECUTION_CLASSES]
        if bad:
            raise ValueError(
                f"unknown execution classes {bad}; "
                f"valid: {sorted(EXECUTION_CLASSES)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "preferred": self.preferred,
            "max_latency_ms": self.max_latency_ms,
            "max_cost": self.max_cost,
            "queue_if_unavailable": self.queue_if_unavailable,
            "privacy": self.privacy,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "InferencePolicy":
        if isinstance(data, str):
            return cls.from_string(data)
        data = data or {}
        return cls(
            allowed=list(data.get("allowed")
                         or data.get("allowed_execution_classes")
                         or ["local", "trusted_edge"]),
            preferred=list(data.get("preferred")
                           or data.get("preferred_execution_classes") or []),
            max_latency_ms=data.get("max_latency_ms"),
            max_cost=data.get("max_cost"),
            queue_if_unavailable=bool(data.get("queue_if_unavailable", False)),
            privacy=dict(data.get("privacy") or {}),
        )


@dataclass
class InferenceTarget:
    """A resolved execution destination (§14)."""

    execution_class: str = "local"             # one of EXECUTION_CLASSES
    device_id: str = ""                        # target node ("" = requester)
    worker_id: str = ""
    provider: str = ""                         # external provider name
    endpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceTarget":
        return cls(
            execution_class=str(data.get("execution_class") or "local"),
            device_id=str(data.get("device_id") or ""),
            worker_id=str(data.get("worker_id") or ""),
            provider=str(data.get("provider") or ""),
            endpoint=str(data.get("endpoint") or ""),
        )


@dataclass
class InferenceRequest:
    """Source-bound inference request (§14/§16).

    ``bindings`` maps model input names to source ids; ``source_device``
    selects where each source lives. No provider-specific fields — those
    belong to the provider implementation, never the request.
    """

    model_id: str
    bindings: Dict[str, str] = field(default_factory=dict)   # input → source_id
    policy: InferencePolicy = field(default_factory=InferencePolicy)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_device: str = ""
    target: Optional[InferenceTarget] = None
    window_seconds: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "bindings": self.bindings,
            "policy": self.policy.to_dict(),
            "source_device": self.source_device,
            "target": self.target.to_dict() if self.target else None,
            "window_seconds": self.window_seconds,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceRequest":
        target = data.get("target")
        return cls(
            request_id=str(data.get("request_id") or uuid.uuid4().hex),
            model_id=str(data.get("model_id") or ""),
            bindings=dict(data.get("bindings") or {}),
            policy=InferencePolicy.from_dict(data.get("policy")),
            source_device=str(data.get("source_device") or ""),
            target=InferenceTarget.from_dict(target)
                if isinstance(target, Mapping) else None,
            window_seconds=data.get("window_seconds"),
            config=dict(data.get("config") or {}),
        )


@dataclass
class InferenceTrace:
    """Complete provenance for one inference run (§15)."""

    model_id: str
    model_version: str = ""
    artifact_hash: str = ""
    runtime_id: str = ""
    execution_device: str = ""
    execution_class: str = "local"
    input_bindings: Dict[str, str] = field(default_factory=dict)
    input_interval: Optional[Dict[str, float]] = None
    inference_timestamp: float = field(default_factory=time.time)
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None
    cpu_percent: Optional[float] = None
    gpu_percent: Optional[float] = None
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceTrace":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class InferenceResult:
    """Canonical inference output (§15)."""

    request_id: str = ""
    prediction: Optional[Prediction] = None
    trace: Optional[InferenceTrace] = None
    status: str = "succeeded"                  # succeeded | failed | queued
    error: str = ""
    outputs: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "error": self.error,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "trace": self.trace.to_dict() if self.trace else None,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceResult":
        pred = data.get("prediction")
        trace = data.get("trace")
        return cls(
            request_id=str(data.get("request_id") or ""),
            status=str(data.get("status") or "succeeded"),
            error=str(data.get("error") or ""),
            prediction=Prediction.from_dict(pred)
                if isinstance(pred, Mapping) else None,
            trace=InferenceTrace.from_dict(trace)
                if isinstance(trace, Mapping) else None,
            outputs=dict(data.get("outputs") or {}),
        )


# ---------------------------------------------------------------------------
# Context model (§30–§35)
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    """Subject–predicate–object edge in the context graph (§32)."""

    subject: str                               # entity id
    predicate: str                             # carries | near | located_in | …
    object: str                                # entity id
    valid_from: float = field(default_factory=time.time)
    valid_until: Optional[float] = None        # None = still valid
    confidence: float = 1.0
    source: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Relationship":
        return cls(
            subject=str(data.get("subject") or ""),
            predicate=str(data.get("predicate") or ""),
            object=str(data.get("object") or ""),
            valid_from=float(data.get("valid_from") or time.time()),
            valid_until=data.get("valid_until"),
            confidence=float(data.get("confidence") or 1.0),
            source=str(data.get("source") or ""),
            provenance=dict(data.get("provenance") or {}),
            id=str(data.get("id") or uuid.uuid4().hex),
        )


@dataclass
class ContextEvidence:
    """One piece of evidence feeding a ContextState (§34).

    Wraps an observation or prediction with provenance. Predictions are
    evidence — never truth.
    """

    key: str                                   # e.g. "spatial.presence/v1"
    value: Any
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str = ""
    device_id: str = ""
    prediction_id: str = ""
    observation_id: str = ""
    model_id: str = ""
    model_version: str = ""
    confidence: Optional[float] = None
    execution_class: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextEvidence":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class ContextState:
    """A derived context statement with evidence links (§34)."""

    key: str                                   # e.g. "semantic.working/v1"
    value: Any
    confidence: float = 1.0
    since: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    entity_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    estimator: str = ""                        # estimator/rule that produced it
    valid_until: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextState":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class ContextEvent:
    """A discrete transition emitted when a ContextState changes (§34)."""

    key: str
    event_type: str                            # entered | exited | changed
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    entity_id: str = ""
    state_id: str = ""
    value: Any = None
    previous_value: Any = None
    confidence: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextEvent":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Canonical minute (§9)
# ---------------------------------------------------------------------------

@dataclass
class MinuteSourceData:
    """Per-source series inside a minute — timestamps are authoritative."""

    source_id: str
    modality: str = ""
    timestamps: List[float] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    second_offsets: List[float] = field(default_factory=list)
    units: Dict[str, str] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MinuteSourceData":
        return cls(
            source_id=str(data.get("source_id") or data.get("sensor_id") or ""),
            modality=str(data.get("modality") or data.get("type") or ""),
            timestamps=list(data.get("timestamps") or []),
            values=list(data.get("values") or []),
            second_offsets=list(data.get("second_offsets") or []),
            units=dict(data.get("units") or {}),
            quality=dict(data.get("quality") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class MinuteManifest:
    """``thoth-minute/v1`` — canonical minute container manifest (§9).

    Replaces the legacy chunk-oriented capture manifest as the domain
    temporal abstraction. ``start_timestamp`` is authoritative; a minute
    may contain multiple heterogeneous sources, predictions, events and
    annotations. Never assume 60 samples.
    """

    minute_id: str                             # e.g. "20260924_1014"
    device_id: str
    start_timestamp: float
    format: str = MINUTE_MANIFEST_FORMAT
    end_timestamp: Optional[float] = None
    duration_seconds: Optional[float] = None
    sources: List[MinuteSourceData] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)   # e.g. {"npz": "capture.npz"}
    checksums: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "minute_id": self.minute_id,
            "device_id": self.device_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "sources": [s.to_dict() for s in self.sources],
            "predictions": self.predictions,
            "events": self.events,
            "annotations": self.annotations,
            "labels": self.labels,
            "quality": self.quality,
            "source_metadata": self.source_metadata,
            "files": self.files,
            "checksums": self.checksums,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MinuteManifest":
        return cls(
            format=str(data.get("format") or MINUTE_MANIFEST_FORMAT),
            minute_id=str(data.get("minute_id") or data.get("minute") or ""),
            device_id=str(data.get("device_id") or ""),
            start_timestamp=float(
                data.get("start_timestamp") or data.get("timestamp") or 0.0),
            end_timestamp=data.get("end_timestamp"),
            duration_seconds=data.get("duration_seconds"),
            sources=[MinuteSourceData.from_dict(s)
                     for s in (data.get("sources") or [])],
            predictions=list(data.get("predictions") or []),
            events=list(data.get("events") or []),
            annotations=list(data.get("annotations") or []),
            labels=dict(data.get("labels") or {}),
            quality=dict(data.get("quality") or {}),
            source_metadata=dict(data.get("source_metadata") or {}),
            files=dict(data.get("files") or {}),
            checksums=dict(data.get("checksums") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


# ``ActionRequest`` is the schema-level name for an actuator invocation;
# ActuatorCommand already carries operation/params/timeout on the wire.
ActionRequest = ActuatorCommand
