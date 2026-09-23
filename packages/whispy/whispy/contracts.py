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

    @property
    def terminal(self) -> bool:
        return self in (
            ActionStatus.SUCCEEDED, ActionStatus.FAILED, ActionStatus.UNSUPPORTED)


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
MODEL_MANIFEST_FORMAT = "thoth-model/v1"


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
    timestamp: float = field(default_factory=time.time)
    scores: Dict[str, float] = field(default_factory=dict)
    source_window: Optional[Dict[str, Any]] = None
    people_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "device_id": self.device_id,
            "runtime_model_id": self.runtime_model_id,
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
    sensor: str
    window_seconds: float = 1.0
    required_sample_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInput":
        return cls(
            sensor=str(data.get("sensor") or ""),
            window_seconds=float(data.get("window_seconds") or 1.0),
            required_sample_rate=data.get("required_sample_rate"),
        )


@dataclass
class ModelManifest:
    """``thoth-model/v1`` artifact manifest (§18)."""

    name: str
    processor: str                             # one of PROCESSOR_TYPES
    inputs: List[ModelInput] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    format: str = MODEL_MANIFEST_FORMAT
    whispy_version: str = ""
    artifact_sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.inputs = [
            i if isinstance(i, ModelInput) else ModelInput.from_dict(i)
            for i in self.inputs
        ]

    def validate(self) -> List[str]:
        """Return a list of validation errors (empty when valid)."""
        errors: List[str] = []
        if self.format != MODEL_MANIFEST_FORMAT:
            errors.append(f"format must be {MODEL_MANIFEST_FORMAT!r}, got {self.format!r}")
        if not self.name:
            errors.append("name is required")
        if self.processor not in PROCESSOR_TYPES:
            errors.append(
                f"processor must be one of {PROCESSOR_TYPES}, got {self.processor!r}")
        if not self.inputs:
            errors.append("at least one input is required")
        for i, inp in enumerate(self.inputs):
            if not inp.sensor:
                errors.append(f"inputs[{i}].sensor is required")
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
        return {
            "format": self.format,
            "name": self.name,
            "processor": self.processor,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": self.outputs,
            "whispy_version": self.whispy_version,
            "artifact_sha256": self.artifact_sha256,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelManifest":
        return cls(
            format=str(data.get("format") or MODEL_MANIFEST_FORMAT),
            name=str(data.get("name") or ""),
            processor=str(data.get("processor") or ""),
            inputs=[ModelInput.from_dict(i) for i in (data.get("inputs") or [])],
            outputs=list(data.get("outputs") or []),
            whispy_version=str(data.get("whispy_version") or ""),
            artifact_sha256=str(data.get("artifact_sha256") or ""),
            metadata=dict(data.get("metadata") or {}),
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
