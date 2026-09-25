"""Actuator contract — real execution outcomes only (§7.6, §17).

Every actuator returns an :class:`~whispy.contracts.ActionResult` whose
status is one of ``queued | executing | succeeded | failed |
unsupported``. No actuator may report ``succeeded`` merely because its
configuration parsed — success requires a confirmed downstream effect or
a provider acknowledgement.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type

from ..contracts import (
    Action, ActionResult, ActionStatus, ActuatorCommand,
    ActuatorDescriptor, Prediction,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "whispy.actuators"


class Actuator(ABC):
    """Base class for all actuator plugins."""

    actuator_type: str = "abstract"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.name = str(self.config.get("name") or self.actuator_type)

    @abstractmethod
    def execute(self, action: Action, prediction: Prediction) -> ActionResult:
        """Perform the action; return the confirmed outcome."""

    def supported(self) -> bool:
        """Whether this actuator can run in the current environment."""
        return True

    def trigger(self, action: Action, prediction: Prediction) -> ActionResult:
        """Gate + execute an action for a prediction.

        Applies confidence gating and label filtering before dispatch;
        the returned result is always explicit.
        """
        if not self.supported():
            return ActionResult(
                status=ActionStatus.UNSUPPORTED, action_type=self.actuator_type,
                detail=f"{type(self).__name__} unsupported in this environment")
        if prediction.confidence < action.min_confidence:
            return ActionResult(
                status=ActionStatus.FAILED, action_type=self.actuator_type,
                detail=(f"confidence {prediction.confidence:.3f} below "
                        f"min_confidence {action.min_confidence:.3f}"))
        if action.trigger_labels and prediction.label not in action.trigger_labels:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED, action_type=self.actuator_type,
                detail=f"label {prediction.label!r} not in trigger_labels")

        attempts = max(1, action.retry_policy.max_attempts)
        result = ActionResult(status=ActionStatus.QUEUED,
                              action_type=self.actuator_type,
                              started_at=time.time())
        for attempt in range(1, attempts + 1):
            result.status = ActionStatus.EXECUTING
            result.attempts = attempt
            try:
                result = self.execute(action, prediction)
                result.attempts = attempt
            except Exception as exc:  # actuator bugs must not kill the loop
                logger.warning("Actuator %s raised: %s", self.name, exc)
                result = ActionResult(
                    status=ActionStatus.FAILED, action_type=self.actuator_type,
                    detail=str(exc), attempts=attempt,
                    started_at=result.started_at)
            if result.status is ActionStatus.SUCCEEDED:
                break
            if attempt < attempts and action.retry_policy.backoff_seconds > 0:
                time.sleep(action.retry_policy.backoff_seconds)
        result.finished_at = time.time()
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "actuator_type": self.actuator_type,
                "config": self.config}


# ---------------------------------------------------------------------------
# Device actuator adapters — physical actuators as device capabilities
# ---------------------------------------------------------------------------

@dataclass
class ActuatorMeta:
    """Adapter metadata — powers registry listings and config UIs."""

    name: str
    version: str = "1.0.0"
    kinds: tuple = ()                           # e.g. ("speaker",), ("matrix",)
    description: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)
    maintainer: Optional[str] = None


class ActuatorHandle(ABC):
    """A controllable actuator on a device (local or remote).

    ``execute`` takes an :class:`ActuatorCommand` — an operation name
    plus params — and returns an explicit :class:`ActionResult`.
    """

    @property
    @abstractmethod
    def info(self) -> ActuatorDescriptor:
        """Static actuator descriptor."""

    @abstractmethod
    def execute(self, command: ActuatorCommand) -> ActionResult:
        """Perform ``command``; return the confirmed outcome."""

    def supports(self, operation: str) -> bool:
        return operation in (self.info.operations or [])

    def close(self) -> None:
        """Release the actuator; safe to call twice."""


class ActuatorAdapter(ABC):
    """Discovers physical actuators and connects handles to them.

    Mirrors :class:`~whispy.sensors.base.SensorAdapter`::

        adapter.discover()         → list[ActuatorDescriptor] (0..n)
        adapter.connect(desc, cfg) → ActuatorHandle
    """

    @abstractmethod
    def metadata(self) -> ActuatorMeta:
        """Static adapter metadata."""

    @abstractmethod
    def discover(self) -> List[ActuatorDescriptor]:
        """Enumerate physical actuators this adapter can serve."""

    @abstractmethod
    def connect(self, descriptor: ActuatorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> ActuatorHandle:
        """Open one discovered actuator; returns an ActuatorHandle."""

    def health(self) -> Dict[str, Any]:
        return {"status": "ok"}

    def close(self) -> None:
        """Release adapter-level resources; safe to call twice."""


# ---------------------------------------------------------------------------
# Action-executor registry — Action.type → Actuator implementation
# ---------------------------------------------------------------------------

_ALIASES = {
    "ha": "home_assistant",
    "device_action": "device",
    "gpio": "device",
    "local": "device",
    "remote": "lan",
    "node": "lan",
}


def _builtin_executors() -> Dict[str, Type[Actuator]]:
    from .home_assistant import HomeAssistantActuator
    from .device import DeviceActuator
    from .lan import LanActuator
    from .webhook import WebhookActuator

    return {
        "home_assistant": HomeAssistantActuator,
        "device": DeviceActuator,
        "lan": LanActuator,
        "webhook": WebhookActuator,
    }


def actuator_executors() -> Dict[str, Type[Actuator]]:
    """Action-executor plugins: built-ins + ``whispy.actuators`` entries.

    Entry points exporting an :class:`Actuator` subclass register under
    their ``actuator_type``; ``ActuatorAdapter`` subclasses in the same
    group are device adapters, not executors, and are skipped here.
    """
    from ..plugins import registry

    out = _builtin_executors()
    for _name, info in registry().discover_actuators().items():
        if not info.available or info.cls is None or info.kind != "executor":
            continue
        act_type = getattr(info.cls, "actuator_type", "") or _name
        out[act_type] = info.cls
    return out


def installed_actuator_adapters() -> Dict[str, ActuatorAdapter]:
    """Instantiate every ``whispy.actuators`` adapter plugin."""
    from ..plugins import registry

    out: Dict[str, ActuatorAdapter] = {}
    for name, info in registry().discover_actuators().items():
        if not info.available or info.cls is None or info.kind != "adapter":
            continue
        try:
            out[name] = info.cls()
        except Exception:
            continue
    return out


def create_actuator(action: Action) -> Actuator:
    """Instantiate the action-executor plugin for an Action contract."""
    kind = _ALIASES.get(action.type.lower(), action.type.lower())
    cls = actuator_executors().get(kind)
    if cls is None:
        raise ValueError(
            f"unknown actuator type: {action.type!r}; installed: "
            f"{sorted(actuator_executors())}")
    return cls(action.config)


__all__ = [
    "ENTRY_POINT_GROUP",
    "Actuator",
    "ActuatorAdapter",
    "ActuatorHandle",
    "ActuatorMeta",
    "actuator_executors",
    "create_actuator",
    "installed_actuator_adapters",
]
