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
from typing import Any, Dict, Optional

from ..contracts import Action, ActionResult, ActionStatus, Prediction

logger = logging.getLogger(__name__)


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


def create_actuator(action: Action) -> Actuator:
    """Instantiate the actuator plugin for an Action contract."""
    from .home_assistant import HomeAssistantActuator
    from .device import DeviceActuator
    from .webhook import WebhookActuator

    kind = action.type.lower()
    if kind in ("home_assistant", "ha"):
        return HomeAssistantActuator(action.config)
    if kind in ("device", "device_action", "gpio", "local"):
        return DeviceActuator(action.config)
    if kind == "webhook":
        return WebhookActuator(action.config)
    raise ValueError(f"unknown actuator type: {action.type!r}")


__all__ = ["Actuator", "create_actuator"]
