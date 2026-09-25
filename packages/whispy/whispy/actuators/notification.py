"""Notification actuator — publish a user-visible notification event.

Produces a normalized ``notification`` payload; the daemon's executor
bridge emits it as a Brain ``event`` frame → ``node_event`` row → SSE
stream/webhook subscribers → the mobile app's notification feed.

``succeeded`` means the payload was accepted onto the node's event
channel (the only delivery path the actuator can confirm locally); the
Brain event id travels in ``response.event_id`` when the frame is acked.

Config::

    {
        "title": "Thoth — {label}",
        "body":  "Room is now {label} ({confidence:.0%})",
        "severity": "info"               # info | warning | alert
    }

``title``/``body`` may reference ``{label}``, ``{confidence}``,
``{device_id}``, ``{timestamp}`` — formatted per prediction.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from ..contracts import Action, ActionResult, ActionStatus, Prediction
from .base import Actuator
from .webhook import WebhookActuator


class NotificationActuator(Actuator):
    """Emit a platform notification (Brain event channel)."""

    actuator_type = "notification"

    def execute(self, action: Action, prediction: Prediction) -> ActionResult:
        cfg = {**self.config, **(action.config or {})}
        fmt = {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "device_id": prediction.device_id,
            "timestamp": prediction.timestamp,
        }
        title = WebhookActuator._render(
            cfg.get("title") or "Thoth — {label}", fmt)
        body = WebhookActuator._render(
            cfg.get("body") or "{label} on {device_id}", fmt)
        notification = {
            "title": str(title),
            "body": str(body),
            "severity": str(cfg.get("severity") or "info"),
            "label": prediction.label,
            "confidence": prediction.confidence,
            "device_id": prediction.device_id,
            "at": time.time(),
        }
        # The daemon bridge picks ``result.response["notification"]`` and
        # emits it as a Brain event frame — confirmed delivery = payload
        # produced + accepted for publication.
        return ActionResult(
            status=ActionStatus.SUCCEEDED,
            action_type=self.actuator_type,
            detail="notification published to node event channel",
            response={"notification": notification})


__all__ = ["NotificationActuator"]
