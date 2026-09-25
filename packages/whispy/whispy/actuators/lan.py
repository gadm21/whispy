"""LAN actuator — execute an actuator on another Thoth node (§17, §24).

Config::

    {
        "host": "10.0.0.22", "port": 5001, "token": "<local_token>",
        "actuator": "light-c483",            # id, name, or kind
        "operation": "set_color",
        "params": {"rgb": [0, 255, 0]}
    }

Uses ``whispy.lan()`` — the same authenticated path the client SDK uses.
``succeeded`` requires the remote node to report a real result, never a
transport-level guess.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..automation import format_context, render
from ..contracts import Action, ActionResult, ActionStatus, Prediction
from .base import Actuator

logger = logging.getLogger(__name__)


class LanActuator(Actuator):
    """Executes ``ActuatorCommand``s on a remote node's local API."""

    actuator_type = "lan"

    def execute(self, action: Action, prediction: Prediction) -> ActionResult:
        fmt = format_context(prediction)
        cfg = render({**self.config, **(action.config or {})}, fmt)
        host = str(cfg.get("host") or "")
        target = cfg.get("actuator") or cfg.get("actuator_id")
        operation = str(cfg.get("operation") or cfg.get("command") or "")
        if not host or not target or not operation:
            return ActionResult(
                status=ActionStatus.FAILED, action_type=self.actuator_type,
                detail="lan action requires 'host', 'actuator', 'operation'")
        try:
            from ..devices.local import lan
            dev = lan(host, port=int(cfg.get("port") or 5000),
                      token=str(cfg.get("token") or ""))
            handle = dev.actuator(str(target))
            if handle is None:
                return ActionResult(
                    status=ActionStatus.UNSUPPORTED,
                    action_type=self.actuator_type,
                    detail=f"no actuator {target!r} on {host}")
            result = handle.execute({
                "operation": operation,
                "params": cfg.get("params") or {},
            })
            return ActionResult(
                status=(ActionStatus.SUCCEEDED
                        if getattr(result, "status", None).value == "succeeded"
                        else ActionStatus.FAILED),
                action_type=self.actuator_type,
                detail=f"remote {target}:{operation} → "
                       f"{getattr(result.status, 'value', result.status)}",
                response={"host": host, "actuator": str(target),
                          "operation": operation,
                          "remote": result.to_dict() if hasattr(
                              result, "to_dict") else str(result)})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type,
                                detail=str(exc))


__all__ = ["LanActuator"]
