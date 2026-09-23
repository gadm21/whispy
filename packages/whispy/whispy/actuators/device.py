"""Device actuator — local hardware effects (GPIO, buzzer, scripts).

Success requires a confirmed effect: a GPIO backend that actually wrote
the pin, a handler callback that returned without raising, or a shell
command that exited 0. Missing hardware → ``unsupported``/``failed``,
never silent success (§46: "no success is returned for nonexistent
hardware").
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any, Callable, Dict, Optional

from ..contracts import Action, ActionResult, ActionStatus, Prediction
from .base import Actuator

logger = logging.getLogger(__name__)

# Optional handler registry: tests/integrations register real callbacks.
_HANDLERS: Dict[str, Callable[[Action, Prediction], Any]] = {}


def register_handler(kind: str, handler: Callable[[Action, Prediction], Any]) -> None:
    """Register a hardware callback for a device action kind.

    The handler performs the physical effect and returns a truthy/None
    value on success or raises on failure.
    """
    _HANDLERS[kind] = handler


class DeviceActuator(Actuator):
    """Executes local device actions.

    Config::

        {"action": "gpio_toggle", "pin": 18, "state": "HIGH",
         "duration_sec": 5.0}
        {"action": "shell", "command": "/opt/thoth/estop.sh"}
        {"action": "buzzer", "frequency": 2000, "duration_sec": 0.5}

    GPIO uses ``gpiozero`` when installed; otherwise the action reports
    ``unsupported`` rather than pretending to toggle a pin.
    """

    actuator_type = "device"

    def execute(self, action: Action, prediction: Prediction) -> ActionResult:
        cfg = {**self.config, **(action.config or {})}
        kind = str(cfg.get("action") or cfg.get("action_kind") or "")

        # Registered handler wins — real hardware integration point.
        handler = _HANDLERS.get(kind)
        if handler is not None:
            try:
                handler(action, prediction)
                return ActionResult(
                    status=ActionStatus.SUCCEEDED, action_type=self.actuator_type,
                    detail=f"handler {kind!r} confirmed",
                    response={"kind": kind})
            except Exception as exc:
                return ActionResult(
                    status=ActionStatus.FAILED, action_type=self.actuator_type,
                    detail=f"handler {kind!r} raised: {exc}")

        if kind in ("gpio_toggle", "gpio", "relay"):
            return self._gpio(cfg)
        if kind == "shell":
            return self._shell(cfg, action.timeout_seconds)
        if kind in ("buzzer", "sensehat_matrix"):
            return ActionResult(
                status=ActionStatus.UNSUPPORTED, action_type=self.actuator_type,
                detail=f"no registered handler for {kind!r} on this host")
        return ActionResult(
            status=ActionStatus.UNSUPPORTED, action_type=self.actuator_type,
            detail=f"unknown device action {kind!r}")

    def _gpio(self, cfg: Dict[str, Any]) -> ActionResult:
        pin = cfg.get("pin")
        if pin is None:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type,
                                detail="gpio action requires 'pin'")
        try:
            import gpiozero  # type: ignore
        except ImportError:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED, action_type=self.actuator_type,
                detail="gpiozero not installed — cannot confirm GPIO write")
        try:
            output = gpiozero.DigitalOutputDevice(int(pin))
            state = str(cfg.get("state") or "HIGH").upper()
            if state == "HIGH":
                output.on()
            else:
                output.off()
            # Confirm by reading the pin value back.
            confirmed = output.value == (1 if state == "HIGH" else 0)
            output.close()
            return ActionResult(
                status=ActionStatus.SUCCEEDED if confirmed else ActionStatus.FAILED,
                action_type=self.actuator_type,
                detail=f"gpio pin {pin} → {state}",
                response={"pin": pin, "state": state, "confirmed": confirmed})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type, detail=str(exc))

    def _shell(self, cfg: Dict[str, Any], timeout: float) -> ActionResult:
        command = cfg.get("command")
        if not command:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type,
                                detail="shell action requires 'command'")
        if isinstance(command, str) and not shutil.which(command.split()[0]):
            return ActionResult(
                status=ActionStatus.UNSUPPORTED, action_type=self.actuator_type,
                detail=f"command not found: {command.split()[0]}")
        try:
            proc = subprocess.run(
                command, shell=isinstance(command, str),
                capture_output=True, text=True, timeout=timeout)
            ok = proc.returncode == 0
            return ActionResult(
                status=ActionStatus.SUCCEEDED if ok else ActionStatus.FAILED,
                action_type=self.actuator_type,
                detail=f"exit {proc.returncode}",
                response={"returncode": proc.returncode,
                          "stdout": proc.stdout[-500:], "stderr": proc.stderr[-500:]})
        except subprocess.TimeoutExpired:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type,
                                detail=f"timeout after {timeout}s")
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type, detail=str(exc))


__all__ = ["DeviceActuator", "register_handler"]
