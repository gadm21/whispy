"""Sense HAT LED matrix actuator — 8×8 display for Whispy.

Operations::

    show(pixels, duration_s?)   — set the 8×8 matrix to an RGB pattern
    show_message(text, speed?, color?) — scroll text
    clear()                     — blank the matrix

``pixels`` is 64 cells of ``[r,g,b]`` (0–255) or 0/1 for monochrome.
Success requires the HAT to accept the frame — no fake success when the
hardware is absent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from whispy.contracts import (
    ActionResult, ActionStatus, ActuatorCommand, ActuatorDescriptor,
)
from whispy.actuators.base import (
    ActuatorAdapter, ActuatorHandle, ActuatorMeta,
)

logger = logging.getLogger(__name__)

HARDWARE_ID = "rpi-sensehat-matrix"


def _sense_hat():
    try:
        from sense_hat import SenseHat  # type: ignore
        return SenseHat()
    except Exception:
        try:
            from sense_emu import SenseHat  # type: ignore
            return SenseHat()
        except Exception:
            return None


def _normalize_pixels(pixels: Any) -> List[List[int]]:
    """Accept 64 flat cells or 8×8 nested; emit 64 [r,g,b] triples."""
    flat: List[Any] = []
    if isinstance(pixels, list) and pixels and isinstance(pixels[0], list) \
            and pixels[0] and isinstance(pixels[0][0], (list, tuple)):
        flat = [cell for row in pixels for cell in row]
    else:
        flat = list(pixels or [])
    out: List[List[int]] = []
    for cell in flat[:64]:
        if isinstance(cell, (list, tuple)) and len(cell) >= 3:
            out.append([int(max(0, min(255, c))) for c in cell[:3]])
        else:
            v = 255 if cell else 0
            out.append([v, v, v])
    while len(out) < 64:
        out.append([0, 0, 0])
    return out


class _MatrixHandle(ActuatorHandle):
    def __init__(self, hat, descriptor: ActuatorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._hat = hat
        self._desc = descriptor
        self._config = dict(config or {})

    @property
    def info(self) -> ActuatorDescriptor:
        return self._desc

    @property
    def descriptor(self) -> ActuatorDescriptor:
        return self._desc

    def execute(self, command: ActuatorCommand) -> ActionResult:
        started = time.time()
        op = command.operation
        params = command.params or {}
        try:
            if op == "show":
                pixels = _normalize_pixels(params.get("pixels"))
                self._hat.set_pixels(pixels)
                duration = float(params.get("duration_s") or 0)
                if duration > 0:
                    time.sleep(duration)
                    self._hat.clear()
                result = ActionResult(
                    status=ActionStatus.SUCCEEDED, action_type="matrix",
                    detail="matrix frame set",
                    response={"cells": len(pixels)})
            elif op == "show_message":
                text = str(params.get("text") or "")
                if not text:
                    result = ActionResult(
                        status=ActionStatus.FAILED, action_type="matrix",
                        detail="show_message requires 'text'")
                else:
                    color = params.get("color") or [255, 255, 255]
                    self._hat.show_message(
                        text, scroll_speed=float(params.get("speed") or 0.1),
                        text_colour=[int(c) for c in color[:3]])
                    result = ActionResult(
                        status=ActionStatus.SUCCEEDED, action_type="matrix",
                        detail=f"scrolled {len(text)} chars")
            elif op == "clear":
                self._hat.clear()
                result = ActionResult(status=ActionStatus.SUCCEEDED,
                                      action_type="matrix",
                                      detail="matrix cleared")
            else:
                result = ActionResult(
                    status=ActionStatus.UNSUPPORTED, action_type="matrix",
                    detail=f"unknown operation {op!r}")
        except Exception as exc:
            result = ActionResult(status=ActionStatus.FAILED,
                                  action_type="matrix", detail=str(exc))
        result.started_at = started
        result.finished_at = time.time()
        return result

    def close(self) -> None:
        try:
            self._hat.clear()
        except Exception:
            pass


class SenseHatMatrixAdapter(ActuatorAdapter):
    """Discovers the Sense HAT LED matrix (8×8)."""

    def __init__(self):
        self._hat = None
        self._hat_tried = False
        self._handles: List[_MatrixHandle] = []

    def metadata(self) -> ActuatorMeta:
        return ActuatorMeta(
            name="sensehat-matrix",
            version="0.1.0",
            kinds=("matrix",),
            description="Sense HAT 8×8 LED matrix",
            config_schema={"type": "object", "properties": {}},
            maintainer="thothcraft",
        )

    def _get_hat(self):
        if not self._hat_tried:
            self._hat = _sense_hat()
            self._hat_tried = True
        return self._hat

    def discover(self) -> List[ActuatorDescriptor]:
        if self._get_hat() is None:
            return []
        return [ActuatorDescriptor(
            id=ActuatorDescriptor.make_id("matrix", HARDWARE_ID),
            kind="matrix",
            adapter="sensehat-matrix",
            name="Sense HAT matrix",
            hardware_id=HARDWARE_ID,
            operations=["show", "show_message", "clear"],
            capabilities=["rgb8x8"],
            stable=True,
        )]

    def connect(self, descriptor: ActuatorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _MatrixHandle:
        hat = self._get_hat()
        if hat is None:
            raise RuntimeError("Sense HAT not available")
        handle = _MatrixHandle(hat, descriptor, config)
        self._handles.append(handle)
        return handle

    def health(self) -> Dict[str, Any]:
        return {"status": "ok" if self._get_hat() is not None else "error"}

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


__all__ = ["SenseHatMatrixAdapter"]
