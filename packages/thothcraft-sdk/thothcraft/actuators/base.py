"""Standard modular actuator interface for the ThothCraft platform.

An Actuator consumes a :class:`~thothcraft.processors.base.Prediction` and
triggers an action on a downstream consumer:
  - Home Assistant (light, switch, script, webhook)
  - Device Hardware (motor, robothand, buzzer, Sense HAT LED matrix, GPIO)
  - Webhook (external REST / automation endpoint)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Actuator(ABC):
    """Abstract base class for all actuator plugins."""

    def __init__(self, name: str, actuator_type: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.actuator_type = actuator_type
        self.config = config or {}

    @abstractmethod
    def trigger(self, prediction: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute action based on the model prediction."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "actuator_type": self.actuator_type,
            "config": self.config,
        }


class HomeAssistantActuator(Actuator):
    """Controls Home Assistant entities (lights, switches, scripts) or fires webhooks

    when a model prediction matches triggering conditions.
    """

    def __init__(
        self,
        name: str = "home_assistant",
        url: str = "http://localhost:8123",
        token: str = "",
        entity_id: str = "",
        positive_action: str = "homeassistant.turn_on",
        negative_action: str = "homeassistant.turn_off",
        trigger_labels: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, "home_assistant", config or {})
        self.url = (config.get("url") if config else None) or url.rstrip("/")
        self.token = (config.get("token") if config else None) or token
        self.entity_id = (config.get("entity_id") if config else None) or entity_id
        self.positive_action = (config.get("positive_action") if config else None) or positive_action
        self.negative_action = (config.get("negative_action") if config else None) or negative_action
        self.trigger_labels = (config.get("trigger_labels") if config else None) or trigger_labels or [
            "occupied", "present", "face_detected", "motion", "overheating", "alert"
        ]

    def trigger(self, prediction: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        label = str(getattr(prediction, "label", "") or "").lower()
        confidence = float(getattr(prediction, "confidence", 1.0) or 1.0)

        is_positive = label in [l.lower() for l in self.trigger_labels]
        action = self.positive_action if is_positive else self.negative_action

        if not action:
            return {"status": "skipped", "reason": "no action defined for label"}

        # Action is in "domain.service" form, e.g. "light.turn_on" or "switch.turn_off"
        parts = action.split(".", 1)
        if len(parts) != 2:
            domain, service = "homeassistant", action
        else:
            domain, service = parts[0], parts[1]

        target_url = f"{self.url}/api/services/{domain}/{service}"
        payload: Dict[str, Any] = {}
        if self.entity_id:
            payload["entity_id"] = self.entity_id

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        try:
            req = urllib.request.Request(
                target_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as res:
                response_data = res.read().decode("utf-8")
                return {
                    "status": "success",
                    "actuator": "home_assistant",
                    "action": action,
                    "entity_id": self.entity_id,
                    "prediction_label": label,
                    "confidence": confidence,
                    "response": response_data[:200],
                }
        except Exception as exc:
            logger.warning("Home Assistant actuator call failed: %s", exc)
            return {
                "status": "error",
                "actuator": "home_assistant",
                "action": action,
                "error": str(exc),
            }


class DeviceActuator(Actuator):
    """Triggers physical actions on the node: motors, robothand, LED matrix, buzzer."""

    def __init__(
        self,
        name: str = "device_actuator",
        action_kind: str = "sensehat_matrix",  # sensehat_matrix | motor | robothand | custom
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, "device_action", config or {})
        self.action_kind = (config.get("action_kind") if config else None) or action_kind
        self.on_trigger_payload = (config.get("on_trigger_payload") if config else None) or {}
        self.on_clear_payload = (config.get("on_clear_payload") if config else None) or {}

    def trigger(self, prediction: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        label = str(getattr(prediction, "label", "") or "").lower()
        is_positive = label not in {"", "empty", "no_face", "normal", "none", "unknown"}

        # If on Sense HAT, can drive LED matrix
        if self.action_kind == "sensehat_matrix":
            try:
                from sense_hat import SenseHat  # type: ignore
                sense = SenseHat()
                if is_positive:
                    # Show alert or color
                    color = self.on_trigger_payload.get("color", [0, 255, 0])
                    text = self.on_trigger_payload.get("text", "")
                    if text:
                        sense.show_message(text, scroll_speed=0.08)
                    else:
                        sense.clear(color)
                else:
                    sense.clear([0, 0, 0])
                return {"status": "success", "device_action": "sensehat_matrix", "positive": is_positive}
            except Exception as exc:
                return {"status": "skipped", "reason": f"SenseHat unavailable: {exc}"}

        # For motor / robothand: call hardware callback or serial port if configured
        handler = (context or {}).get(f"actuator_handler_{self.action_kind}")
        if callable(handler):
            try:
                res = handler(prediction, is_positive)
                return {"status": "success", "device_action": self.action_kind, "result": res}
            except Exception as exc:
                return {"status": "error", "device_action": self.action_kind, "error": str(exc)}

        return {
            "status": "success",
            "device_action": self.action_kind,
            "dispatched": is_positive,
            "payload": self.on_trigger_payload if is_positive else self.on_clear_payload,
        }


class WebhookActuator(Actuator):
    """Sends HTTP POST to an arbitrary webhook endpoint with prediction details."""

    def __init__(
        self,
        name: str = "webhook",
        url: str = "",
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, "webhook", config or {})
        self.url = (config.get("url") if config else None) or url
        self.headers = (config.get("headers") if config else None) or headers or {}

    def trigger(self, prediction: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.url:
            return {"status": "skipped", "reason": "no webhook url"}
        payload = {
            "prediction": getattr(prediction, "to_dict", lambda: {"label": str(prediction)})(),
            "context": context or {},
        }
        try:
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **self.headers},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as res:
                return {"status": "success", "actuator": "webhook", "code": res.status}
        except Exception as exc:
            return {"status": "error", "actuator": "webhook", "error": str(exc)}


def create_actuator(config: Dict[str, Any]) -> Actuator:
    """Factory to instantiate the appropriate Actuator from configuration."""
    act_type = str(config.get("type") or config.get("actuator_type") or "home_assistant").lower()
    name = str(config.get("name") or act_type)
    if act_type in ("home_assistant", "ha"):
        return HomeAssistantActuator(
            name=name,
            url=config.get("url", "http://localhost:8123"),
            token=config.get("token", ""),
            entity_id=config.get("entity_id", ""),
            positive_action=config.get("positive_action", "homeassistant.turn_on"),
            negative_action=config.get("negative_action", "homeassistant.turn_off"),
            trigger_labels=config.get("trigger_labels"),
            config=config,
        )
    if act_type in ("device", "device_action", "motor", "robothand", "sensehat_matrix"):
        action_kind = config.get("action_kind") or act_type
        return DeviceActuator(name=name, action_kind=action_kind, config=config)
    if act_type == "webhook":
        return WebhookActuator(name=name, url=config.get("url", ""), headers=config.get("headers"), config=config)
    raise ValueError(f"Unknown actuator type: {act_type}")
