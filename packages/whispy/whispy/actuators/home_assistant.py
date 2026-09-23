"""Home Assistant actuator — REST service calls with real confirmation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ..contracts import Action, ActionResult, ActionStatus, Prediction
from .base import Actuator


class HomeAssistantActuator(Actuator):
    """Calls a Home Assistant ``/api/services/<domain>/<service>`` endpoint.

    Config::

        {
            "base_url": "http://homeassistant.local:8123",
            "token": "<long-lived token>",
            "entity_id": "light.office_desk",
            "action": "turn_on",                 # or full "light.turn_on"
            "data": {"brightness": 255}          # extra service data
        }

    ``succeeded`` requires an HTTP 2xx from Home Assistant — the service
    call was accepted by the provider.
    """

    actuator_type = "home_assistant"

    def execute(self, action: Action, prediction: Prediction) -> ActionResult:
        cfg = {**self.config, **(action.config or {})}
        base_url = str(cfg.get("base_url") or cfg.get("url") or "").rstrip("/")
        token = str(cfg.get("token") or "")
        entity_id = str(cfg.get("entity_id") or "")
        service = str(cfg.get("action") or cfg.get("service") or "")
        if not base_url or not service:
            return ActionResult(
                status=ActionStatus.FAILED, action_type=self.actuator_type,
                detail="home_assistant action requires base_url and action/service")

        parts = service.split(".", 1)
        if len(parts) == 2:
            domain, svc = parts
        elif entity_id and "." in entity_id:
            domain, svc = entity_id.split(".", 1)[0], service
        else:
            domain, svc = "homeassistant", service

        url = f"{base_url}/api/services/{domain}/{svc}"
        payload: Dict[str, Any] = dict(cfg.get("data") or {})
        if entity_id:
            payload["entity_id"] = entity_id
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=action.timeout_seconds) as res:
                body = res.read().decode("utf-8", errors="replace")[:500]
                ok = 200 <= res.status < 300
                return ActionResult(
                    status=ActionStatus.SUCCEEDED if ok else ActionStatus.FAILED,
                    action_type=self.actuator_type,
                    detail=f"HTTP {res.status} from {domain}.{svc}",
                    response={"status": res.status, "body": body,
                              "entity_id": entity_id, "service": f"{domain}.{svc}"})
        except urllib.error.HTTPError as exc:
            return ActionResult(
                status=ActionStatus.FAILED, action_type=self.actuator_type,
                detail=f"HTTP {exc.code}: {exc.reason}",
                response={"status": exc.code})
        except Exception as exc:
            return ActionResult(
                status=ActionStatus.FAILED, action_type=self.actuator_type,
                detail=str(exc))


__all__ = ["HomeAssistantActuator"]
