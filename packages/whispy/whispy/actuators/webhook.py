"""Webhook actuator — signed HTTP POST to external endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.request
from typing import Any, Dict

from ..contracts import Action, ActionResult, ActionStatus, Prediction
from .base import Actuator


class WebhookActuator(Actuator):
    """POSTs a structured payload to a webhook endpoint.

    Config::

        {
            "url": "https://hooks.slack.com/services/...",
            "method": "POST",
            "headers": {"X-Custom-Auth": "..."},
            "body": {"text": "Alert at {device_id}: {label}"},
            "hmac_secret": "optional-sha256-signing-key"
        }

    ``body`` values may reference ``{label}``, ``{confidence}``,
    ``{device_id}``, ``{timestamp}`` — formatted per prediction.
    ``succeeded`` requires an HTTP 2xx response.
    """

    actuator_type = "webhook"

    def execute(self, action: Action, prediction: Prediction) -> ActionResult:
        cfg = {**self.config, **(action.config or {})}
        url = str(cfg.get("url") or "")
        if not url:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type,
                                detail="webhook action requires 'url'")
        method = str(cfg.get("method") or "POST").upper()
        headers = {"Content-Type": "application/json",
                   **{str(k): str(v) for k, v in (cfg.get("headers") or {}).items()}}

        fmt = {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "device_id": prediction.device_id,
            "timestamp": prediction.timestamp,
        }
        body_template = cfg.get("body") or {
            "text": "{label} ({confidence:.2f}) on {device_id}",
            "prediction": prediction.to_dict(),
        }
        body = self._render(body_template, fmt)
        data = json.dumps(body).encode("utf-8")

        secret = cfg.get("hmac_secret")
        if secret:
            headers["X-Thoth-Signature"] = hmac.new(
                str(secret).encode(), data, hashlib.sha256).hexdigest()

        try:
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method=method)
            with urllib.request.urlopen(req, timeout=action.timeout_seconds) as res:
                ok = 200 <= res.status < 300
                return ActionResult(
                    status=ActionStatus.SUCCEEDED if ok else ActionStatus.FAILED,
                    action_type=self.actuator_type,
                    detail=f"HTTP {res.status}",
                    response={"status": res.status, "url": url})
        except urllib.error.HTTPError as exc:
            return ActionResult(
                status=ActionStatus.FAILED, action_type=self.actuator_type,
                detail=f"HTTP {exc.code}: {exc.reason}",
                response={"status": exc.code, "url": url})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type=self.actuator_type, detail=str(exc))

    @staticmethod
    def _render(template: Any, fmt: Dict[str, Any]) -> Any:
        if isinstance(template, str):
            try:
                return template.format(**fmt)
            except (KeyError, IndexError, ValueError):
                return template
        if isinstance(template, dict):
            return {k: WebhookActuator._render(v, fmt) for k, v in template.items()}
        if isinstance(template, list):
            return [WebhookActuator._render(v, fmt) for v in template]
        return template


__all__ = ["WebhookActuator"]
