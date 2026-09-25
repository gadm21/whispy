"""whispy-actuator-homeassistant — HA entities as device actuators.

Discovers the entities listed in ``HA_ENTITIES`` (comma-separated
``light.desk,switch.fan``) on the Home Assistant instance at ``HA_URL``
authenticated by ``HA_TOKEN``. Each entity becomes an actuator of kind
``light``/``switch`` with operations::

    turn_on                        → light.turn_on (optional brightness,
                                     rgb_color, transition params)
    turn_off                       → light.turn_off
    toggle                         → light.toggle
    set_color  {rgb_color:[r,g,b]} → turn_on + color

Config resolution order (per entity): ``connect`` config →
``HA_URL``/``HA_TOKEN`` env. A node hosting HA itself typically sets
``HA_URL=http://localhost:8123``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from whispy.actuators.base import (
    ActuatorAdapter, ActuatorHandle, ActuatorMeta)
from whispy.contracts import (
    ActionResult, ActionStatus, ActuatorCommand, ActuatorDescriptor)


def _cfg(key: str, config: Dict[str, Any], default: str = "") -> str:
    return str(config.get(key) or os.getenv(key.upper()) or default)


class _EntityHandle(ActuatorHandle):
    def __init__(self, descriptor: ActuatorDescriptor, base_url: str,
                 token: str, entity_id: str):
        self._desc = descriptor
        self._base = base_url.rstrip("/")
        self._token = token
        self._entity = entity_id

    @property
    def info(self) -> ActuatorDescriptor:
        return self._desc

    @property
    def descriptor(self) -> ActuatorDescriptor:
        return self._desc

    def _call(self, service: str, data: Dict[str, Any]) -> ActionResult:
        domain, svc = ("light", service) if "." not in service \
            else tuple(service.split(".", 1))
        url = f"{self._base}/api/services/{domain}/{svc}"
        payload = {"entity_id": self._entity, **data}
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                ok = 200 <= res.status < 300
                return ActionResult(
                    status=ActionStatus.SUCCEEDED if ok
                    else ActionStatus.FAILED,
                    action_type="home_assistant",
                    detail=f"HA {domain}.{svc} → HTTP {res.status}",
                    response={"entity_id": self._entity,
                              "service": f"{domain}.{svc}"})
        except urllib.error.HTTPError as exc:
            return ActionResult(
                status=ActionStatus.FAILED, action_type="home_assistant",
                detail=f"HA {domain}.{svc} → HTTP {exc.code}",
                response={"entity_id": self._entity, "status": exc.code})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="home_assistant",
                                detail=str(exc))

    def execute(self, command: ActuatorCommand) -> ActionResult:
        started = time.time()
        op, params = command.operation, dict(command.params or {})
        if op == "set_color":
            color = params.pop("rgb_color", params.pop("color", None))
            if color is None:
                result = ActionResult(
                    status=ActionStatus.FAILED,
                    action_type="home_assistant",
                    detail="set_color requires rgb_color [r,g,b]")
            else:
                params["rgb_color"] = [int(c) for c in list(color)[:3]]
                result = self._call("turn_on", params)
        elif op in ("turn_on", "turn_off", "toggle"):
            result = self._call(op, params)
        else:
            result = ActionResult(
                status=ActionStatus.UNSUPPORTED,
                action_type="home_assistant",
                detail=f"unknown operation {op!r}")
        result.started_at = started
        result.finished_at = time.time()
        return result


class HomeAssistantAdapter(ActuatorAdapter):
    """Exposes HA entities from ``HA_ENTITIES`` as device actuators."""

    def __init__(self):
        self._handles: List[_EntityHandle] = []

    def metadata(self) -> ActuatorMeta:
        return ActuatorMeta(
            name="home-assistant", version="0.1.0",
            kinds=("light", "switch"),
            description="Home Assistant entities (HA_URL/HA_TOKEN/HA_ENTITIES)",
            config_schema={"type": "object", "properties": {
                "ha_url": {"type": "string"}, "ha_token": {"type": "string"},
                "ha_entities": {"type": "string"}}},
            maintainer="thothcraft")

    def _entities(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        base = _cfg("ha_url", cfg).rstrip("/")
        token = _cfg("ha_token", cfg)
        raw = _cfg("ha_entities", cfg)
        if not base or not raw:
            return []
        return [(e.strip(), base, token) for e in raw.split(",")
                if e.strip()]

    def discover(self) -> List[ActuatorDescriptor]:
        out = []
        for entity, _base, _token in self._entities():
            kind = entity.split(".", 1)[0]
            out.append(ActuatorDescriptor(
                id=ActuatorDescriptor.make_id(kind, f"ha:{entity}"),
                kind=kind, adapter="home-assistant", name=entity,
                hardware_id=f"ha:{entity}",
                operations=["turn_on", "turn_off", "toggle", "set_color"]
                if kind == "light" else ["turn_on", "turn_off", "toggle"],
                stable=True,
                metadata={"entity_id": entity}))
        return out

    def connect(self, descriptor: ActuatorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _EntityHandle:
        entity = str(descriptor.metadata.get("entity_id")
                   or descriptor.name or "")
        base = _cfg("ha_url", dict(config or {}))
        token = _cfg("ha_token", dict(config or {}))
        if not base or not token or not entity:
            raise RuntimeError(
                "home-assistant: need HA_URL, HA_TOKEN and an entity")
        handle = _EntityHandle(descriptor, base, token, entity)
        self._handles.append(handle)
        return handle

    def close(self) -> None:
        self._handles.clear()


__all__ = ["HomeAssistantAdapter"]
