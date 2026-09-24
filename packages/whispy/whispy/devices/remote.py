"""Remote device — a Thoth node reached through Brain (§21, §48).

    import whispy
    client = whispy.Client()
    pi = client.device("thoth-pi-a")
    radar = pi.sensor("radar")
    for sample in radar.stream(max_samples=10):
        ...

The Pi needs no inbound public port — Brain brokers the authorized
stream over the device's outbound channel.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from ..contracts import (
    ActionResult, ActuatorCommand, ActuatorDescriptor, Device, Sensor,
    SensorSample,
)
from ..errors import APIError, AuthError, EntitlementError, NotFoundError
from .base import DeviceHandle, SensorHandle


class _Http:
    def __init__(self, base_url: str, token: Optional[str] = None,
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str,
                params: Optional[dict] = None,
                body: Optional[dict] = None) -> Any:
        url = self.base_url + path
        if params:
            qs = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
            if qs:
                url += "?" + qs
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                raw = res.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            if exc.code == 401:
                raise AuthError(f"HTTP 401 on {path}: {detail}") from exc
            if exc.code == 403:
                raise EntitlementError(f"HTTP 403 on {path}: {detail}") from exc
            if exc.code == 404:
                raise NotFoundError(f"HTTP 404 on {path}: {detail}") from exc
            raise APIError(f"HTTP {exc.code} on {path}: {detail}",
                           status_code=exc.code) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise APIError(f"connection error on {path}: {exc}") from exc


class _RemoteSensorHandle(SensorHandle):
    def __init__(self, http: _Http, device_id: str, sensor: Sensor):
        self._http = http
        self._device_id = device_id
        self._info = sensor

    @property
    def info(self) -> Sensor:
        return self._info

    def stream(self, max_samples: Optional[int] = None,
               poll_s: float = 0.25) -> Iterator[SensorSample]:
        """Authorized sample stream via Brain's stream endpoint.

        Brain authenticates the caller, verifies ``sensor:stream`` scope
        and device ownership, then relays samples from the device channel.
        """
        cursor = ""
        count = 0
        while max_samples is None or count < max_samples:
            body = self._http.request(
                "GET",
                f"/v1/devices/{self._device_id}/streams/{self._info.id}",
                params={"cursor": cursor})
            cursor = str(body.get("cursor") or cursor)
            for raw in body.get("samples") or []:
                yield SensorSample.from_dict(raw)
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
            if not body.get("samples"):
                if body.get("state") in ("disconnected", "error"):
                    raise APIError(
                        f"remote stream {self._info.id}: {body.get('state')}")
                time.sleep(poll_s)


class _RemoteActuatorHandle:
    """Executes commands on an actuator through Brain's device channel."""

    def __init__(self, http: _Http, device_id: str,
                 descriptor: ActuatorDescriptor):
        self._http = http
        self._device_id = device_id
        self._info = descriptor

    @property
    def info(self) -> ActuatorDescriptor:
        return self._info

    @property
    def descriptor(self) -> ActuatorDescriptor:
        return self._info

    def execute(self, command) -> ActionResult:
        cmd = command if isinstance(command, ActuatorCommand) \
            else ActuatorCommand.from_dict(command)
        body = self._http.request(
            "POST",
            f"/v1/devices/{self._device_id}/actuators/{self._info.id}/actions",
            body=cmd.to_dict())
        return ActionResult.from_dict(body)

    def supports(self, operation: str) -> bool:
        return operation in (self._info.operations or [])

    def close(self) -> None:
        pass


class RemoteDevice(DeviceHandle):
    """A Thoth node accessed through the Brain v1 API."""

    def __init__(self, http: _Http, device: Device):
        self._http = http
        self._info = device

    @property
    def info(self) -> Device:
        return self._info

    def refresh(self) -> Device:
        self._info = Device.from_dict(
            self._http.request("GET", f"/v1/devices/{self._info.id}"))
        return self._info

    def sensors(self) -> List[Sensor]:
        payload = self._http.request(
            "GET", f"/v1/devices/{self._info.id}/sensors")
        items = payload.get("sensors") if isinstance(payload, dict) else payload
        return [Sensor.from_dict(s) for s in (items or [])]

    def sensor(self, sensor_id_or_type: str) -> SensorHandle:
        matches = [s for s in self.sensors()
                   if s.id == sensor_id_or_type
                   or s.metadata.get("name") == sensor_id_or_type]
        if not matches:
            matches = [s for s in self.sensors()
                       if s.type == sensor_id_or_type]
        if len(matches) == 1:
            return _RemoteSensorHandle(self._http, self._info.id, matches[0])
        if len(matches) > 1:
            raise KeyError(
                f"ambiguous sensor {sensor_id_or_type!r} on device "
                f"{self._info.name}: {[s.id for s in matches]}")
        raise KeyError(
            f"no sensor {sensor_id_or_type!r} on device {self._info.name}")

    def actuators(self) -> List[ActuatorDescriptor]:
        payload = self._http.request(
            "GET", f"/v1/devices/{self._info.id}/actuators")
        items = payload.get("actuators") if isinstance(payload, dict) else payload
        return [ActuatorDescriptor.from_dict(a) for a in (items or [])]

    def actuator(self, actuator_id_or_kind: str) -> _RemoteActuatorHandle:
        descriptors = self.actuators()
        for desc in descriptors:
            if actuator_id_or_kind in (desc.id, desc.name) \
                    and (desc.id or desc.name):
                return _RemoteActuatorHandle(self._http, self._info.id, desc)
        matches = [d for d in descriptors if d.kind == actuator_id_or_kind]
        if len(matches) == 1:
            return _RemoteActuatorHandle(self._http, self._info.id, matches[0])
        if len(matches) > 1:
            raise KeyError(
                f"ambiguous actuator {actuator_id_or_kind!r} on device "
                f"{self._info.name}: {[d.id for d in matches]}")
        raise KeyError(
            f"no actuator {actuator_id_or_kind!r} on device "
            f"{self._info.name}")

    def captures(self) -> List[Dict[str, Any]]:
        payload = self._http.request(
            "GET", f"/v1/devices/{self._info.id}/captures")
        return payload.get("captures") or []

    def start_capture(self, sensors: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._http.request(
            "POST", f"/v1/devices/{self._info.id}/captures",
            body={"sensors": sensors or []})

    def stop_capture(self, capture_id: str) -> Dict[str, Any]:
        return self._http.request(
            "POST", f"/v1/captures/{capture_id}/stop")

    def predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        payload = self._http.request(
            "GET", f"/v1/devices/{self._info.id}/predictions",
            params={"limit": limit})
        return payload.get("predictions") or []


__all__ = ["RemoteDevice"]
