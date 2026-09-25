"""Local device — sensors/actuators on this machine or a LAN Thoth node.

Two modes:

- **In-process** (``whispy.local()`` with no host): enumerate adapters
  installed on this machine and stream directly from hardware.
- **LAN node** (``whispy.lan("rpi1.local", token=…)``): talk to a Thoth
  daemon's authenticated local API over HTTP.

Sensor inventory is descriptor-based: each adapter's ``discover()``
returns :class:`SensorDescriptor` objects representing *physical*
instances (``camera-f91a``), not modalities. ``sensor("camera")``
resolves only when exactly one camera is present.
"""

from __future__ import annotations

import json
import platform
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Iterator, List, Optional

from ..contracts import (
    ActionResult, ActuatorCommand, ActuatorDescriptor, Device, Sensor,
    SensorDescriptor, SensorSample,
)
from ..errors import (
    APIError, AmbiguousSourceError, SourceNotFoundError,
)
from ..sensors.base import (
    SensorAdapter, SensorDriver, SensorDriverAdapter, all_adapters,
)
from .base import DeviceHandle, SensorHandle


class _LocalSensorHandle(SensorHandle):
    """Lazily connects an adapter to one descriptor on first stream."""

    def __init__(self, adapter: SensorAdapter, descriptor: SensorDescriptor,
                 device_id: str, config: Optional[Dict[str, Any]] = None):
        self._adapter = adapter
        self._descriptor = descriptor
        self._device_id = device_id
        self._config = dict(config or {})
        self._inner = None

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._descriptor

    @property
    def info(self) -> Sensor:
        return self._descriptor.to_sensor()

    def _connect(self):
        if self._inner is None:
            self._inner = self._adapter.connect(self._descriptor, self._config)
        return self._inner

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        handle = self._connect()
        count = 0
        for sample in handle.stream():
            if sample.device_id in ("", "local", "fixture-device"):
                sample.device_id = self._device_id
            if not sample.sensor_id:
                sample.sensor_id = self._descriptor.id
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                return

    def close(self) -> None:
        if self._inner is not None and hasattr(self._inner, "close"):
            try:
                self._inner.close()
            except Exception:
                pass
            self._inner = None


class _LocalActuatorHandle:
    """Lazily connects an actuator adapter to one descriptor."""

    def __init__(self, adapter, descriptor: ActuatorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._adapter = adapter
        self._descriptor = descriptor
        self._config = dict(config or {})
        self._inner = None

    @property
    def descriptor(self) -> ActuatorDescriptor:
        return self._descriptor

    @property
    def info(self) -> ActuatorDescriptor:
        return self._descriptor

    def _connect(self):
        if self._inner is None:
            self._inner = self._adapter.connect(self._descriptor, self._config)
        return self._inner

    def execute(self, command) -> ActionResult:
        cmd = command if isinstance(command, ActuatorCommand) \
            else ActuatorCommand.from_dict(command)
        return self._connect().execute(cmd)

    def supports(self, operation: str) -> bool:
        return operation in (self._descriptor.operations or [])

    def close(self) -> None:
        if self._inner is not None and hasattr(self._inner, "close"):
            try:
                self._inner.close()
            except Exception:
                pass
            self._inner = None


class LocalDevice(DeviceHandle):
    """Sensors/actuators on this machine via installed Whispy plugins.

    ``drivers`` accepts legacy ``SensorDriver`` instances/classes (wrapped
    in :class:`SensorDriverAdapter`); ``adapters`` accepts
    ``SensorAdapter`` instances; ``actuator_adapters`` accepts
    ``ActuatorAdapter`` instances. When all are omitted, installed
    plugins are auto-detected.
    """

    def __init__(self, device_id: Optional[str] = None,
                 drivers: Optional[Dict[str, Any]] = None,
                 adapters: Optional[Dict[str, SensorAdapter]] = None,
                 actuator_adapters: Optional[Dict[str, Any]] = None):
        self._device_id = device_id or self._stable_id()
        self._adapters: Dict[str, SensorAdapter] = {}
        self._act_adapters: Dict[str, Any] = dict(actuator_adapters or {})
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._handles: List[Any] = []
        self._desc_cache: Optional[List[SensorDescriptor]] = None
        self._act_cache: Optional[List[ActuatorDescriptor]] = None
        if drivers is None and adapters is None:
            self._autodetect()
        else:
            for name, drv in (drivers or {}).items():
                self._adapters[name] = drv if isinstance(drv, SensorAdapter) \
                    else SensorDriverAdapter(drv, name=name)
            for name, adp in (adapters or {}).items():
                self._adapters[name] = adp
        if actuator_adapters is None and drivers is None and adapters is None:
            self._autodetect_actuators()

    @staticmethod
    def _stable_id() -> str:
        return f"local-{uuid.getnode():012x}"

    def _autodetect(self) -> None:
        for name, adapter in all_adapters().items():
            try:
                found = adapter.discover()
            except Exception:
                continue
            if found:
                self._adapters[name] = adapter

    def _autodetect_actuators(self) -> None:
        from ..actuators.base import installed_actuator_adapters
        for name, adapter in installed_actuator_adapters().items():
            try:
                found = adapter.discover()
            except Exception:
                continue
            if found:
                self._act_adapters[name] = adapter

    def open(self, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> "LocalDevice":
        """Store per-adapter configs; hardware opens lazily on connect."""
        self._configs.update(configs or {})
        return self

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        for adapter in list(self._adapters.values()) + \
                list(self._act_adapters.values()):
            try:
                adapter.close()
            except Exception:
                pass

    @property
    def info(self) -> Device:
        return Device(
            id=self._device_id,
            stable_uuid=self._device_id,
            name=platform.node() or "localhost",
            platform=platform.system().lower(),
            architecture=platform.machine(),
            online=True,
            sensors=self.sensors(),
        )

    # -- sensors ---------------------------------------------------------------
    def _discover_sensors(self) -> List[SensorDescriptor]:
        out: List[SensorDescriptor] = []
        for name, adapter in self._adapters.items():
            try:
                for desc in adapter.discover():
                    if not desc.adapter:
                        desc.adapter = name
                    out.append(desc)
            except Exception:
                continue
        return out

    def sensor_descriptors(self) -> List[SensorDescriptor]:
        """Physical sensor inventory — cached.

        ``discover()`` probes real hardware; on some drivers (DirectShow)
        probing a capture index steals or blacks out a graph another handle
        is streaming from. The daemon calls this per heartbeat, so the
        result is cached and only recomputed on ``refresh()`` or when
        ``sensor()`` can't resolve an id.
        """
        if self._desc_cache is None:
            self._desc_cache = self._discover_sensors()
        return self._desc_cache

    def refresh(self) -> "LocalDevice":
        """Drop cached descriptors; next call re-probes hardware."""
        self._desc_cache = None
        self._act_cache = None
        return self

    def sensors(self) -> List[Sensor]:
        return [d.to_sensor() for d in self.sensor_descriptors()]

    def sensor(self, sensor_id_or_type: str) -> SensorHandle:
        """Resolve by descriptor id, name, adapter name, or modality.

        A modality (``"camera"``) resolves only when exactly one sensor
        matches — otherwise the error lists the candidate ids so the
        caller can pick a stable id.
        """
        key = sensor_id_or_type
        descriptors = self.sensor_descriptors()

        for desc in descriptors:
            if key in (desc.id, desc.name) and (desc.id or desc.name):
                return self._sensor_handle(desc)

        matches = [d for d in descriptors if d.modality == key]
        if len(matches) == 1:
            return self._sensor_handle(matches[0])
        if len(matches) > 1:
            raise AmbiguousSourceError(key, [d.id for d in matches])

        # Unknown id — maybe hardware appeared since the cached probe.
        if self._desc_cache is not None:
            self._desc_cache = self._discover_sensors()
            for desc in self._desc_cache:
                if key in (desc.id, desc.name) and (desc.id or desc.name):
                    return self._sensor_handle(desc)

        for name, adapter in self._adapters.items():
            if name == key:
                try:
                    found = adapter.discover()
                except Exception:
                    found = []
                if found:
                    return self._sensor_handle(found[0])
        raise SourceNotFoundError(key, [d.id for d in descriptors])

    def _sensor_handle(self, desc: SensorDescriptor) -> _LocalSensorHandle:
        adapter = self._adapters.get(desc.adapter) or \
            self._adapters.get(desc.adapter or "")
        if adapter is None:
            # Descriptor came from an adapter keyed differently; find it.
            for name, candidate in self._adapters.items():
                try:
                    if any(d.id == desc.id for d in candidate.discover()):
                        adapter = candidate
                        break
                except Exception:
                    continue
        if adapter is None:
            raise KeyError(f"no adapter for sensor {desc.id!r}")
        handle = _LocalSensorHandle(
            adapter, desc, self._device_id,
            config=self._configs.get(desc.adapter))
        self._handles.append(handle)
        return handle

    # -- actuators -------------------------------------------------------------
    def actuators(self) -> List[ActuatorDescriptor]:
        """Actuator inventory — cached like sensor_descriptors()."""
        if self._act_cache is None:
            out: List[ActuatorDescriptor] = []
            for name, adapter in self._act_adapters.items():
                try:
                    for desc in adapter.discover():
                        if not desc.adapter:
                            desc.adapter = name
                        out.append(desc)
                except Exception:
                    continue
            self._act_cache = out
        return self._act_cache

    def actuator(self, actuator_id_or_kind: str) -> _LocalActuatorHandle:
        key = actuator_id_or_kind
        descriptors = self.actuators()
        for desc in descriptors:
            if key in (desc.id, desc.name) and (desc.id or desc.name):
                return self._actuator_handle(desc)
        matches = [d for d in descriptors if d.kind == key]
        if len(matches) == 1:
            return self._actuator_handle(matches[0])
        if len(matches) > 1:
            raise KeyError(
                f"ambiguous actuator {key!r}: {[d.id for d in matches]}; "
                f"use a stable id")
        raise KeyError(f"no local actuator {key!r}; "
                       f"available: {[d.id for d in descriptors]}")

    def _actuator_handle(self, desc: ActuatorDescriptor) -> _LocalActuatorHandle:
        adapter = self._act_adapters.get(desc.adapter)
        if adapter is None:
            for name, candidate in self._act_adapters.items():
                try:
                    if any(d.id == desc.id for d in candidate.discover()):
                        adapter = candidate
                        break
                except Exception:
                    continue
        if adapter is None:
            raise KeyError(f"no adapter for actuator {desc.id!r}")
        handle = _LocalActuatorHandle(
            adapter, desc, config=self._configs.get(desc.adapter))
        self._handles.append(handle)
        return handle


class _Http:
    """Minimal JSON client for a LAN Thoth node's local API."""

    def __init__(self, base_url: str, token: Optional[str] = None,
                 timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def get_json(self, path: str, params: Optional[dict] = None) -> Any:
        url = self.base_url + path
        if params:
            qs = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
            if qs:
                url += "?" + qs
        req = urllib.request.Request(url)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise APIError(f"local request failed: {url}: HTTP {exc.code}",
                           status_code=exc.code) from exc
        except Exception as exc:
            raise APIError(f"local request failed: {url}: {exc}") from exc

    def post_json(self, path: str, body: Optional[dict] = None) -> Any:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body or {}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise APIError(f"local request failed: {path}: HTTP {exc.code}",
                           status_code=exc.code) from exc
        except Exception as exc:
            raise APIError(f"local request failed: {path}: {exc}") from exc


class _LanSensorHandle(SensorHandle):
    """Streams samples from a LAN node's tail endpoint."""

    def __init__(self, http: _Http, sensor: Sensor):
        self._http = http
        self._info = sensor

    @property
    def info(self) -> Sensor:
        return self._info

    def stream(self, max_samples: Optional[int] = None,
               poll_s: float = 0.1) -> Iterator[SensorSample]:
        # Prime the cursor: the first tail response carries the daemon's
        # whole ring buffer (minutes of history). A stream is live-only —
        # adopt the cursor and discard the backlog so consumers like
        # capture_window see only samples produced after stream start.
        body = self._http.get_json(
            f"/api/sensors/{self._info.id}/tail", {"cursor": ""})
        cursor = str(body.get("cursor") or "")
        count = 0
        while max_samples is None or count < max_samples:
            body = self._http.get_json(
                f"/api/sensors/{self._info.id}/tail", {"cursor": cursor})
            cursor = str(body.get("cursor") or cursor)
            for raw in body.get("samples") or []:
                yield SensorSample.from_dict(raw)
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
            if not body.get("samples"):
                time.sleep(poll_s)


class _LanActuatorHandle:
    """Executes commands on a LAN node's actuator via the local API."""

    def __init__(self, http: _Http, descriptor: ActuatorDescriptor):
        self._http = http
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
        body = self._http.post_json(
            f"/api/actuators/{self._info.id}/actions", cmd.to_dict())
        return ActionResult.from_dict(body)

    def supports(self, operation: str) -> bool:
        return operation in (self._info.operations or [])

    def close(self) -> None:
        pass


class LanDevice(DeviceHandle):
    """A Thoth node reachable on the LAN via its authenticated local API."""

    def __init__(self, host: str, port: int = 5000,
                 token: Optional[str] = None, timeout: int = 15):
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        elif ":" not in host.split("://", 1)[1]:
            host = f"{host}:{port}"
        self._http = _Http(host if host.rsplit(":", 1)[-1].isdigit()
                           else f"{host}:{port}", token=token,
                           timeout=timeout)
        self.host = host

    @property
    def info(self) -> Device:
        return Device.from_dict(self._http.get_json("/api/device"))

    def sensors(self) -> List[Sensor]:
        payload = self._http.get_json("/api/sensors")
        items = payload.get("sensors") if isinstance(payload, dict) else payload
        return [Sensor.from_dict(s) for s in (items or [])]

    def sensor_descriptors(self) -> List[SensorDescriptor]:
        try:
            payload = self._http.get_json("/api/sensors",
                                          {"descriptors": "1"})
            items = payload.get("descriptors")
            if items:
                return [SensorDescriptor.from_dict(d) for d in items]
        except APIError:
            pass
        return [SensorDescriptor(
            id=s.id, modality=s.type, adapter=s.driver,
            capabilities=list(s.capabilities),
            hardware_id=str(s.metadata.get("hardware_id") or ""),
            name=str(s.metadata.get("name") or ""),
            stable=bool(s.metadata.get("stable")),
        ) for s in self.sensors()]

    def sensor(self, sensor_id_or_type: str) -> SensorHandle:
        matches = [s for s in self.sensors()
                   if s.id == sensor_id_or_type
                   or s.metadata.get("name") == sensor_id_or_type]
        if not matches:
            matches = [s for s in self.sensors()
                       if s.type == sensor_id_or_type]
        if len(matches) == 1:
            return _LanSensorHandle(self._http, matches[0])
        if len(matches) > 1:
            raise AmbiguousSourceError(
                sensor_id_or_type, [s.id for s in matches])
        raise SourceNotFoundError(sensor_id_or_type)

    def actuators(self) -> List[ActuatorDescriptor]:
        payload = self._http.get_json("/api/actuators")
        items = payload.get("actuators") if isinstance(payload, dict) else payload
        return [ActuatorDescriptor.from_dict(a) for a in (items or [])]

    def actuator(self, actuator_id_or_kind: str) -> _LanActuatorHandle:
        descriptors = self.actuators()
        for desc in descriptors:
            if actuator_id_or_kind in (desc.id, desc.name) \
                    and (desc.id or desc.name):
                return _LanActuatorHandle(self._http, desc)
        matches = [d for d in descriptors if d.kind == actuator_id_or_kind]
        if len(matches) == 1:
            return _LanActuatorHandle(self._http, matches[0])
        if len(matches) > 1:
            raise KeyError(
                f"ambiguous actuator {actuator_id_or_kind!r} on "
                f"{self.host}: {[d.id for d in matches]}")
        raise KeyError(
            f"no actuator {actuator_id_or_kind!r} on {self.host}")

    def status(self) -> Dict[str, Any]:
        return self._http.get_json("/api/status")

    def captures(self) -> List[Dict[str, Any]]:
        payload = self._http.get_json("/api/captures")
        return payload.get("captures") or []

    def predict(self, label: str, confidence: float = 1.0) -> Dict[str, Any]:
        return self._http.post_json("/api/internal/prediction", {
            "class": label, "confidence": confidence})


def local(host: Optional[str] = None, port: int = 5000,
          token: Optional[str] = None, **kwargs: Any) -> DeviceHandle:
    """Connect to local sensing.

    ``local()`` → in-process adapters on this machine.
    ``local("rpi1.local")`` → LAN node via authenticated local API.
    """
    if host is None:
        return LocalDevice(**kwargs).open()
    return LanDevice(host, port=port, token=token)


def lan(host: str, port: int = 5000, token: Optional[str] = None,
        timeout: int = 15) -> LanDevice:
    """Connect to a Thoth node on the LAN by hostname/IP.

    ``whispy.lan("rpi1.local", token=…)`` is identical to
    ``whispy.local("rpi1.local", …)`` but reads explicitly.
    """
    return LanDevice(host, port=port, token=token, timeout=timeout)


__all__ = ["LocalDevice", "LanDevice", "local", "lan"]
