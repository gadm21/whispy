"""Local device — sensors attached to this machine or a LAN Thoth node.

Two modes:

- **In-process** (``whispy.local()`` with no host): enumerate drivers
  installed on this machine and stream directly from hardware.
- **LAN node** (``whispy.local("thoth-pi-a.local")``): talk to a Thoth
  daemon's authenticated local API over HTTP.
"""

from __future__ import annotations

import itertools
import json
import platform
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Iterator, List, Optional

from ..contracts import Device, Sensor, SensorSample
from ..errors import APIError
from ..sensors.base import SensorDriver, all_drivers
from .base import DeviceHandle, SensorHandle


class _LocalSensorHandle(SensorHandle):
    def __init__(self, driver: SensorDriver, sensor_id: str,
                 sensor_type: str, device_id: str):
        self._driver = driver
        self._sensor_id = sensor_id
        self._sensor_type = sensor_type
        self._device_id = device_id
        self._seq = itertools.count()

    @property
    def info(self) -> Sensor:
        meta = self._driver.metadata()
        return Sensor(
            id=self._sensor_id,
            type=self._sensor_type,
            driver=meta.name,
            driver_version=meta.version,
            online=True,
            capabilities=list(meta.modalities),
        )

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        count = 0
        for sample in self._driver.stream():
            if sample.device_id in ("", "local", "fixture-device"):
                sample.device_id = self._device_id
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                return


class LocalDevice(DeviceHandle):
    """Sensors on this machine via installed Whispy drivers."""

    def __init__(self, device_id: Optional[str] = None,
                 drivers: Optional[Dict[str, SensorDriver]] = None):
        self._device_id = device_id or self._stable_id()
        self._drivers: Dict[str, SensorDriver] = dict(drivers or {})
        self._opened: List[SensorDriver] = []
        if drivers is None:
            self._autodetect()

    @staticmethod
    def _stable_id() -> str:
        return f"local-{uuid.getnode():012x}"

    def _autodetect(self) -> None:
        for name, cls in all_drivers().items():
            try:
                driver = cls()
                found = driver.discover()
            except Exception:
                continue
            if found:
                self._drivers[name] = driver

    def open(self, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> "LocalDevice":
        for name, driver in self._drivers.items():
            try:
                driver.open((configs or {}).get(name, {}))
                self._opened.append(driver)
            except Exception:
                continue
        return self

    def close(self) -> None:
        for driver in self._opened:
            try:
                driver.close()
            except Exception:
                pass
        self._opened.clear()

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

    def _inventory(self) -> List[Dict[str, Any]]:
        """One entry per advertised sensor: id, modality, driver, handle."""
        out: List[Dict[str, Any]] = []
        for name, driver in self._drivers.items():
            try:
                meta = driver.metadata()
            except Exception:
                continue
            for modality in meta.modalities or (name,):
                out.append({
                    "id": f"{modality}-0",
                    "modality": modality,
                    "driver_name": name,
                    "driver": driver,
                    "meta": meta,
                })
        return out

    def sensors(self) -> List[Sensor]:
        out: List[Sensor] = []
        for item in self._inventory():
            meta = item["meta"]
            out.append(Sensor(
                id=item["id"], type=item["modality"],
                driver=meta.name, driver_version=meta.version,
                online=True, capabilities=list(meta.modalities)))
        return out

    def sensor(self, sensor_id_or_type: str) -> SensorHandle:
        # Accept the exact inventory id (``system-0``), a modality
        # (``system``), or a driver name — all resolve to a live handle.
        for item in self._inventory():
            if sensor_id_or_type in (item["id"], item["modality"],
                                     item["driver_name"]):
                return _LocalSensorHandle(
                    item["driver"], item["id"],
                    item["modality"], self._device_id)
        raise KeyError(f"no local sensor {sensor_id_or_type!r}; "
                       f"available: {[s.id for s in self.sensors()]}")


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
        cursor = ""
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


class LanDevice(DeviceHandle):
    """A Thoth node reachable on the LAN via its authenticated local API."""

    def __init__(self, host: str, port: int = 5000,
                 token: Optional[str] = None, timeout: int = 15):
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        self._http = _Http(f"{host}:{port}", token=token, timeout=timeout)
        self.host = host

    @property
    def info(self) -> Device:
        return Device.from_dict(self._http.get_json("/api/device"))

    def sensors(self) -> List[Sensor]:
        payload = self._http.get_json("/api/sensors")
        items = payload.get("sensors") if isinstance(payload, dict) else payload
        return [Sensor.from_dict(s) for s in (items or [])]

    def sensor(self, sensor_id_or_type: str) -> SensorHandle:
        for s in self.sensors():
            if s.id == sensor_id_or_type or s.type == sensor_id_or_type:
                return _LanSensorHandle(self._http, s)
        raise KeyError(f"no sensor {sensor_id_or_type!r} on {self.host}")

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

    ``local()`` → in-process drivers on this machine (auto-opened).
    ``local("thoth-pi-a.local")`` → LAN node via authenticated local API.
    """
    if host is None:
        return LocalDevice(**kwargs).open()
    return LanDevice(host, port=port, token=token)


__all__ = ["LocalDevice", "LanDevice", "local"]
