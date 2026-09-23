"""Remote Thoth device: identity, control, live stream, minutes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Optional

from .minutes import Minute


class Device:
    """A Thoth device — any authenticated node running the device protocol
    (Raspberry Pi, laptop via thothcraftd, or dedicated hardware)."""

    def __init__(self, http, info: dict):
        self._http = http
        self.info = info
        self.uuid = str(info.get("device_uuid") or info.get("device_id") or "")
        self.name = info.get("device_name") or info.get("name") or self.uuid
        self.online = bool(info.get("online") or info.get("is_online"))

    @property
    def hostname(self) -> str | None:
        """mDNS hostname advertised by the node, e.g. ``thoth-denver.local``."""
        hw = self.info.get("hardware_info") or {}
        if isinstance(hw, dict):
            return hw.get("hostname") or hw.get("device_hostname")
        return None

    @property
    def dashboard_url(self) -> str:
        """Direct URL of the node's local dashboard/API (port 5000)."""
        host = self.hostname or self.info.get("ip_address") or self.name
        return f"http://{host}:5000"

    # -- control ---------------------------------------------------------
    def command(self, command: str, payload: Optional[dict] = None) -> dict:
        return self._http.post_json(
            f"/api/device/{self.uuid}/commands",
            {"command": command, "payload": payload or {}})

    def start(self) -> dict:
        return self.command("start_collection")

    def stop(self) -> dict:
        return self.command("stop_collection")

    def capture_settings(self) -> dict:
        return self._http.get_json(f"/api/device/{self.uuid}/capture-settings")

    def sensors(self) -> dict:
        """Advertised sensor capabilities (camera, radar, csi, mic, env)."""
        hw = self.info.get("hardware_info") or {}
        return hw.get("sensors", hw) if isinstance(hw, dict) else {}

    # -- live ------------------------------------------------------------
    def live(self, cursor: str | None = None, poll_s: float = 1.0,
             max_items: Optional[int] = None) -> Iterator[dict]:
        """Yield live analysis chunks as they arrive (long-poll loop)."""
        seen = 0
        if poll_s <= 0:
            raise ValueError('poll_s must be positive')
        if max_items is not None and max_items <= 0:
            return
        while True:
            payload = self._http.get_json(
                f"/api/device/{self.uuid}/live-chunks", {"after": cursor})
            chunks = payload.get("chunks") or payload.get("data") or []
            for chunk in chunks:
                yield chunk
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            cursor = payload.get("cursor") or cursor
            time.sleep(poll_s)

    def stream(self, cursor: str | None = None, poll_s: float = 1.0,
               max_items: int | None = None) -> Iterator[dict]:
        """Yield live chunks, e.g. ``for chunk in device.stream(): ...``."""
        return self.live(cursor, poll_s, max_items)

    def predictions(self, minute: str | Minute | None = None) -> list:
        """Return one snapshot of model predictions; never wait for new chunks."""
        if minute is not None:
            return (minute if isinstance(minute, Minute) else self.minute(minute)).predictions
        payload = self._http.get_json(f'/api/device/{self.uuid}/live-chunks')
        from .minutes import extract_predictions
        return extract_predictions(payload)

    def deploy(self, model_or_path, wait: bool = True, *, timeout: float = 180,
               config: dict | None = None, **upload_options):
        """Deploy a Model/id, a registry name ('thothcraft/radar-occupancy-v2'),
        or upload a local .pt path (with name/classes/input_spec) first."""
        from .client import Client
        from .models import Model
        client = Client(self._http.base_url)
        client._http = self._http
        if isinstance(model_or_path, (str, Path)):
            if Path(model_or_path).exists():
                model = client.upload_model(model_or_path, **upload_options)
            else:
                model = client.resolve_model(str(model_or_path))
            model_id = model.id
        else:
            model_id = model_or_path.id if isinstance(model_or_path, Model) else int(model_or_path)
        deployment = client.deploy_model(model_id, self.uuid, config)
        return deployment.wait(timeout) if wait else deployment

    # -- placement ---------------------------------------------------------
    def place(self, space, *, x: float = 0.0, y: float = 0.0,
              rotation_deg: float = 0.0, fov_deg: float = 90.0,
              range_m: float = 8.0) -> dict:
        """Position this device inside a Space (plan coordinates, meters)."""
        space_id = space.id if hasattr(space, "id") else int(space)
        payload = self._http.put_json(
            f"/api/spaces/devices/{self.uuid}/placement",
            {"space_id": space_id, "x": x, "y": y,
             "rotation_deg": rotation_deg, "fov_deg": fov_deg,
             "range_m": range_m})
        return payload.get("placement") or payload

    def unplace(self) -> dict:
        return self._http.delete(f"/api/spaces/devices/{self.uuid}/placement")

    # -- data ------------------------------------------------------------
    def files(self) -> list:
        payload = self._http.get_json(f"/api/device/{self.uuid}/files")
        return payload.get("files") or payload.get("data") or []

    def minutes(self) -> list:
        """Distinct captured-minute ids known for this device."""
        ids = []
        for f in self.files():
            m = f.get("minute") or f.get("minute_id") or f.get("folder_name")
            if m and m not in ids:
                ids.append(m)
        return ids

    def data(self) -> Iterator[Minute]:
        """Iterate captured minutes, newest first."""
        for minute_id in sorted(self.minutes(), reverse=True):
            yield self.minute(minute_id)

    def collect(self, minutes: int = 30) -> "CollectionSession":
        """Start a timed collection session on this device."""
        return CollectionSession(self, minutes)

    def minute(self, minute_id: str) -> Minute:
        return Minute(self._http, minute_id, device_id=self.uuid)

    def __repr__(self) -> str:
        return f"Device({self.name}, uuid={self.uuid}, online={self.online})"


class CollectionSession:
    """A bounded collection: start → wait → stop → yield new minutes."""

    def __init__(self, device: Device, minutes: int):
        self.device = device
        self.minutes = minutes
        self._before: set = set()

    def __enter__(self) -> "CollectionSession":
        self._before = set(self.device.minutes())
        self.device.start()
        return self

    def __exit__(self, *exc) -> None:
        self.device.stop()

    def new_minutes(self) -> Iterator[Minute]:
        for m in sorted(set(self.device.minutes()) - self._before):
            yield self.device.minute(m)
