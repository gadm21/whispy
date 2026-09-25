"""Whispy cloud client — authenticated access to the Brain v1 API.

    import whispy
    client = whispy.Client()                      # ~/.whispy/credentials.json
    client = whispy.Client(api_key="wk_...")      # scoped automation key
    for device in client.devices():
        print(device.name, device.online)
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import Device
from ..devices.remote import RemoteDevice, _Http
from ..errors import AuthError

DEFAULT_BASE_URL = os.getenv("WHISPY_API_URL", "https://api.thothcraft.com")
_CREDENTIALS_PATH = os.path.expanduser(
    os.getenv("WHISPY_CREDENTIALS", "~/.whispy/credentials.json"))


class Client:
    """Remote client for the Brain v1 API.

    Credentials resolution order:

    1. explicit ``api_key``/``token`` argument
    2. ``WHISPY_API_KEY`` env var
    3. ``~/.whispy/credentials.json`` (written by :meth:`login`)
    """

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 token: Optional[str] = None,
                 timeout: int = 30):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        credential = api_key or token or os.getenv("WHISPY_API_KEY") \
            or self._load_stored_token()
        self._http = _Http(f"{self.base_url}", token=credential,
                           timeout=timeout)

    # -- auth ---------------------------------------------------------------
    @staticmethod
    def _load_stored_token() -> Optional[str]:
        try:
            data = json.loads(Path(_CREDENTIALS_PATH).read_text())
            return data.get("api_key") or data.get("access_token")
        except Exception:
            return None

    @classmethod
    def login(cls, email: Optional[str] = None, password: Optional[str] = None,
              base_url: Optional[str] = None, *,
              api_key: Optional[str] = None) -> "Client":
        """Authenticate and persist credentials for later sessions."""
        base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        if api_key:
            client = cls(base_url=base, api_key=api_key)
            client._verify()
            client._store({"api_key": api_key})
            return client
        if not email or not password:
            raise AuthError("login requires email+password or api_key")
        http = _Http(base)
        res = http.request("POST", "/api/token",
                           body={"username": email, "password": password})
        token = res.get("access_token")
        if not token:
            raise AuthError("login failed: no access_token in response")
        client = cls(base_url=base, token=token)
        client._store({"access_token": token})
        return client

    @staticmethod
    def _store(data: Dict[str, Any]) -> None:
        path = Path(_CREDENTIALS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                pass
        existing.update(data)
        path.write_text(json.dumps(existing, indent=2))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _verify(self) -> None:
        self.account()

    # -- account --------------------------------------------------------------
    def account(self) -> Dict[str, Any]:
        return self._http.request("GET", "/v1/account")

    # -- devices --------------------------------------------------------------
    def devices(self) -> List[RemoteDevice]:
        payload = self._http.request("GET", "/v1/devices")
        items = payload.get("devices") if isinstance(payload, dict) else payload
        return [RemoteDevice(self._http, Device.from_dict(d))
                for d in (items or [])]

    def device(self, name_or_id: str) -> RemoteDevice:
        """Resolve a device by name (``thoth-pi-a``), id, or stable_uuid."""
        for dev in self.devices():
            if name_or_id in (dev.info.name, dev.info.id, dev.info.stable_uuid):
                return dev
        raise KeyError(f"no device {name_or_id!r}; "
                       f"available: {[d.info.name for d in self.devices()]}")

    # -- models / deployments ---------------------------------------------------
    def models(self) -> List[Dict[str, Any]]:
        payload = self._http.request("GET", "/v1/models")
        return payload.get("models") or []

    def deployments(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"device_id": device_id} if device_id else None
        payload = self._http.request("GET", "/v1/deployments", params=params)
        return payload.get("deployments") or []

    def deploy(self, model_id: str, device_id: str) -> Dict[str, Any]:
        return self._http.request(
            "POST", "/v1/deployments",
            body={"model_id": model_id, "device_id": device_id})

    # -- captures ---------------------------------------------------------------
    def captures(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"device_id": device_id} if device_id else None
        payload = self._http.request("GET", "/v1/captures", params=params)
        return payload.get("captures") or []

    # -- collection control -----------------------------------------------------
    def capture_start(self, device_id: str,
                      sensors: Optional[List[str]] = None,
                      label: Optional[str] = None) -> Dict[str, Any]:
        """Start a synchronized capture on a node (remote control).

        Rides the same ``POST /v1/devices/{id}/captures`` contract the
        portal uses — Brain relays to the node over its WS tunnel.
        ``sensors=None`` captures every available sensor at its native
        rate (node-side default).
        """
        body: Dict[str, Any] = {"sensors": sensors or []}
        if label:
            body["label"] = label
        return self._http.request(
            "POST", f"/v1/devices/{device_id}/captures", body=body)

    def capture_stop(self, capture_id: str) -> Dict[str, Any]:
        return self._http.request("POST", f"/v1/captures/{capture_id}/stop")

    def capture_label(self, capture_id: str, label: str,
                      device_id: Optional[str] = None,
                      start: Optional[float] = None,
                      end: Optional[float] = None) -> Dict[str, Any]:
        """Manual label on a running/stopped capture."""
        body: Dict[str, Any] = {"label": label}
        if start is not None:
            body["start"] = start
        if end is not None:
            body["end"] = end
        if device_id:
            return self.node_api(device_id, "POST",
                                 f"/api/captures/{capture_id}/label",
                                 body=body)
        return self._http.request(
            "POST", f"/v1/captures/{capture_id}/label", body=body)

    def node_api(self, device_id: str, method: str, path: str,
                 body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic REST→WS relay to a node's local API (§3).

        ``GET /api/v1/context``, ``POST /api/captures/start``, automations,
        actuators — anything the node serves is reachable remotely without
        a VPN; the node answers over its outbound WS tunnel.
        """
        return self._http.request(
            "POST", f"/v1/nodes/{device_id}/api",
            body={"method": method.upper(), "path": path,
                  "body": body})

    # -- context ----------------------------------------------------------------
    def context(self, key: Optional[str] = None,
                entity_id: Optional[str] = None,
                active_only: bool = False) -> List[Dict[str, Any]]:
        """Current context states (``GET /v1/context/state``)."""
        params = {"key": key, "entity_id": entity_id,
                  "active_only": active_only or None}
        payload = self._http.request("GET", "/v1/context/state",
                                     params=params)
        return payload.get("states") or []

    def context_snapshot(self) -> Dict[str, Any]:
        """Entities + relationships + live states in one document."""
        return self._http.request("GET", "/v1/context/snapshot")

    def context_events(self, key: Optional[str] = None,
                       since: Optional[float] = None,
                       limit: int = 200) -> List[Dict[str, Any]]:
        """Context transitions (``GET /v1/context/events``)."""
        payload = self._http.request(
            "GET", "/v1/context/events",
            params={"key": key, "since": since, "limit": limit})
        return payload.get("events") or []

    # -- events -----------------------------------------------------------------
    def events(self, device_id: Optional[str] = None,
               kind: Optional[str] = None,
               since: Optional[str] = None,
               limit: int = 50) -> List[Dict[str, Any]]:
        """Node event feed (``GET /v1/events``)."""
        payload = self._http.request(
            "GET", "/v1/events",
            params={"device_id": device_id, "kind": kind,
                    "since": since, "limit": limit})
        return payload.get("events") or []

    def event_stream(self, device_id: Optional[str] = None,
                     kind: Optional[str] = None,
                     last_event_id: Optional[str] = None,
                     timeout: int = 0):
        """Subscribe to the live event stream — SSE, no polling.

        Yields parsed event dicts as they arrive. ``last_event_id``
        resumes after a reconnect (Brain replays rows > id). Caller
        controls reconnect policy — a simple ``for`` loop over a
        reconnecting generator covers drop-reconnect.
        """
        import urllib.request
        params = {"device_id": device_id, "kind": kind,
                  "token": self._http.token}
        qs = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}/v1/events/stream?{qs}"
        headers = {}
        if self._http.token:
            headers["Authorization"] = f"Bearer {self._http.token}"
        if last_event_id:
            headers["Last-Event-ID"] = str(last_event_id)
        req = urllib.request.Request(url, headers=headers)
        res = urllib.request.urlopen(
            req, timeout=timeout or None)
        buf = b""
        event: Dict[str, Any] = {}
        try:
            while True:
                chunk = res.read(4096)
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if not line:
                        if event:
                            yield event
                            event = {}
                        continue
                    if line.startswith(b":"):
                        continue                    # heartbeat comment
                    if line.startswith(b"id:"):
                        event["id"] = line[2:].strip().decode()
                    elif line.startswith(b"event:"):
                        event["kind"] = line[6:].strip().decode()
                    elif line.startswith(b"data:"):
                        try:
                            event.update(json.loads(
                                line[5:].strip().decode()))
                        except ValueError:
                            event["data_raw"] = line[5:].strip().decode()
        finally:
            res.close()

    # -- automation rules (server-side, §17) --------------------------------------
    def rules(self) -> List[Dict[str, Any]]:
        payload = self._http.request("GET", "/v1/automation/rules")
        return payload.get("rules") or []

    def add_rule(self, name: str, when: Dict[str, Any],
                 then: Dict[str, Any],
                 cooldown_s: float = 0.0) -> Dict[str, Any]:
        """Create an edge-triggered context rule, e.g.::

            client.add_rule("occupied→lights",
                when={"key": "prediction", "equals": "occupied"},
                then={"actuator_id": "ha-light", "operation": "turn_on",
                      "device_id": "..."})
        """
        return self._http.request("POST", "/v1/automation/rules", body={
            "name": name, "when": when, "then": then,
            "cooldown_s": cooldown_s})

    def delete_rule(self, name: str) -> Dict[str, Any]:
        return self._http.request(
            "DELETE", f"/v1/automation/rules/{name}")

    # -- webhook subscriptions ----------------------------------------------------
    def subscribe_webhook(self, url: str,
                          kinds: Optional[List[str]] = None,
                          device_id: Optional[str] = None) -> Dict[str, Any]:
        """POST every matching event to ``url`` (signed, retried)."""
        payload = self._http.request("POST", "/v1/subscriptions", body={
            "url": url, "kinds": kinds or [], "device_id": device_id})
        return payload.get("subscription") or payload

    # -- face assets (eigenface store) ------------------------------------------
    def face_gallery(self, ttl_s: float = 300.0):
        """TTL-cached view of the enrolled gallery + PCA basis bytes."""
        from .faces import FaceGallery
        return FaceGallery(
            fetcher=lambda: self._http.request("GET", "/v1/faces/gallery"),
            basis_fetcher=lambda: self._http.request(
                "GET", "/v1/faces/basis", raw=True),
            ttl_s=ttl_s)

    def fit_face_basis(self, images: List[bytes],
                       name: str = "default", image_size: int = 64,
                       n_components: int = 40) -> Dict[str, Any]:
        """Fit a PCA basis from raw image bytes (>=2) server-side."""
        return self._http.request("POST", "/v1/faces/basis", body={
            "name": name, "image_size": image_size,
            "n_components": n_components,
            "images": [base64.b64encode(b).decode() for b in images]})

    def enroll_face(self, name: str, photo: bytes,
                    mime: str = "image/jpeg") -> Dict[str, Any]:
        """Enroll one face photo as a person asset (stored projection)."""
        return self._http.request("POST", "/v1/faces/persons", body={
            "name": name,
            "photo_b64": base64.b64encode(photo).decode(),
            "photo_mime": mime})

    def persons(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"name": name} if name else None
        payload = self._http.request("GET", "/v1/faces/persons",
                                     params=params)
        return payload.get("persons") or []

    def delete_face(self, asset_id: str) -> Dict[str, Any]:
        """Delete one enrolled person asset by id."""
        return self._http.request("DELETE",
                                  f"/v1/faces/persons/{asset_id}")


__all__ = ["Client", "DEFAULT_BASE_URL"]  # + face_* methods above
