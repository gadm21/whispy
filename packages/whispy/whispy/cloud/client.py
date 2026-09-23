"""Whispy cloud client — authenticated access to the Brain v1 API.

    import whispy
    client = whispy.Client()                      # ~/.whispy/credentials.json
    client = whispy.Client(api_key="wk_...")      # scoped automation key
    for device in client.devices():
        print(device.name, device.online)
"""

from __future__ import annotations

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


__all__ = ["Client", "DEFAULT_BASE_URL"]
