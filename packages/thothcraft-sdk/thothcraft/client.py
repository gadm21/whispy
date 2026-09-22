"""Authenticated client for a ThothCraft Brain deployment.

Works identically against the cloud Brain and a local thothcraftd
runtime — only ``base_url`` changes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .models import Model, Deployment

from .errors import APIError, AuthError, EntitlementError, NotFoundError

DEFAULT_BASE_URL = os.getenv("THOTHCRAFT_API_URL", "https://api.thothcraft.com")
_CREDENTIALS_PATH = os.path.expanduser(
    os.getenv("THOTHCRAFT_CREDENTIALS", "~/.thothcraft/credentials.json")
)


class _Http:
    """Minimal urllib JSON/bytes client with a bearer token."""

    def __init__(self, base_url: str, token: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, params: dict | None = None,
                 body: bytes | None = None, headers: dict | None = None) -> bytes:
        url = self.base_url + path
        if params:
            qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if qs:
                url += "?" + qs
        hdrs = {"Accept": "application/json"}
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            if e.code == 401:
                raise AuthError(f"HTTP 401 on {path}: {detail}") from e
            if e.code == 403:
                raise EntitlementError(f"HTTP 403 on {path}: {detail}") from e
            if e.code == 404:
                raise NotFoundError(f"HTTP 404 on {path}: {detail}") from e
            raise APIError(f"HTTP {e.code} on {path}", status_code=e.code, detail=detail) from e
        except (urllib.error.URLError, OSError) as e:
            raise APIError(f"Connection error on {path}: {e}") from e

    def get_json(self, path: str, params: dict | None = None) -> Any:
        raw = self._request("GET", path, params=params)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise APIError(f"Non-JSON response on {path}") from e

    def get_bytes(self, path: str, params: dict | None = None) -> bytes:
        return self._request("GET", path, params=params)

    def post_json(self, path: str, body: dict | None = None,
                  params: dict | None = None, form: dict | None = None) -> Any:
        if form is not None:
            data = urllib.parse.urlencode(form).encode("utf-8")
            hdrs = {"Content-Type": "application/x-www-form-urlencoded"}
        else:
            data = json.dumps(body or {}).encode("utf-8")
            hdrs = {"Content-Type": "application/json"}
        raw = self._request("POST", path, params=params, body=data, headers=hdrs)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {"raw": raw.decode("utf-8", errors="replace")}

    def post_multipart(self, path: str, field: str, filename: str,
                       content: bytes, params: dict | None = None,
                       fields: dict[str, str] | None = None) -> Any:
        boundary = "----thothcraft" + os.urandom(8).hex()
        filename = filename.replace('"', '_').replace('\r', '_').replace('\n', '_')
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        prefix = b''.join(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
             f'{value}\r\n').encode('utf-8') for key, value in (fields or {}).items()
        )
        body = prefix + body
        raw = self._request(
            "POST", path, params=params, body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {"raw": raw.decode("utf-8", errors="replace")}


class Client:
    """Entry point: authenticate against Brain and open devices/minutes."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 token: Optional[str] = None, timeout: int = 30):
        self._http = _Http(base_url, token=token, timeout=timeout)

    # -- auth ------------------------------------------------------------
    @classmethod
    def login(cls, username: Optional[str] = None, password: Optional[str] = None,
              base_url: Optional[str] = None, token: Optional[str] = None,
              timeout: int = 30) -> "Client":
        """Authenticate and return a ready client.

        With no arguments, uses the token stored by ``thothcraft login``
        (CLI) or ``Client.store_token``.
        """
        client = cls(base_url or cls.load_base_url(), timeout=timeout)
        if token:
            client._http.token = token
            return client
        if username is not None and password is not None:
            client._login(username, password)
            return client
        stored = cls.load_token()
        if stored:
            client._http.token = stored
            return client
        raise AuthError("Provide token, username+password, or run `thothcraft login`")

    def _login(self, username: str, password: str) -> str:
        payload = self._http.post_json("/api/token", body={
            "username": username, "password": password})
        token = payload.get("access_token") or payload.get("token")
        if not token:
            raise AuthError("No access_token in /api/token response")
        self._http.token = token
        return token

    # -- credential storage (shared with thothcraft-cli) ------------------
    @staticmethod
    def store_token(token: str, base_url: str = DEFAULT_BASE_URL) -> None:
        os.makedirs(os.path.dirname(_CREDENTIALS_PATH), exist_ok=True)
        with open(_CREDENTIALS_PATH, "w", encoding="utf-8") as fh:
            json.dump({"base_url": base_url, "token": token}, fh)
        try:
            os.chmod(_CREDENTIALS_PATH, 0o600)
        except OSError:
            pass

    @staticmethod
    def load_token() -> Optional[str]:
        try:
            with open(_CREDENTIALS_PATH, encoding="utf-8") as fh:
                return json.load(fh).get("token")
        except (OSError, ValueError):
            return None

    @staticmethod
    def load_base_url() -> str:
        try:
            with open(_CREDENTIALS_PATH, encoding="utf-8") as fh:
                return json.load(fh).get("base_url") or DEFAULT_BASE_URL
        except (OSError, ValueError):
            return DEFAULT_BASE_URL

    @staticmethod
    def clear_token() -> None:
        try:
            os.remove(_CREDENTIALS_PATH)
        except OSError:
            pass

    # -- account ----------------------------------------------------------
    def entitlements(self) -> dict:
        """Plan + entitlement set for the authenticated account."""
        return self._http.get_json("/api/account/entitlements")

    def storage_usage(self) -> dict:
        return self._http.get_json("/api/storage/usage")

    # -- devices / minutes -------------------------------------------------
    def devices(self) -> list:
        from .devices import Device
        payload = self._http.get_json("/api/device/list")
        items = payload.get("devices") or payload.get("data") or []
        return [Device(self._http, d) for d in items]

    def device(self, device_id: str):
        from .devices import Device
        for d in self.devices():
            if d.uuid == device_id or d.name == device_id:
                return d
        return Device(self._http, {"device_uuid": device_id})

    def minute(self, minute_id: str, device=None):
        from .minutes import Minute
        from .devices import Device
        device_id = device.uuid if isinstance(device, Device) else device
        return Minute(self._http, minute_id, device_id=device_id)

    # -- labs --------------------------------------------------------------
    def upload_model(self, path: str | Path, *, name: str,
                     classes: list[str], input_spec: dict | list[dict],
                     version: str = '1', output_kind: str = 'logits',
                     execution: str = 'chunk') -> Model:
        """Upload a TorchScript archive with explicit device preprocessing metadata."""
        path = Path(path)
        if path.suffix.lower() not in {'.pt', '.pth'}:
            raise ValueError('Model filename must end in .pt or .pth')
        if not name.strip() or not classes or not input_spec:
            raise ValueError('name, classes and input_spec are required')
        metadata = {'schema': 'thoth-model/v1', 'name': name, 'version': version,
                    'class_names': classes,
                    'inputs': [input_spec] if isinstance(input_spec, dict) else input_spec,
                    'output': {'kind': output_kind, 'path': []}, 'execution': execution}
        content = path.read_bytes()
        if not content:
            raise ValueError('Model artifact is empty')
        payload = self._http.post_multipart('/api/datasets/models/upload', 'model',
                                            path.name, content,
                                            fields={'metadata': json.dumps(metadata)})
        return Model(self, payload['model'])

    def models(self) -> list[Model]:
        payload = self._http.get_json('/api/datasets/models')
        return [Model(self, row) for row in payload.get('models', [])]

    def deploy_model(self, model_id: int, device_id: str,
                     config: dict | None = None) -> Deployment:
        payload = self._http.post_json(f'/api/datasets/models/{model_id}/deploy',
                                      {'model_id': model_id, 'device_id': device_id,
                                       'config': config})
        return Deployment(self, payload)

    def deployments(self) -> list[Deployment]:
        payload = self._http.get_json('/api/datasets/models/deployments')
        return [Deployment(self, row) for row in payload.get('deployments', [])]

    def cancel_deployment(self, deployment_id: str) -> dict[str, Any]:
        raw = self._http._request('DELETE', f'/api/datasets/models/deployments/{deployment_id}')
        return json.loads(raw) if raw else {}

    def set_deployment_active(self, deployment_id: str, enabled: bool) -> dict[str, Any]:
        return self._http.post_json(f'/api/datasets/models/deployments/{deployment_id}/activation',
                                    {'enabled': enabled})

    def labs(self, track: Optional[str] = None) -> list:
        params = {"track": track} if track else None
        return self._http.get_json("/api/labs", params=params).get("labs", [])

    def lab_tracks(self) -> list:
        return self._http.get_json("/api/labs/tracks").get("tracks", [])

    def submit_lab(self, lab_id: int, notebook_path: str) -> dict:
        with open(notebook_path, "rb") as fh:
            content = fh.read()
        return self._http.post_multipart(
            f"/api/labs/{lab_id}/submit", "notebook",
            os.path.basename(notebook_path), content)

    def __repr__(self) -> str:
        return f"Client({self._http.base_url})"
