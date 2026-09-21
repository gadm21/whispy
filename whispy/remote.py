"""Remote access to Thoth devices and captured minute data via the Brain API.

This complements :mod:`whispy.cloud_client` (which *pushes* data up to the
Whispy SaaS backend) — ``remote`` *reads* captured minutes and *controls*
devices on a Thoth Brain deployment.

Quick start
-----------
    from whispy.remote import Client

    c = Client("https://your-brain.example.com",
               username="you@example.com", password="secret")

    for d in c.devices():
        print(d.name, d.online)
        d.start()                       # queue start_collection
        for chunk in d.live():          # stream live analysis chunks
            print(chunk)
        d.stop()                        # queue stop_collection

    m = c.minute("20260921_0133", device=d)
    print(m.metadata)
    radar = m.radar()                   # SensorData (frames + timestamps)
    csi = m.csi()                       # SensorData (parsed subcarrier amp)
    frames = m.camera()                 # list[bytes] JPEG, one per second
    sense = m.sense()                   # list[dict] Sense HAT rows
    print(m.predictions)
    m.download("out/")                  # fetch the minute bundle

Dependencies: only ``numpy`` (already a whispy dependency) + stdlib urllib.
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Iterator, Sequence

import numpy as np

__all__ = [
    "Client", "Device", "Minute", "SensorData",
    "ThothError", "AuthError", "NotFoundError", "APIError",
]


# ── errors ──────────────────────────────────────────────────────────────
class ThothError(Exception):
    """Base error for all remote failures."""


class AuthError(ThothError):
    """Authentication failed or the token expired."""


class NotFoundError(ThothError):
    """The requested device, minute, or resource was not found."""


class APIError(ThothError):
    """The backend returned a non-success response."""

    def __init__(self, message: str, status_code: int | None = None, detail: object = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


# ── low-level HTTP ──────────────────────────────────────────────────────
class _Http:
    """Minimal urllib JSON/bytes client with a JWT bearer token."""

    def __init__(self, base_url: str, token: str | None = None, timeout: int = 30):
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
            if e.code in (401, 403):
                raise AuthError(f"HTTP {e.code} on {path}: {detail}") from e
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


# ── capture.npz decoding ────────────────────────────────────────────────
def _blob_at(data: np.ndarray, offsets: np.ndarray, index: int) -> bytes:
    if index < 0 or index + 1 >= len(offsets):
        return b""
    return data[int(offsets[index]):int(offsets[index + 1])].tobytes()


class _Container:
    """Lazy decoder for a capture.npz byte string."""

    def __init__(self, content: bytes):
        self._archive = np.load(io.BytesIO(content), allow_pickle=False)
        raw = self._archive["metadata_json"].astype(np.uint8).tobytes()
        self.metadata = json.loads(raw.decode("utf-8"))

    def close(self):
        try:
            self._archive.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _series(self, prefix: str) -> dict[str, np.ndarray]:
        a = self._archive
        out = {}
        for key in ("second_index", "unix_ns", "receiver_index", "sequence"):
            name = f"{prefix}_{key}"
            if name in a.files:
                out[key] = a[name]
        return out


# ── sensor data container ───────────────────────────────────────────────
class SensorData:
    """A per-sensor series: parallel arrays of samples + timestamps.

    Attributes
    ----------
    samples : list of per-sample payloads (bytes for radar, str/dict for CSI/sense).
    unix_ns : np.ndarray of capture timestamps (ns), aligned with ``samples``.
    second_index : np.ndarray of per-sample second offsets within the minute.
    """

    def __init__(self, samples: list[Any], unix_ns=None, second_index=None, extra: dict | None = None):
        self.samples = samples
        self.unix_ns = np.asarray(unix_ns) if unix_ns is not None else np.zeros(len(samples), dtype=np.int64)
        self.second_index = np.asarray(second_index) if second_index is not None else np.zeros(len(samples), dtype=np.int64)
        self.extra = extra or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self) -> str:
        return f"SensorData(n={len(self.samples)})"


# ── minute ──────────────────────────────────────────────────────────────
class Minute:
    """A captured minute on a device, decoded from its capture.npz."""

    def __init__(self, http: _Http, minute_id: str, device_id: str | None = None):
        self._http = http
        self.id = minute_id
        self.device_id = device_id
        self._container: _Container | None = None
        self._metadata: dict | None = None

    # -- container / metadata -------------------------------------------
    def _params(self, extra: dict | None = None) -> dict:
        p = {"device_id": self.device_id}
        if extra:
            p.update(extra)
        return p

    def _load_container(self) -> _Container:
        if self._container is None:
            content = self._http.get_bytes(
                f"/api/file/minute/{self.id}/download", self._params())
            self._container = _Container(content)
        return self._container

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            try:
                payload = self._http.get_json(
                    f"/api/file/minute/{self.id}/container/metadata", self._params())
                self._metadata = payload.get("metadata") or payload
            except ThothError:
                self._metadata = self._load_container().metadata
        return self._metadata

    @property
    def predictions(self) -> list[dict]:
        md = self.metadata
        return md.get("model_predictions") or md.get("minute_predictions") or []

    # -- per-sensor accessors -------------------------------------------
    def radar(self) -> SensorData:
        """Raw radar frames (12-byte header + uint12 payload) + timestamps."""
        with self._load_container() as c:
            a = c._archive
            if "radar_sample_bytes" not in a.files:
                return SensorData([])
            payload, offsets = a["radar_sample_bytes"], a["radar_sample_offsets"]
            n = len(a["radar_sample_second_index"])
            samples = [_blob_at(payload, offsets, i) for i in range(n)]
            return SensorData(
                samples,
                unix_ns=a["radar_sample_unix_ns"] if "radar_sample_unix_ns" in a.files else None,
                second_index=a["radar_sample_second_index"],
                extra={"sequence": a["radar_sample_sequence"] if "radar_sample_sequence" in a.files else None},
            )

    def csi(self, parse: bool = True) -> SensorData:
        """CSI samples; ``parse=True`` decodes subcarrier amplitudes via whispy.core."""
        with self._load_container() as c:
            a = c._archive
            if "csi_sample_bytes" not in a.files:
                return SensorData([])
            payload, offsets = a["csi_sample_bytes"], a["csi_sample_offsets"]
            n = len(a["csi_sample_second_index"])
            lines = [_blob_at(payload, offsets, i).decode("utf-8", "replace") for i in range(n)]
            receivers = a["csi_sample_receiver_index"] if "csi_sample_receiver_index" in a.files else None
            samples: list[Any] = lines
            if parse:
                from whispy.core import parse_csi_line
                parsed = []
                for i, line in enumerate(lines):
                    row = parse_csi_line(line)
                    if row is not None and receivers is not None:
                        row["receiver_index"] = int(receivers[i])
                    parsed.append(row if row is not None else line)
                samples = parsed
            return SensorData(
                samples,
                unix_ns=a["csi_sample_unix_ns"] if "csi_sample_unix_ns" in a.files else None,
                second_index=a["csi_sample_second_index"],
                extra={"receiver_index": receivers},
            )

    def camera(self) -> list[bytes]:
        """JPEG bytes, one per captured second (empty where no frame)."""
        with self._load_container() as c:
            a = c._archive
            if "camera_jpeg_bytes" not in a.files:
                return []
            present = a["camera_present"]
            payload, offsets = a["camera_jpeg_bytes"], a["camera_jpeg_offsets"]
            return [_blob_at(payload, offsets, i) if bool(present[i]) else b""
                    for i in range(len(present))]

    def sense(self) -> list[dict]:
        """Parsed Sense HAT JSON rows."""
        with self._load_container() as c:
            a = c._archive
            if "sense_sample_bytes" not in a.files:
                return []
            payload, offsets = a["sense_sample_bytes"], a["sense_sample_offsets"]
            n = len(a["sense_sample_second_index"])
            rows = []
            for i in range(n):
                try:
                    row = json.loads(_blob_at(payload, offsets, i).decode("utf-8", "replace"))
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            return rows

    # -- persistence -----------------------------------------------------
    def download(self, out_dir: str = ".") -> str:
        """Download the raw minute bundle (npz) to ``out_dir/<id>.npz``."""
        import os
        content = self._http.get_bytes(
            f"/api/file/minute/{self.id}/download", self._params())
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.id}.npz")
        with open(path, "wb") as fh:
            fh.write(content)
        return path

    def __repr__(self) -> str:
        return f"Minute({self.id}, device={self.device_id})"


# ── device ──────────────────────────────────────────────────────────────
class Device:
    """A remote Thoth device: identity, control, live stream, minutes."""

    def __init__(self, http: _Http, info: dict):
        self._http = http
        self.info = info
        self.uuid = str(info.get("device_uuid") or info.get("device_id") or "")
        self.name = info.get("device_name") or info.get("name") or self.uuid
        self.online = bool(info.get("online") or info.get("is_online"))

    # -- control ---------------------------------------------------------
    def command(self, command: str, payload: dict | None = None) -> dict:
        return self._http.post_json(
            f"/api/device/{self.uuid}/commands",
            {"command": command, "payload": payload or {}})

    def start(self) -> dict:
        return self.command("start_collection")

    def stop(self) -> dict:
        return self.command("stop_collection")

    def capture_settings(self) -> dict:
        return self._http.get_json(f"/api/device/{self.uuid}/capture-settings")

    # -- live ------------------------------------------------------------
    def live(self, cursor: int = 0, poll_s: float = 1.0,
             max_items: int | None = None) -> Iterator[dict]:
        """Yield live analysis chunks as they arrive (long-poll loop)."""
        seen = 0
        while True:
            payload = self._http.get_json(
                f"/api/device/{self.uuid}/live-chunks", {"cursor": cursor})
            chunks = payload.get("chunks") or payload.get("data") or []
            for chunk in chunks:
                yield chunk
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            cursor = int(payload.get("cursor") or payload.get("next_cursor") or cursor)
            if not chunks:
                time.sleep(poll_s)

    # -- data ------------------------------------------------------------
    def files(self) -> list[dict]:
        payload = self._http.get_json(f"/api/device/{self.uuid}/files")
        return payload.get("files") or payload.get("data") or []

    def minutes(self) -> list[str]:
        """Distinct captured-minute ids known for this device."""
        ids = []
        for f in self.files():
            m = f.get("minute") or f.get("minute_id") or f.get("folder_name")
            if m and m not in ids:
                ids.append(m)
        return ids

    def minute(self, minute_id: str) -> Minute:
        return Minute(self._http, minute_id, device_id=self.uuid)

    def __repr__(self) -> str:
        return f"Device({self.name}, uuid={self.uuid}, online={self.online})"


# ── client ──────────────────────────────────────────────────────────────
class Client:
    """Entry point: authenticate against a Thoth Brain and open devices/minutes."""

    def __init__(self, base_url: str, username: str | None = None,
                 password: str | None = None, token: str | None = None,
                 timeout: int = 30):
        self._http = _Http(base_url, timeout=timeout)
        if token:
            self._http.token = token
        elif username is not None and password is not None:
            self.login(username, password)
        else:
            raise AuthError("Provide either token or username+password")

    def login(self, username: str, password: str) -> str:
        payload = self._http.post_json("/api/token", form={
            "username": username, "password": password})
        token = payload.get("access_token") or payload.get("token")
        if not token:
            raise AuthError("No access_token in /api/token response")
        self._http.token = token
        return token

    def devices(self) -> list[Device]:
        payload = self._http.get_json("/api/device/list")
        items = payload.get("devices") or payload.get("data") or []
        return [Device(self._http, d) for d in items]

    def device(self, device_id: str) -> Device:
        for d in self.devices():
            if d.uuid == device_id or d.name == device_id:
                return d
        return Device(self._http, {"device_uuid": device_id})

    def minute(self, minute_id: str, device: Device | str | None = None) -> Minute:
        device_id = device.uuid if isinstance(device, Device) else device
        return Minute(self._http, minute_id, device_id=device_id)

    def __repr__(self) -> str:
        return f"Client({self._http.base_url})"
