"""REST-only client for pushing data to the Whispy hosted backend.

Used by devices that connect to the Railway-hosted SaaS backend without
needing an MQTT broker.  Authenticates with an API key.

Usage
-----
    from whispy.cloud_client import CloudClient

    client = CloudClient(
        backend_url="https://whispy-backend.up.railway.app",
        api_key="your-api-key",
        node_id="lab-toronto-01",
    )
    client.register(location="Toronto", latitude=43.65, longitude=-79.38)
    client.push_prediction(label="occupied", class_idx=1, confidence=0.94)
    client.push_diagnostics(csi_rate=148.5, cpu_temp=52.3)
"""
from __future__ import annotations

import json
import time
import threading
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class CloudClient:
    """Lightweight REST client for the Whispy cloud backend.

    All methods are non-blocking by default (fire in background thread)
    to avoid slowing down the real-time inference loop.

    Parameters
    ----------
    backend_url : Base URL of the backend (e.g. https://whispy.up.railway.app).
    api_key : API key for authentication (X-API-Key header).
    node_id : Unique device identifier.
    timeout : HTTP request timeout in seconds.
    async_push : If True (default), pushes run in background threads.
    """

    def __init__(
        self,
        backend_url: str,
        api_key: str,
        node_id: str,
        timeout: int = 10,
        async_push: bool = True,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.api_key = api_key
        self.node_id = node_id
        self.timeout = timeout
        self.async_push = async_push
        self._errors: list[str] = []
        self._push_count = 0
        self._error_count = 0

    # ── internal HTTP helpers ───────────────────────────────
    def _post(self, path: str, body: dict) -> dict | None:
        """POST JSON to backend.  Returns parsed response or None on error."""
        url = f"{self.backend_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = Request(
            url, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
            },
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                self._push_count += 1
                return json.loads(resp.read().decode())
        except HTTPError as e:
            msg = f"HTTP {e.code} on {path}: {e.read().decode('utf-8', errors='replace')[:200]}"
            self._errors.append(msg)
            self._error_count += 1
            return None
        except (URLError, OSError) as e:
            msg = f"Connection error on {path}: {e}"
            self._errors.append(msg)
            self._error_count += 1
            return None

    def _get(self, path: str, params: dict | None = None) -> Any:
        """GET from backend."""
        url = f"{self.backend_url}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if qs:
                url += f"?{qs}"
        req = Request(
            url, method="GET",
            headers={"X-API-Key": self.api_key},
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, OSError) as e:
            self._errors.append(f"GET {path}: {e}")
            self._error_count += 1
            return None

    def _async(self, fn, *args):
        if self.async_push:
            threading.Thread(target=fn, args=args, daemon=True).start()
        else:
            fn(*args)

    # ── device registration ─────────────────────────────────
    def register(
        self,
        location: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
        task: str = "occupancy",
        labels: list[str] | None = None,
    ) -> dict | None:
        """Register this device with the backend (synchronous)."""
        from whispy.device import DeviceInfo
        dev = DeviceInfo.from_system(
            node_id=self.node_id,
            location=location,
            latitude=latitude,
            longitude=longitude,
            task=task,
        )
        dev.labels = labels or []
        return self._post("/devices/register", dev.to_dict())

    # ── data ingest ─────────────────────────────────────────
    def push_prediction(
        self,
        label: str = "",
        class_idx: int = -1,
        confidence: float = 0.0,
        probabilities: dict[str, float] | None = None,
    ) -> None:
        """Push a prediction to the backend (async by default)."""
        body = {
            "node_id": self.node_id,
            "label": label,
            "class": class_idx,
            "confidence": confidence,
        }
        if probabilities:
            body["probabilities"] = probabilities
        self._async(self._post, "/ingest/prediction", body)

    def push_diagnostics(
        self,
        csi_rate: float = 0,
        cpu_temp: float | None = None,
        esp32_alive: bool = True,
        cache_mb: float = 0,
    ) -> None:
        """Push diagnostics to the backend (async by default)."""
        body = {
            "node_id": self.node_id,
            "csi_rate": csi_rate,
            "cpu_temp": cpu_temp,
            "esp32_alive": esp32_alive,
            "cache_mb": cache_mb,
        }
        self._async(self._post, "/ingest/diagnostics", body)

    def upload_file(self, filepath: str, label: str = "unknown") -> dict | None:
        """Upload a .npz cache export to the backend (synchronous).

        Uses multipart/form-data via urllib (no requests dependency).
        """
        import mimetypes
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            self._errors.append(f"File not found: {filepath}")
            return None

        boundary = f"----WhispyBoundary{int(time.time() * 1000)}"
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        body_parts = []
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
        )
        body_parts.append(path.read_bytes())
        body_parts.append(f"\r\n--{boundary}--\r\n".encode())

        data = b"".join(body_parts)
        url = f"{self.backend_url}/upload/{self.node_id}?label={label}"
        req = Request(
            url, data=data, method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-API-Key": self.api_key,
            },
        )
        try:
            with urlopen(req, timeout=60) as resp:
                self._push_count += 1
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, OSError) as e:
            self._errors.append(f"Upload {filepath}: {e}")
            self._error_count += 1
            return None

    # ── query endpoints ─────────────────────────────────────
    def get_predictions(self, limit: int = 100) -> list[dict] | None:
        return self._get("/predictions", {"node_id": self.node_id, "limit": limit})

    def get_diagnostics(self, limit: int = 100) -> list[dict] | None:
        return self._get("/diagnostics", {"node_id": self.node_id, "limit": limit})

    def get_devices(self) -> list[dict] | None:
        return self._get("/devices")

    def health(self) -> dict | None:
        return self._get("/health")

    # ── status ──────────────────────────────────────────────
    @property
    def stats(self) -> dict:
        return {
            "pushes": self._push_count,
            "errors": self._error_count,
            "last_errors": self._errors[-5:] if self._errors else [],
        }

    def __repr__(self) -> str:
        return (f"CloudClient({self.backend_url}, node={self.node_id}, "
                f"pushes={self._push_count}, errors={self._error_count})")
