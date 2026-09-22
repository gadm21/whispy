"""Local device access — talk to a Thoth node's dashboard directly.

    import thothcraft
    node = thothcraft.local("thoth-chen.local")   # or "192.168.1.42"
    node.sensors()                               # probed capabilities
    node.occupancy()                             # live radar occupancy
    for cap in node.captures():                  # iterate minute folders
        ...

The local dashboard (thoth runtime, port 5000) exposes the same
concepts as Brain — sensors, captures, models — without cloud auth.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from .errors import APIError


class _LocalHttp:
    def __init__(self, base_url: str, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_json(self, path: str, params: Optional[dict] = None) -> Any:
        url = self.base_url + path
        if params:
            qs = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
            if qs:
                url += "?" + qs
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            raise APIError(f"local request failed: {url}: {exc}") from exc

    def post_json(self, path: str, body: Optional[dict] = None) -> Any:
        data = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            raise APIError(f"local request failed: {path}: {exc}") from exc


class LocalDevice:
    """A Thoth node reachable on the LAN (dashboard on port 5000)."""

    def __init__(self, host: str, port: int = 5000, timeout: int = 15):
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        self._http = _LocalHttp(f"{host}:{port}", timeout=timeout)
        self.host = host

    # -- introspection ----------------------------------------------------
    def sensors(self) -> Dict[str, Any]:
        """Probed sensor capabilities reported by the node."""
        return self._http.get_json("/api/sensors")

    def status(self) -> Dict[str, Any]:
        return self._http.get_json("/api/settings")

    def occupancy(self) -> Dict[str, Any]:
        """Live radar occupancy reading."""
        return self._http.get_json("/api/radar/occupancy")

    def radar_live(self) -> Dict[str, Any]:
        return self._http.get_json("/api/radar/live")

    # -- captures -----------------------------------------------------------
    def captures(self) -> List[Dict[str, Any]]:
        payload = self._http.get_json("/api/captures")
        return payload.get("captures") or payload.get("minutes") or []

    def capture(self, minute: str) -> Dict[str, Any]:
        return self._http.get_json(f"/api/captures/{minute}")

    def sensor_data(self, minute: str, sensor: str) -> Any:
        return self._http.get_json(f"/api/captures/{minute}/sensor/{sensor}")

    # -- control ------------------------------------------------------------
    def live_session(self, action: str = "start") -> Dict[str, Any]:
        return self._http.post_json("/api/live/session", {"action": action})

    def __repr__(self) -> str:
        return f"LocalDevice({self.host})"


def local(host: str = "thoth.local", port: int = 5000,
          timeout: int = 15) -> LocalDevice:
    """Connect to a Thoth node on the LAN.

        node = thothcraft.local("thoth-april.local")
        node.occupancy()
    """
    return LocalDevice(host, port=port, timeout=timeout)
