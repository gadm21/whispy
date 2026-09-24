"""Device registry — a client's view of paired Thoth nodes.

Pulls ``/v1/devices`` from Brain (canonical contract) and caches each
node's advertised compute capability. LAN reachability is probed
lazily — a device is never assumed online from its registry entry.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional


class DeviceRegistry:
    def __init__(self, brain_url: str = "", token: str = "",
                 fetcher: Optional[Callable[[], List[Dict[str, Any]]]] = None):
        self.brain_url = brain_url.rstrip("/")
        self.token = token
        self._fetcher = fetcher
        self._devices: Dict[str, Dict[str, Any]] = {}
        self._fetched_at = 0.0

    def _fetch(self) -> List[Dict[str, Any]]:
        if self._fetcher:
            return self._fetcher()
        req = urllib.request.Request(
            f"{self.brain_url}/v1/devices",
            headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=10) as res:
            return json.loads(res.read()).get("devices") or []

    def refresh(self) -> List[Dict[str, Any]]:
        devices = self._fetch()
        self._devices = {d.get("id") or d.get("device_id"): d
                         for d in devices if d.get("id") or d.get("device_id")}
        self._fetched_at = time.time()
        return list(self._devices.values())

    def list(self) -> List[Dict[str, Any]]:
        return list(self._devices.values())

    def get(self, device_id: str) -> Optional[Dict[str, Any]]:
        return self._devices.get(device_id)

    def probe(self, device_id: str, timeout: float = 3.0) -> Dict[str, Any]:
        """Live-probe a node's local API health (LAN only)."""
        dev = self._devices.get(device_id) or {}
        host = dev.get("lan_host") or dev.get("ip_address")
        port = dev.get("local_port") or 5000
        token = dev.get("local_token") or ""
        if not host:
            return {"device_id": device_id, "reachable": False,
                    "error": "no LAN address known"}
        try:
            req = urllib.request.Request(
                f"http://{host}:{port}/api/v1/health",
                headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return {"device_id": device_id, "reachable": True,
                        "health": json.loads(res.read())}
        except Exception as exc:
            return {"device_id": device_id, "reachable": False,
                    "error": str(exc)}
