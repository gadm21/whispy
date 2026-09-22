"""Spatial context API — spaces, zones, and live occupancy state.

    client = thothcraft.Client.login()
    office = client.space("office")
    office.occupied            # bool
    office.people_count        # int
    office.zones["desk"].occupied
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional


class Zone:
    """A named polygonal region inside a :class:`Space`."""

    def __init__(self, space: "Space", info: Dict[str, Any]):
        self._space = space
        self.id = info.get("id")
        self.name = info.get("name") or ""
        self.polygon = info.get("polygon") or []
        self._state: Dict[str, Any] = {}

    @property
    def occupied(self) -> bool:
        return bool(self._state.get("occupied"))

    @property
    def people_count(self) -> int:
        return int(self._state.get("people_count") or 0)

    @property
    def confidence(self) -> float:
        return float(self._state.get("confidence") or 0.0)

    def __repr__(self) -> str:
        return f"Zone({self.name}, occupied={self.occupied})"


class _ZoneMap:
    """Dict-like zone access: ``space.zones["desk"]`` plus iteration."""

    def __init__(self, space: "Space"):
        self._space = space
        self._zones: Dict[str, Zone] = {}

    def _sync(self, zone_infos: List[Dict[str, Any]]) -> None:
        for info in zone_infos:
            name = info.get("name") or ""
            if name in self._zones:
                self._zones[name].id = info.get("id")
                self._zones[name].polygon = info.get("polygon") or []
            else:
                self._zones[name] = Zone(self._space, info)

    def __getitem__(self, name: str) -> Zone:
        return self._zones[name]

    def __iter__(self) -> Iterator[Zone]:
        return iter(self._zones.values())

    def __len__(self) -> int:
        return len(self._zones)

    def keys(self):
        return self._zones.keys()


class Space:
    """A named physical area with zones, placed devices, and live state."""

    def __init__(self, http, info: Dict[str, Any]):
        self._http = http
        self.id = info.get("id")
        self.name = info.get("name") or ""
        self.parent_id = info.get("parent_id")
        self.width_m = info.get("width_m")
        self.height_m = info.get("height_m")
        self.zones = _ZoneMap(self)
        self.zones._sync(info.get("zones") or [])
        self.devices = info.get("devices") or []
        self._state: Dict[str, Any] = {}

    # -- structure -------------------------------------------------------
    def add_zone(self, name: str, polygon: List[List[float]]) -> Zone:
        payload = self._http.post_json(
            f"/api/spaces/{self.id}/zones", {"name": name, "polygon": polygon})
        zone = Zone(self, payload.get("zone") or {})
        self.zones._zones[zone.name] = zone
        return zone

    def remove_zone(self, name: str) -> None:
        zone = self.zones[name]
        self._http.delete(f"/api/spaces/{self.id}/zones/{zone.id}")
        del self.zones._zones[name]

    def rename(self, name: str) -> None:
        self._http.put_json(f"/api/spaces/{self.id}", {"name": name})
        self.name = name

    def delete(self) -> None:
        self._http.delete(f"/api/spaces/{self.id}")

    # -- live state ------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        """Fetch and cache the live spatial state for this space."""
        payload = self._http.get_json(f"/api/spaces/{self.id}/state")
        self._state = payload.get("state") or {}
        for name, zs in (self._state.get("zones") or {}).items():
            if name in self.zones._zones:
                self.zones._zones[name]._state = zs
        return self._state

    def refresh(self) -> "Space":
        self.state()
        return self

    @property
    def occupied(self) -> bool:
        return bool(self._state.get("occupied"))

    @property
    def people_count(self) -> int:
        return int(self._state.get("people_count") or 0)

    @property
    def activity(self) -> Optional[str]:
        return self._state.get("activity")

    @property
    def confidence(self) -> float:
        return float(self._state.get("confidence") or 0.0)

    @property
    def last_activity(self) -> Optional[str]:
        return self._state.get("last_activity")

    def __repr__(self) -> str:
        return f"Space({self.name}, id={self.id}, occupied={self.occupied})"
