"""Public egress geolocation — each node resolves itself.

``public_geo()`` returns the caller's egress IP plus coarse postal
metadata via ipwho.is (no key required). Used by the thoth daemon's
``/api/v1/location`` endpoint and by ``LocalDevice.location()`` so a
node's postal code always reflects its own vantage point, not a
database row that can go stale.
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict, Optional

_CACHE_TTL_S = 3600.0
_cache: Dict[str, Any] = {"at": 0.0, "result": None}


def public_geo(timeout: float = 8.0,
               force: bool = False) -> Optional[Dict[str, Any]]:
    """Resolve this node's public IP → coarse location.

    Returns ``{"ip", "postal_code", "city", "region", "country",
    "latitude", "longitude"}`` or ``None`` when unreachable. Cached for
    one hour — egress IPs change slowly and the lookup should never be
    on a hot path.
    """
    if not force and _cache["result"] and \
            time.time() - _cache["at"] < _CACHE_TTL_S:
        return dict(_cache["result"])
    try:
        with urllib.request.urlopen(
                "https://ipwho.is/", timeout=timeout) as res:
            data = json.loads(res.read())
    except Exception:
        return None
    if not data.get("success", True):
        return None
    loc = data.get("location") or {}
    out = {
        "ip": data.get("ip"),
        "postal_code": data.get("postal") or loc.get("postal_code"),
        "city": data.get("city"),
        "region": data.get("region"),
        "country": data.get("country"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
    }
    _cache["at"] = time.time()
    _cache["result"] = out
    return dict(out)


__all__ = ["public_geo"]
