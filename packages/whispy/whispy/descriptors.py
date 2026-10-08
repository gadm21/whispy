"""Physical descriptors — compact, JSON-safe summaries of a SensorWindow.

Descriptors are what leaves a node for context building: per-sensor
statistics, never raw frames. They are heterogeneous by design (radar
SNR, CSI amplitude variance, RSSI sightings, IMU energy, audio level…)
and share one shape so an LLM context builder can compare them:

    {"<sensor_id>": {"type": "radar", "n": 50, "rate_hz": 10.0,
                     "age_s": 0.1, "fields": {"snr_db": {"mean": …,
                     "std": …, "min": …, "max": …}}, "scan": {...}}}

Detail levels:

* ``minimal``      — type, n, rate, age only.
* ``descriptors``  — + numeric field statistics (default).
* ``full``         — + radio scan summaries (top emitters, beacons).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .contracts import SensorSample, SensorWindow

DETAIL_LEVELS = ("minimal", "descriptors", "full")
MAX_FIELDS = 12
MAX_VECTOR = 4096
MAX_EMITTERS = 8

_SCAN_TYPES = {"ble_scan", "wifi_scan"}


def _stats(xs: List[float]) -> Dict[str, float]:
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def _numeric(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return None


def _flatten_numbers(v: Any, out: List[float]) -> None:
    if len(out) >= MAX_VECTOR:
        return
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        for x in v:
            _flatten_numbers(x, out)
            if len(out) >= MAX_VECTOR:
                return
        return
    f = _numeric(v)
    if f is not None:
        out.append(f)


def _fields(samples: Iterable[SensorSample]) -> Dict[str, List[float]]:
    """Numeric leaves per field name across samples (dict payloads one
    level deep; array payloads → ``value``, plus ``magnitude`` for
    3-vectors such as IMU)."""
    fields: Dict[str, List[float]] = {}
    for s in samples:
        p = s.payload
        if hasattr(p, "tolist"):
            p = p.tolist()
        if isinstance(p, Mapping):
            for k, v in p.items():
                if k.endswith("_b64") or k in ("devices", "networks", "frames"):
                    continue
                f = _numeric(v)
                if f is not None:
                    fields.setdefault(str(k), []).append(f)
                elif isinstance(v, (list, tuple)) and v and len(fields) < MAX_FIELDS:
                    vals: List[float] = []
                    _flatten_numbers(v, vals)
                    if vals:
                        fields.setdefault(f"{k}_mean", []).append(
                            sum(vals) / len(vals))
        elif isinstance(p, (list, tuple)):
            vals = []
            _flatten_numbers(p, vals)
            if not vals:
                continue
            if len(vals) == 3:
                fields.setdefault("magnitude", []).append(
                    math.sqrt(sum(x * x for x in vals)))
            fields.setdefault("value", []).extend(vals)
        else:
            f = _numeric(p)
            if f is not None:
                fields.setdefault("value", []).append(f)
        for k, v in (s.metadata or {}).items():
            f = _numeric(v)
            if f is not None and k in ("snr_db", "rssi", "snr_f", "level_db",
                                       "confidence", "temperature"):
                fields.setdefault(str(k), []).append(f)
    return fields


def _emitters(samples: Iterable[SensorSample]) -> Dict[str, Any]:
    """Radio scan summary: unique emitters, strongest, decoded beacons."""
    best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for s in samples:
        p = s.payload if isinstance(s.payload, Mapping) else {}
        items = p.get("devices") or p.get("networks") or []
        if not items and ("mac" in p or "bssid" in p):
            items = [p]
        for d in items:
            if not isinstance(d, Mapping):
                continue
            ident = str(d.get("mac") or d.get("bssid") or d.get("addr") or "")
            rssi = _numeric(d.get("rssi"))
            if not ident or rssi is None:
                continue
            prev = best.get(ident)
            merged = dict(prev[1]) if prev else {}
            merged.update({k: v for k, v in d.items() if v not in (None, "")})
            best[ident] = (max(rssi, prev[0]) if prev else rssi, merged)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:MAX_EMITTERS]
    out_top = []
    for ident, (rssi, d) in top:
        entry: Dict[str, Any] = {"id": ident, "rssi": rssi}
        for k in ("name", "ssid", "beacon", "channel"):
            if d.get(k):
                entry[k] = d[k]
        out_top.append(entry)
    return {"emitters": len(best), "top": out_top}


def sensor_descriptor(samples: List[SensorSample], *, detail: str = "descriptors",
                      now: Optional[float] = None) -> Dict[str, Any]:
    now = now if now is not None else time.time()
    if not samples:
        return {"type": None, "n": 0}
    first, last = samples[0], samples[-1]
    span = max(0.0, last.timestamp - first.timestamp)
    desc: Dict[str, Any] = {
        "type": last.sensor_type,
        "n": len(samples),
        "rate_hz": round((len(samples) - 1) / span, 2) if span > 0 else None,
        "age_s": round(max(0.0, now - last.timestamp), 2),
    }
    if detail == "minimal":
        return desc
    is_scan = last.sensor_type in _SCAN_TYPES or any(
        isinstance(s.payload, Mapping) and
        s.payload.get("kind") in _SCAN_TYPES for s in samples[-5:])
    if not is_scan:
        fields = _fields(samples)
        desc["fields"] = {k: _stats(v) for k, v in
                          list(fields.items())[:MAX_FIELDS] if v}
    if detail == "full" or is_scan:
        scan = _emitters(samples)
        if scan["emitters"]:
            if detail != "full":
                scan.pop("top", None)
            desc["scan"] = scan
    return desc


def window_descriptors(window: SensorWindow, detail: str = "descriptors",
                       now: Optional[float] = None) -> Dict[str, Any]:
    """Per-sensor physical descriptors for one synchronized window."""
    if detail not in DETAIL_LEVELS:
        raise ValueError(f"detail must be one of {DETAIL_LEVELS}")
    now = now if now is not None else time.time()
    out: Dict[str, Any] = {}
    for sid, samples in window.samples.items():
        if samples:
            out[sid] = sensor_descriptor(list(samples), detail=detail, now=now)
    for sid, m in window.modalities.items():
        if sid not in out:
            out[sid] = {"type": None, "n": 0,
                        "state": getattr(m, "state", None)}
    return out


__all__ = ["DETAIL_LEVELS", "sensor_descriptor", "window_descriptors"]
