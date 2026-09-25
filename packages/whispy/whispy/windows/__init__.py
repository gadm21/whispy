"""Window feature extraction over SensorWindow samples.

``WindowFeatures`` exposes the scalar features rule processors reference
(``snr_mean``, ``radar_energy``, ``rms_accel`` …) computed from real
:class:`~whispy.contracts.SensorSample` payloads.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from ..contracts import SensorSample, SensorWindow

try:
    import numpy as np
except ImportError:  # numpy is a hard dep but keep import-time safety
    np = None  # type: ignore


def _payload_array(sample: SensorSample):
    payload = sample.payload
    if isinstance(payload, dict):
        # Structured payloads (e.g. radar_frame) expose scalar features
        # like snr_db — surface them so rules can reference them.
        for key in ("snr_db", "value", "reading"):
            if isinstance(payload.get(key), (int, float)):
                return [float(payload[key])]
        return []
    if np is not None:
        return np.asarray(payload, dtype=float)
    if isinstance(payload, (list, tuple)):
        return [float(x) for x in payload]
    return [float(payload)]


def _flatten(samples: List[SensorSample]) -> List[float]:
    out: List[float] = []
    for s in samples:
        try:
            arr = _payload_array(s)
            if np is not None:
                out.extend(np.asarray(arr, dtype=float).ravel().tolist())
            else:
                out.extend(arr)
        except (TypeError, ValueError):
            continue
    return out


def resolve_sensor_id(window: SensorWindow, name: str) -> Optional[str]:
    """Resolve a modality/sensor name to a concrete sensor id in a window.

    Manifests and rules reference modalities like ``radar``; windows key
    samples by sensor id like ``radar-0``. Resolution order: exact id,
    ``<name>-*`` prefix, then a sample whose ``sensor_type`` matches.
    Returns ``None`` when no sensor matches.
    """
    if name in window.samples or name in window.modalities:
        return name
    prefix = name + "-"
    for sid in list(window.samples) + list(window.modalities):
        if sid.startswith(prefix):
            return sid
    for sid, chunk in window.samples.items():
        for s in chunk:
            if s.sensor_type == name:
                return sid
    return None


class WindowFeatures:
    """Derived scalar features over a SensorWindow.

    Supported names:

    - ``<sensor>_mean|std|max|min|energy|rms|peak_to_peak``
    - ``snr_mean`` — mean over any SNR-bearing sensor (snr/radar_snr/radar)
    - ``rms_accel`` — RMS over IMU accel vectors (payloads of len ≥ 3)
    - direct scalars present in ``window.preprocessing`` or sample metadata
    """

    _STAT_RE = re.compile(r"(\w+)_(mean|std|max|min|energy|rms|peak_to_peak)$")

    def __init__(self, window: SensorWindow):
        self._window = window

    def feature(self, name: str) -> float:
        # Direct scalar in preprocessing metadata
        if name in self._window.preprocessing and isinstance(
                self._window.preprocessing[name], (int, float)):
            return float(self._window.preprocessing[name])

        # Scalar in latest sample metadata of any sensor
        for chunk in self._window.samples.values():
            if chunk and isinstance(chunk[-1].metadata.get(name), (int, float)):
                return float(chunk[-1].metadata[name])

        if name == "snr_mean":
            for key in ("snr", "radar_snr", "radar"):
                # Exact key first, then id resolution — window.samples is
                # keyed by sensor id (radar-0), not modality (radar).
                sid = key if self._window.samples.get(key) else \
                    resolve_sensor_id(self._window, key)
                if sid and self._window.samples.get(sid):
                    vals = _flatten(self._window.samples[sid])
                    if vals:
                        return sum(vals) / len(vals)
            raise KeyError("no SNR-bearing sensor in window")

        if name == "rms_accel":
            for key in ("imu", "accel", "accelerometer"):
                sid = key if self._window.samples.get(key) else \
                    resolve_sensor_id(self._window, key)
                if sid and self._window.samples.get(sid):
                    return self._rms_vectors(self._window.samples[sid])
            raise KeyError("no IMU/accelerometer sensor in window")

        m = self._STAT_RE.fullmatch(name)
        if m:
            key = resolve_sensor_id(self._window, m.group(1))
            if key is not None:
                vals = _flatten(self._window.samples.get(key) or [])
                if not vals:
                    raise KeyError(f"no samples for feature: {name}")
                return self._stat(vals, m.group(2))

        raise KeyError(f"unknown feature: {name}")

    @staticmethod
    def _stat(vals: List[float], op: str) -> float:
        n = len(vals)
        mean = sum(vals) / n
        if op == "mean":
            return mean
        if op == "std":
            return math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        if op == "max":
            return max(vals)
        if op == "min":
            return min(vals)
        if op == "energy":
            return sum(v * v for v in vals) / n
        if op == "rms":
            return math.sqrt(sum(v * v for v in vals) / n)
        if op == "peak_to_peak":
            return max(vals) - min(vals)
        raise KeyError(op)

    @staticmethod
    def _rms_vectors(samples: List[SensorSample]) -> float:
        """RMS magnitude across [x,y,z]-style payloads."""
        total, count = 0.0, 0
        for s in samples:
            p = s.payload
            if isinstance(p, dict):
                p = [p.get(k, 0.0) for k in ("x", "y", "z")]
            try:
                vec = [float(v) for v in p][:3]
            except (TypeError, ValueError):
                continue
            if vec:
                total += sum(v * v for v in vec)
                count += 1
        return math.sqrt(total / count) if count else 0.0


__all__ = ["WindowFeatures", "resolve_sensor_id"]
