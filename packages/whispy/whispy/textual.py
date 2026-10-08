"""Textual descriptors — short, human-readable cues per sensor.

Physical descriptors (:mod:`whispy.descriptors`) are numbers. Textual
descriptors turn them — and, where a small on-device model helps, the
raw data itself — into one sentence per sensor plus structured
``cues``. The LLM context builder reads these instead of raw frames:

=============  ==========================================  ==============
data type      textual cue                                 model
=============  ==========================================  ==============
microphone     ``speech: "turn the lights off"``           whisper-stt (tiny.en)
camera         ``2 people visible``                         opencv-haar-person
camera         ``face: Gad (distance 6.8)``                 opencv-haar-face → pca-face-recognizer
radar / csi    ``high motion (SNR std 2.9 dB)``             rules over descriptors
imu            ``moving (|a| std 1.4)``                     rules over descriptors
ble/wifi scan  ``11 emitters; strongest watch -45 dBm``     rules over descriptors
=============  ==========================================  ==============

Model providers are optional: if the plugin isn't installed or fails,
the sensor still gets its rule-based sentence and the cue carries an
``error``. Model providers are rate-limited (``min_interval_s``) and
cache their last result so a 2 Hz loop never runs Whisper twice a
second.

    from whispy.textual import TextualDescriber
    describer = TextualDescriber.default()           # rules + installed models
    text = describer.describe(window)                # {sensor_id: {"text", "cues"}}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .contracts import SensorWindow
from .descriptors import window_descriptors

logger = logging.getLogger(__name__)

TEXT_MAX = 240

ModelFactory = Callable[[str, Dict[str, Any]], Any]


def _field(desc: Mapping[str, Any], name: str, stat: str) -> Optional[float]:
    f = (desc.get("fields") or {}).get(name)
    if isinstance(f, Mapping) and isinstance(f.get(stat), (int, float)):
        return float(f[stat])
    return None


def _level(x: float, low: float, high: float) -> str:
    return "low" if x < low else ("high" if x >= high else "moderate")


# ---------------------------------------------------------------------------
# Rule-based sentences over physical descriptors
# ---------------------------------------------------------------------------

def physical_sentence(desc: Mapping[str, Any]) -> Dict[str, Any]:
    """One sentence + cues for a physical descriptor of any type."""
    t = str(desc.get("type") or "")
    n = int(desc.get("n") or 0)
    if n == 0:
        state = desc.get("state") or "no data"
        return {"text": f"{t or 'sensor'}: {state}", "cues": {"state": state}}
    age = desc.get("age_s")
    stale = isinstance(age, (int, float)) and age > 10
    cues: Dict[str, Any] = {}
    text = ""
    if t == "radar":
        std = _field(desc, "snr_db", "std")
        mean = _field(desc, "snr_db", "mean")
        if std is not None:
            lvl = _level(std, 0.8, 2.0)
            cues.update(motion=lvl, snr_std_db=round(std, 2))
            text = f"radar: {lvl} motion (SNR std {std:.1f} dB"
            text += f", mean {mean:.1f} dB)" if mean is not None else ")"
    elif t == "csi":
        std = _field(desc, "value", "std") or _field(desc, "amplitude_mean", "std")
        if std is not None:
            lvl = _level(std, 0.5, 2.0)
            cues.update(disturbance=lvl, amplitude_std=round(std, 3))
            text = f"wifi csi: {lvl} channel disturbance (amplitude std {std:.2f})"
    elif t in ("imu", "accelerometer"):
        std = _field(desc, "magnitude", "std") or _field(desc, "value", "std")
        if std is not None:
            state = "moving" if std >= 0.5 else "still"
            cues.update(motion=state, accel_std=round(std, 3))
            text = f"imu: {state} (|a| std {std:.2f})"
    elif t in ("ble_scan", "wifi_scan", "radio"):
        scan = desc.get("scan") or {}
        k = int(scan.get("emitters") or 0)
        cues["emitters"] = k
        kind = "wifi" if t == "wifi_scan" else "ble"
        text = f"{kind} scan: {k} emitter{'s' if k != 1 else ''}"
        top = scan.get("top") or []
        if top:
            s = top[0]
            label = s.get("name") or s.get("ssid") or s.get("beacon") or s.get("id")
            cues["strongest"] = {"id": s.get("id"), "label": label,
                                 "rssi": s.get("rssi")}
            text += f"; strongest {label} {s.get('rssi'):.0f} dBm"
    elif t == "microphone":
        std = _field(desc, "value", "std")
        if std is not None:
            lvl = _level(std, 300.0, 3000.0)
            cues.update(sound=lvl, level_std=round(std, 1))
            text = f"microphone: {lvl} sound level"
    elif t == "camera":
        text = f"camera: {n} frame{'s' if n != 1 else ''}"
    if not text:
        fields = desc.get("fields") or {}
        parts = [f"{k} {v.get('mean'):.3g}" for k, v in list(fields.items())[:3]
                 if isinstance(v, Mapping) and isinstance(v.get("mean"), (int, float))]
        text = f"{t or 'sensor'}: " + (", ".join(parts) if parts else f"{n} samples")
    if stale:
        text += f" (stale {age:.0f}s)"
        cues["stale"] = True
    return {"text": text[:TEXT_MAX], "cues": cues}


# ---------------------------------------------------------------------------
# Model-backed providers
# ---------------------------------------------------------------------------

def _default_factory(name: str, config: Dict[str, Any]) -> Any:
    from .models import model
    return model(name, config)


def _sub_window(window: SensorWindow, sensor_id: str) -> SensorWindow:
    return SensorWindow(start_timestamp=window.start_timestamp,
                        end_timestamp=window.end_timestamp,
                        samples={sensor_id: list(window.samples[sensor_id])},
                        preprocessing=dict(window.preprocessing))


@dataclass
class ModelTextProvider:
    """Runs a small model on one sensor type and renders its prediction."""

    name: str
    sensor_types: tuple
    models: tuple                                 # plugin names, chained
    render: Callable[[List[Any]], Optional[Dict[str, Any]]]
    config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    min_interval_s: float = 10.0
    factory: Optional[ModelFactory] = None
    _handles: Optional[List[Any]] = field(default=None, repr=False)
    _error: str = field(default="", repr=False)
    _last: Dict[str, float] = field(default_factory=dict, repr=False)
    _cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def _load(self) -> bool:
        if self._handles is not None:
            return bool(self._handles)
        try:
            make = self.factory or _default_factory
            self._handles = [make(m, dict(self.config.get(m) or {}))
                             for m in self.models]
        except Exception as exc:
            self._handles = []
            self._error = f"{type(exc).__name__}: {exc}"[:160]
            logger.info("text provider %s unavailable: %s", self.name, self._error)
        return bool(self._handles)

    def describe(self, window: SensorWindow, sensor_id: str,
                 now: float) -> Optional[Dict[str, Any]]:
        if now - self._last.get(sensor_id, -1e18) < self.min_interval_s:
            return self._cache.get(sensor_id)
        self._last[sensor_id] = now
        if not self._load():
            out = {"cues": {self.name: {"error": self._error}}}
            self._cache[sensor_id] = out
            return out
        sub = _sub_window(window, sensor_id)
        preds = []
        try:
            for handle in self._handles:
                pred = handle.predict(sub)
                preds.append(pred)
                dets = (pred.metadata or {}).get("detections")
                if dets is not None:
                    sub.preprocessing["detections"] = dets
        except Exception as exc:
            out = {"cues": {self.name: {"error": str(exc)[:160]}}}
            self._cache[sensor_id] = out
            return out
        out = self.render(preds)
        if out is not None:
            out["at"] = now
        self._cache[sensor_id] = out
        return out


def _render_speech(preds: List[Any]) -> Optional[Dict[str, Any]]:
    p = preds[-1]
    text = str((p.metadata or {}).get("text") or p.label or "").strip()
    err = (p.metadata or {}).get("error")
    if err:
        return {"cues": {"speech": {"error": err}}}
    if not text:
        return {"cues": {"speech": {"text": ""}}}
    return {"text": f'speech: "{text[:180]}"',
            "cues": {"speech": {"text": text,
                                "confidence": round(float(p.confidence or 0), 3)}}}


def _render_person(preds: List[Any]) -> Optional[Dict[str, Any]]:
    p = preds[-1]
    count = p.people_count
    if count is None:
        count = len((p.metadata or {}).get("detections") or []) or (
            1 if str(p.label).lower() in ("person", "present", "occupied") else 0)
    word = "nobody" if count == 0 else (
        "1 person" if count == 1 else f"{count} people")
    return {"text": f"camera: {word} visible",
            "cues": {"people": int(count),
                     "confidence": round(float(p.confidence or 0), 3)}}


def _render_face(preds: List[Any]) -> Optional[Dict[str, Any]]:
    det = preds[0]
    faces = len((det.metadata or {}).get("detections") or [])
    if faces == 0 or str(det.label) == "no_face":
        return {"cues": {"faces": 0}}
    cue: Dict[str, Any] = {"faces": faces}
    text = f"camera: {faces} face{'s' if faces != 1 else ''}"
    if len(preds) > 1:
        rec = preds[-1]
        who = str(rec.label or "")
        dist = (rec.metadata or {}).get("distance")
        if who and who != "person:unknown":
            name = who.split(":", 1)[-1]
            cue["identity"] = name
            text = f"face: {name}"
            if isinstance(dist, (int, float)):
                cue["distance"] = round(float(dist), 2)
                text += f" (distance {dist:.1f})"
        else:
            cue["identity"] = "unknown"
            text += ", not recognized"
    return {"text": text, "cues": cue}


def speech_provider(variant: str = "tiny.en", **kw: Any) -> ModelTextProvider:
    return ModelTextProvider(
        name="speech", sensor_types=("microphone",), models=("whisper-stt",),
        render=_render_speech,
        config={"whisper-stt": {"variant": variant, "input": "microphone"}},
        min_interval_s=kw.pop("min_interval_s", 15.0), **kw)


def person_provider(**kw: Any) -> ModelTextProvider:
    return ModelTextProvider(
        name="person", sensor_types=("camera",), models=("opencv-haar-person",),
        render=_render_person, min_interval_s=kw.pop("min_interval_s", 5.0), **kw)


def face_provider(recognizer_config: Optional[Dict[str, Any]] = None,
                  **kw: Any) -> ModelTextProvider:
    chain = ("opencv-haar-face",) + (
        ("pca-face-recognizer",) if recognizer_config else ())
    return ModelTextProvider(
        name="face", sensor_types=("camera",), models=chain, render=_render_face,
        config={"pca-face-recognizer": dict(recognizer_config or {})},
        min_interval_s=kw.pop("min_interval_s", 5.0), **kw)


# ---------------------------------------------------------------------------
# Describer
# ---------------------------------------------------------------------------

class TextualDescriber:
    """Rule sentences for every sensor + optional model providers."""

    def __init__(self, providers: Iterable[ModelTextProvider] = ()):
        self.providers = list(providers)

    @classmethod
    def default(cls, *, speech: bool = True, person: bool = True,
                face: bool = True,
                face_recognizer: Optional[Dict[str, Any]] = None,
                factory: Optional[ModelFactory] = None) -> "TextualDescriber":
        ps: List[ModelTextProvider] = []
        if speech:
            ps.append(speech_provider(factory=factory))
        if person:
            ps.append(person_provider(factory=factory))
        if face:
            ps.append(face_provider(face_recognizer, factory=factory))
        return cls(ps)

    def describe(self, window: SensorWindow,
                 physical: Optional[Mapping[str, Any]] = None,
                 now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        now = time.time() if now is None else now
        physical = physical if physical is not None else window_descriptors(
            window, "full", now)
        out: Dict[str, Dict[str, Any]] = {}
        for sid, desc in physical.items():
            entry = physical_sentence(desc)
            texts = [entry["text"]]
            stype = desc.get("type")
            for prov in self.providers:
                if stype not in prov.sensor_types or not window.samples.get(sid):
                    continue
                res = prov.describe(window, sid, now)
                if not res:
                    continue
                entry["cues"].update(res.get("cues") or {})
                if res.get("text"):
                    texts.append(res["text"])
            entry["text"] = "; ".join(texts)[:TEXT_MAX]
            out[sid] = entry
        return out

    def summary(self, described: Mapping[str, Mapping[str, Any]]) -> str:
        """All sensor sentences joined — a compact scene line."""
        return " | ".join(str(v.get("text")) for v in described.values()
                          if v.get("text"))[:TEXT_MAX * 4]


__all__ = ["TextualDescriber", "ModelTextProvider", "physical_sentence",
           "speech_provider", "person_provider", "face_provider"]
