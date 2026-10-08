"""One-call local sensing snapshot: physical + textual descriptors.

    import whispy
    snap = whispy.snapshot(seconds=3)
    print(snap["scene"])                 # "radar: high motion ... | ble scan: 11 emitters ..."
    snap["sensors"]["radar-a316"]        # {"type", "n", "fields", "text", "cues"}

Opens every (or the selected) local sensor for ``seconds``, cuts one
synchronized window, describes it, and closes the sensors again. No
cloud, no daemon required — the same descriptors a Thoth node uplinks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

from .descriptors import window_descriptors


def snapshot(seconds: float = 3.0, *, device: Any = None,
             sensors: Optional[Iterable[str]] = None,
             detail: str = "descriptors", text: bool = True,
             describer: Any = None) -> Dict[str, Any]:
    """Capture ``seconds`` of local sensing and describe it."""
    from .streams import SampleStream
    from .synchronization import WindowSynchronizer

    if device is None:
        from .devices import local
        device = local()
    wanted = set(sensors) if sensors is not None else None
    streams: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    try:
        for s in device.sensors():
            if wanted is not None and s.id not in wanted and s.type not in wanted:
                continue
            try:
                handle = device.sensor(s.id)
                streams[s.id] = SampleStream(handle.stream, maxlen=4096,
                                             name=s.id).start()
            except Exception as exc:
                errors[s.id] = str(exc)[:160]
        time.sleep(max(0.1, float(seconds)))
        now = time.time()
        sync = WindowSynchronizer(streams, expected=list(streams),
                                  stale_after_s=max(5.0, seconds * 2))
        window = sync.rolling(float(seconds))
    finally:
        for st in streams.values():
            try:
                st.close()
            except Exception:
                pass
    physical = window_descriptors(window, detail, now)
    out: Dict[str, Any] = {"timestamp": now, "seconds": float(seconds),
                           "sensors": physical, "scene": None,
                           "errors": errors}
    if text:
        if describer is None:
            from .textual import TextualDescriber
            describer = TextualDescriber.default()
        described = describer.describe(window, window_descriptors(
            window, "full", now), now)
        for sid, t in described.items():
            if sid in physical:
                physical[sid]["text"] = t.get("text")
                if t.get("cues"):
                    physical[sid]["cues"] = t["cues"]
        out["scene"] = describer.summary(described)
    return out


__all__ = ["snapshot"]
