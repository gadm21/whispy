"""Multi-sensor window synchronization.

Builds :class:`~whispy.contracts.SensorWindow` objects from a set of
:class:`~whispy.streams.SampleStream` buffers. Missing and stale
modalities are recorded explicitly (§15, §47) — a fusion processor can
never mistake absent data for valid zeros.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

from ..contracts import ModalityState, SensorWindow
from ..streams import SampleStream


class WindowSynchronizer:
    """Cuts synchronized windows across named sample streams.

    Parameters
    ----------
    streams:
        Mapping of sensor_id → SampleStream.
    expected:
        Sensor ids that *should* be present. Streams not in ``streams``
        are reported ``missing``; streams whose newest sample is older
        than ``stale_after_s`` are reported ``stale``.
    stale_after_s:
        Age threshold for the stale marker.
    """

    def __init__(self, streams: Mapping[str, SampleStream],
                 expected: Optional[List[str]] = None,
                 stale_after_s: float = 5.0) -> None:
        self._streams = dict(streams)
        self._expected = list(expected) if expected is not None else list(streams)
        self._stale_after = stale_after_s

    def add_stream(self, sensor_id: str, stream: SampleStream) -> None:
        self._streams[sensor_id] = stream
        if sensor_id not in self._expected:
            self._expected.append(sensor_id)

    def remove_stream(self, sensor_id: str) -> None:
        self._streams.pop(sensor_id, None)

    def cut(self, start: float, end: float,
            now: Optional[float] = None) -> SensorWindow:
        """Produce a window for [start, end] with modality markers."""
        now = now if now is not None else time.time()
        samples: Dict[str, Any] = {}
        modalities: Dict[str, ModalityState] = {}

        for sensor_id in self._expected:
            stream = self._streams.get(sensor_id)
            if stream is None:
                modalities[sensor_id] = ModalityState(
                    sensor_id=sensor_id, state="missing",
                    detail="no stream for expected sensor")
                continue
            chunk = stream.window(start, end)
            last_ts = stream.last_timestamp
            if not chunk:
                state = "stale" if (last_ts is not None and now - last_ts > self._stale_after) else "missing"
                modalities[sensor_id] = ModalityState(
                    sensor_id=sensor_id, state=state,
                    last_sample_timestamp=last_ts,
                    detail="no samples inside window")
                samples[sensor_id] = []
                continue
            stale = last_ts is not None and (now - last_ts) > self._stale_after
            modalities[sensor_id] = ModalityState(
                sensor_id=sensor_id,
                state="stale" if stale else "ok",
                last_sample_timestamp=last_ts)
            samples[sensor_id] = chunk

        return SensorWindow(
            start_timestamp=start,
            end_timestamp=end,
            samples=samples,
            modalities=modalities,
            timing={"cut_at": now, "stale_after_s": self._stale_after},
        )

    def rolling(self, duration_s: float,
                now: Optional[float] = None) -> SensorWindow:
        """Cut a trailing window of ``duration_s`` ending now."""
        now = now if now is not None else time.time()
        return self.cut(now - duration_s, now, now=now)


__all__ = ["WindowSynchronizer"]
