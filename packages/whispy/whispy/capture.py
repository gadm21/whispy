"""Capture sessions — orchestrated pre-roll/post-roll window capture.

Used by demos and tests that must coordinate an actuator with a sensor
window (e.g. speak → remote microphone → STT)::

    session = whispy.capture_window(mic, pre_roll=1.0)
    session.start()                       # consumes + pre-roll silence
    speaker.execute(Speak("test seven"))
    window = session.finish(post_roll=1.5)

``finish`` returns a :class:`SensorWindow` covering the whole session —
pre-roll, the action, and post-roll — so the model sees the complete
utterance including its onset.
"""

from __future__ import annotations

import time
from typing import Optional

from .contracts import ModalityState, SensorWindow
from .devices.base import SensorHandle
from .streams import SampleStream


class CaptureSession:
    """A timed capture over one sensor handle.

    ``start()`` begins consuming the stream and waits ``pre_roll``
    seconds so the retained window includes silence/context before the
    orchestrated event. ``finish(post_roll)`` keeps recording for
    ``post_roll`` seconds after the caller's action completes, then stops
    ingestion and returns the full window.
    """

    def __init__(self, handle: SensorHandle, pre_roll: float = 0.0,
                 maxlen: int = 8192):
        self._handle = handle
        self._pre_roll = max(0.0, pre_roll)
        self._stream: Optional[SampleStream] = None
        self._maxlen = maxlen
        self._started_at: Optional[float] = None

    @property
    def sensor_id(self) -> str:
        return self._handle.info.id

    def start(self, warmup_timeout: float = 15.0) -> "CaptureSession":
        """Begin consuming the sensor; blocks for the pre-roll period.

        Waits until the stream has delivered its first sample before the
        pre-roll clock starts — a LAN handle's priming poll can take
        seconds on a loaded node, and without this a short session can
        finish before any sample arrives.
        """
        self._stream = SampleStream(
            self._handle.stream(), maxlen=self._maxlen,
            name=f"capture:{self.sensor_id}")
        self._stream.start()
        deadline = time.time() + warmup_timeout
        while not self._stream.snapshot():
            if time.time() > deadline:
                break
            time.sleep(0.05)
        self._started_at = time.time()
        if self._pre_roll:
            time.sleep(self._pre_roll)
        return self

    def finish(self, post_roll: float = 0.0) -> SensorWindow:
        """Stop after ``post_roll`` seconds; return the captured window."""
        if self._stream is None or self._started_at is None:
            raise RuntimeError("CaptureSession.finish() before start()")
        if post_roll > 0:
            time.sleep(post_roll)
        end = time.time()
        # Drop backlog: a LAN handle replays the remote daemon's ring
        # buffer on first poll, so the snapshot can contain minutes of
        # samples produced before this session started. Keep only what
        # the session actually captured (30 s tolerance for clock skew
        # between producer and consumer).
        cutoff = self._started_at - 30.0
        samples = [s for s in self._stream.snapshot()
                   if s.timestamp >= cutoff]
        self._stream.close()
        self._stream = None
        sid = self.sensor_id
        return SensorWindow(
            start_timestamp=self._started_at,
            end_timestamp=end,
            samples={sid: samples},
            modalities={sid: ModalityState(
                sensor_id=sid,
                state="ok" if samples else "missing",
                last_sample_timestamp=samples[-1].timestamp if samples else None,
                detail="" if samples else "no samples captured")},
            timing={"pre_roll": self._pre_roll, "post_roll": post_roll},
        )

    def __enter__(self) -> "CaptureSession":
        return self.start()

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def capture_window(handle: SensorHandle, pre_roll: float = 0.0,
                   maxlen: int = 8192) -> CaptureSession:
    """Create a capture session over a sensor handle."""
    return CaptureSession(handle, pre_roll=pre_roll, maxlen=maxlen)


__all__ = ["CaptureSession", "capture_window"]
