"""Bounded timestamped sample streams.

A :class:`SampleStream` wraps a driver ``stream()`` iterator into a
bounded, thread-safe ring buffer that windowing/synchronization consumes.
Slow consumers drop the oldest samples rather than blocking ingestion
(§55: "sensor ingestion must not block").
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Iterator, List, Optional

from ..contracts import SensorSample


class StreamSubscription:
    """A per-consumer bounded queue fed by a :class:`SampleStream`.

    Each subscription receives every sample the parent stream ingests from
    the moment it is created. Reading a subscription drains *only* that
    subscription's private queue — it never removes samples from the
    parent buffer or from other subscriptions, so independent consumers
    (captures, telemetry, inference) cannot starve one another (§55).
    """

    def __init__(self, maxlen: int = 4096, name: str = "") -> None:
        self._buf: Deque[SensorSample] = deque(maxlen=maxlen)
        self._name = name
        self._lock = threading.Lock()
        self._dropped = 0
        self._closed = False

    def _put(self, sample: SensorSample) -> None:
        with self._lock:
            if self._closed:
                return
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(sample)

    def read(self) -> List[SensorSample]:
        """Return and clear the samples buffered for this consumer."""
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
            return out

    def peek(self) -> List[SensorSample]:
        with self._lock:
            return list(self._buf)

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._buf.clear()


class SampleStream:
    """Bounded ring buffer over a sensor sample iterator.

    ``maxlen`` bounds memory; ``put`` never blocks. Consumers call
    ``snapshot()`` or iterate ``drain()`` to read without stopping the
    producer thread. Consumers that must not disturb the shared buffer
    call :meth:`subscribe` for a private queue instead.
    """

    def __init__(self, source: Iterator[SensorSample], maxlen: int = 4096,
                 name: str = "") -> None:
        self._source = source
        self._buf: Deque[SensorSample] = deque(maxlen=maxlen)
        self._name = name
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._dropped = 0
        self._thread: Optional[threading.Thread] = None
        self._last: Optional[SensorSample] = None
        self._subs: List[StreamSubscription] = []

    # -- producer side ------------------------------------------------------
    def start(self, daemon: bool = True) -> "SampleStream":
        """Pump the source iterator into the buffer on a background thread."""
        if self._thread is not None:
            return self

        def _pump() -> None:
            try:
                for sample in self._source:
                    if self._closed.is_set():
                        break
                    self.put(sample)
            except Exception:
                # Source failures surface via health(); ingestion must not
                # kill the daemon (§55).
                pass

        self._thread = threading.Thread(
            target=_pump, name=f"whispy-stream-{self._name or 'sensor'}",
            daemon=daemon)
        self._thread.start()
        return self

    def put(self, sample: SensorSample) -> None:
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(sample)
            self._last = sample
            subs = list(self._subs)
        for sub in subs:
            sub._put(sample)

    # -- subscriptions (non-destructive consumers) ---------------------------
    def subscribe(self, maxlen: int = 4096,
                  name: str = "") -> StreamSubscription:
        """Return a private queue receiving every future sample.

        The subscription starts empty — it does not replay samples already
        in the shared buffer — so a consumer sees only samples ingested
        after it subscribes.
        """
        sub = StreamSubscription(maxlen=maxlen, name=name or self._name)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: StreamSubscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
        sub.close()

    # -- consumer side ------------------------------------------------------
    def snapshot(self) -> List[SensorSample]:
        with self._lock:
            return list(self._buf)

    def since(self, timestamp: float) -> List[SensorSample]:
        with self._lock:
            return [s for s in self._buf if s.timestamp >= timestamp]

    def window(self, start: float, end: float) -> List[SensorSample]:
        with self._lock:
            return [s for s in self._buf if start <= s.timestamp <= end]

    def drain(self) -> List[SensorSample]:
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
            return out

    @property
    def last(self) -> Optional[SensorSample]:
        return self._last

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def last_timestamp(self) -> Optional[float]:
        return self._last.timestamp if self._last else None

    def stale(self, max_age_s: float, now: Optional[float] = None) -> bool:
        """True when the newest sample is older than ``max_age_s``."""
        if self._last is None:
            return True
        return ((now if now is not None else time.time()) - self._last.timestamp) > max_age_s

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            subs = list(self._subs)
            self._subs.clear()
        for sub in subs:
            sub.close()

    def __enter__(self) -> "SampleStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["SampleStream", "StreamSubscription"]
