"""Context cache — a client's local mirror of Brain's context model.

Polls ``/v1/context/snapshot`` and serves it locally so consumers read
context without hammering Brain. Predictions arrive as evidence; states
are derived server-side and cached here.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional


class ContextCache:
    """TTL-cached snapshot of the user's context model."""

    def __init__(self, fetcher: Callable[[], Dict[str, Any]],
                 ttl_s: float = 5.0):
        self._fetcher = fetcher
        self._ttl = ttl_s
        self._lock = threading.Lock()
        self._snapshot: Dict[str, Any] = {
            "entities": [], "relationships": [], "states": [],
            "generated_at": 0.0,
        }
        self._fetched_at = 0.0
        self._last_error: Optional[str] = None

    def refresh(self) -> Dict[str, Any]:
        try:
            snap = self._fetcher()
            with self._lock:
                self._snapshot = snap
                self._fetched_at = time.time()
                self._last_error = None
            return snap
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                return dict(self._snapshot)

    def snapshot(self, max_age_s: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            age = time.time() - self._fetched_at
            stale = self._fetched_at == 0 or age > (max_age_s or self._ttl)
        if stale:
            return self.refresh()
        with self._lock:
            return dict(self._snapshot)

    def state(self, key: str, entity_id: str = "") -> Optional[Dict[str, Any]]:
        for s in self.snapshot().get("states") or []:
            if s.get("key") == key and (not entity_id
                                        or s.get("entity_id") == entity_id):
                return s
        return None

    def entities(self, kind: str = "") -> List[Dict[str, Any]]:
        ents = self.snapshot().get("entities") or []
        return [e for e in ents if not kind or e.get("kind") == kind]

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
