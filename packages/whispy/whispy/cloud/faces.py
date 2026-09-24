"""FaceGallery — client helper for Brain's face asset store.

Pulls the PCA basis (``GET /v1/faces/basis``) and the enrolled gallery
(``GET /v1/faces/gallery``) so an edge ``EigenfaceRecognizer`` can
match live faces against known persons. Mirrors ContextCache: inject a
fetcher, TTL-cached, keeps the last good snapshot on fetch error.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class FaceGallery:
    """TTL-cached view of Brain's face basis + enrolled projections.

    ``fetcher`` must return the ``/v1/faces/gallery`` payload::

        {"basis_id": "...", "image_size": 64, "max_distance": 12.3,
         "persons": [{"name": "gad", "projection": [...]}, ...]}

    ``basis_fetcher`` (optional) returns the ``/v1/faces/basis`` .npz
    bytes; pass it when the recognizer needs the basis too.
    """

    def __init__(self,
                 fetcher: Callable[[], Dict[str, Any]],
                 basis_fetcher: Optional[Callable[[], bytes]] = None,
                 ttl_s: float = 300.0):
        self._fetch = fetcher
        self._fetch_basis = basis_fetcher
        self._ttl = float(ttl_s)
        self._gallery: Dict[str, Any] = {}
        self._basis: Optional[bytes] = None
        self._fetched_at = 0.0
        self.last_error: Optional[str] = None

    def _stale(self) -> bool:
        return not self._fetched_at or \
            (time.time() - self._fetched_at) > self._ttl

    def refresh(self) -> Dict[str, Any]:
        try:
            self._gallery = dict(self._fetch() or {})
            if self._fetch_basis is not None:
                self._basis = self._fetch_basis()
            self._fetched_at = time.time()
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
        return self._gallery

    def gallery(self) -> Dict[str, Any]:
        if self._stale():
            self.refresh()
        return self._gallery

    def basis_bytes(self) -> Optional[bytes]:
        if self._stale():
            self.refresh()
        return self._basis

    def persons(self) -> List[Dict[str, Any]]:
        return list(self.gallery().get("persons") or [])

    def projections(self) -> Dict[str, List[List[float]]]:
        """{name: [projection, ...]} — ready for EigenfaceRecognizer."""
        out: Dict[str, List[List[float]]] = {}
        for row in self.persons():
            name, proj = row.get("name"), row.get("projection")
            if name and proj:
                out.setdefault(str(name), []).append(list(proj))
        return out

    def max_distance(self) -> float:
        return float(self.gallery().get("max_distance") or 0.0)


__all__ = ["FaceGallery"]
