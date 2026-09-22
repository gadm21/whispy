"""Environmental sensor (Sense HAT-style) helpers."""

from __future__ import annotations

import numpy as np


def series(rows: list, key: str) -> np.ndarray:
    """Extract a numeric series for one sensor key from sense rows."""
    vals = []
    for row in rows:
        v = row.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return np.asarray(vals)


def keys(rows: list) -> list:
    """Union of keys present across sense rows."""
    out = []
    for row in rows:
        for k in row:
            if k not in out:
                out.append(k)
    return out
