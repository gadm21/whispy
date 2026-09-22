"""Wi-Fi CSI parsing and subcarrier processing (ported from whispy.core)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Valid OFDM subcarriers for ESP32 20 MHz CSI (64 raw, 52 usable).
CSI_SUBCARRIER_MASK = np.array([
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
], dtype=bool)

N_SUBCARRIERS_RAW = 64
N_SUBCARRIERS_VALID = int(CSI_SUBCARRIER_MASK.sum())  # 52


def parse_csi_line(text: str) -> Optional[dict]:
    """Parse a single ``CSI_DATA,...`` line into components.

    Returns dict with keys: rssi (float), timestamp (int),
    real (64,), imag (64,) — or None if malformed.
    """
    try:
        parts = text.split(",", 14)
        if len(parts) < 15 or not parts[0].startswith("CSI_DATA"):
            return None
        rssi = float(parts[3])
        ts = int(parts[9])
        csi_str = parts[14].strip()
        vals = [int(x) for x in csi_str[1:-1].split(",")]
        if len(vals) != 128:
            return None
        imag = np.array(vals[0::2], dtype=np.float64)
        real = np.array(vals[1::2], dtype=np.float64)
        return {"rssi": rssi, "timestamp": ts, "real": real, "imag": imag}
    except Exception:
        return None


def amplitude(sample: dict, valid_only: bool = True) -> np.ndarray:
    """Subcarrier amplitudes from a parsed CSI sample."""
    mag = np.sqrt(sample["real"] ** 2 + sample["imag"] ** 2)
    return mag[CSI_SUBCARRIER_MASK] if valid_only else mag


def phase(sample: dict, valid_only: bool = True) -> np.ndarray:
    """Subcarrier phases from a parsed CSI sample."""
    ph = np.arctan2(sample["imag"], sample["real"])
    return ph[CSI_SUBCARRIER_MASK] if valid_only else ph


def amplitude_matrix(samples: list) -> np.ndarray:
    """(N, 52) amplitude matrix from parsed CSI samples."""
    rows = [amplitude(s) for s in samples if isinstance(s, dict) and "real" in s]
    return np.stack(rows) if rows else np.empty((0, N_SUBCARRIERS_VALID))
