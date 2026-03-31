"""Core CSI processing primitives.

All functions operate on numpy arrays and have no side effects.
This module has zero dependency on serial, torch, or any optional package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =========================================================================
# Subcarrier mask — ESP32 HT20 LLTF (64 → 52 valid tones)
# =========================================================================
CSI_SUBCARRIER_MASK = np.array([
    # 0-5: lower guard (6 nulls)
    0, 0, 0, 0, 0, 0,
    # 6-31: 26 negative-frequency tones (including pilots at 11, 25)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # 32: DC null
    0,
    # 33-58: 26 positive-frequency tones (including pilots at 39, 53)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # 59-63: upper guard (5 nulls)
    0, 0, 0, 0, 0,
], dtype=bool)  # 52 True values

N_SUBCARRIERS_RAW = 64
N_SUBCARRIERS_VALID = int(CSI_SUBCARRIER_MASK.sum())  # 52


# =========================================================================
# Parsing
# =========================================================================
def parse_csi_line(text: str) -> dict[str, Any] | None:
    """Parse a single ``CSI_DATA,...`` line into components.

    Returns
    -------
    dict with keys: rssi (float), timestamp (int), real (64,), imag (64,)
    or None if the line is malformed.
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


def parse_csi_file(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load a raw ESP32 CSI CSV and return structured arrays.

    Returns
    -------
    dict with keys:
        mag (N,64), phase (N,64), real (N,64), imag (N,64),
        rssi (N,), timestamp (N,), n_errors (int)
    """
    filepath = str(filepath)
    df = pd.read_csv(filepath, header=0, on_bad_lines="skip", low_memory=False)

    mask = df["type"].str.startswith("CSI_DATA", na=False)
    df = df[mask]

    ts_series = pd.to_numeric(df["local_timestamp"], errors="coerce")
    timestamps = ts_series.fillna(0).values.astype(np.int64)

    real_list, imag_list, rssi_list = [], [], []
    n_errors = 0
    for i, row_data in enumerate(df["data"].values):
        try:
            vals = [int(x) for x in row_data[1:-1].split(",")]
            if len(vals) != 128:
                n_errors += 1
                continue
            imag_list.append(vals[0::2])
            real_list.append(vals[1::2])
            rssi_list.append(df["rssi"].iloc[i])
        except Exception:
            n_errors += 1

    real = np.array(real_list, dtype=np.float64)
    imag = np.array(imag_list, dtype=np.float64)
    mag = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)
    rssi = np.array(rssi_list, dtype=np.float64)
    ts = timestamps[: len(real)]

    return {
        "mag": mag,
        "phase": phase,
        "real": real,
        "imag": imag,
        "rssi": rssi,
        "timestamp": ts,
        "n_errors": n_errors,
    }


# =========================================================================
# Resampling
# =========================================================================
def resample(
    data: dict[str, np.ndarray],
    target_sr: int = 150,
    keys: tuple[str, ...] = ("mag", "phase"),
) -> dict[str, np.ndarray]:
    """Resample CSI arrays to a uniform sampling rate via bin averaging.

    Parameters
    ----------
    data : dict from ``parse_csi_file`` (must contain 'timestamp' in µs).
    target_sr : target sampling rate in Hz.
    keys : which 2-D array keys to resample.

    Returns
    -------
    dict with resampled arrays + 'resampling_meta' sub-dict.
    """
    ts_us = data["timestamp"]
    n_orig = len(ts_us)
    if n_orig < 2:
        data["resampling_meta"] = {"status": "too_few_samples"}
        return data

    ts_sec = ts_us.astype(np.float64) / 1_000_000
    start, end = ts_sec[0], ts_sec[-1]
    duration = end - start
    if duration < 0.01:
        data["resampling_meta"] = {"status": "too_short"}
        return data

    actual_sr = n_orig / duration
    n_out = max(2, int(np.ceil(duration * target_sr)))
    target_t = start + np.arange(n_out) / target_sr
    dt = 1.0 / target_sr

    bin_edges = target_t - dt / 2
    bin_idx = np.clip(np.searchsorted(bin_edges, ts_sec, side="right") - 1, 0, n_out - 1)
    counts = np.bincount(bin_idx, minlength=n_out).astype(np.float64)

    out = {}
    for k in keys:
        src = data[k]
        if src.ndim != 2:
            continue
        acc = np.zeros((n_out, src.shape[1]), dtype=np.float64)
        np.add.at(acc, bin_idx, src)
        pop = counts > 0
        acc[pop] /= counts[pop, None]
        # interpolate empties
        empty = ~pop
        if empty.any():
            valid = np.where(pop)[0]
            if len(valid) >= 2:
                for c in range(src.shape[1]):
                    acc[empty, c] = np.interp(target_t[empty], target_t[valid], acc[valid, c])
        out[k] = acc

    # resample 1-D arrays
    if "rssi" in data:
        rssi_acc = np.zeros(n_out, dtype=np.float64)
        np.add.at(rssi_acc, bin_idx, data["rssi"])
        pop = counts > 0
        rssi_acc[pop] /= counts[pop]
        empty = ~pop
        if empty.any():
            valid = np.where(pop)[0]
            if len(valid) >= 2:
                rssi_acc[empty] = np.interp(target_t[empty], target_t[valid], rssi_acc[valid])
        out["rssi"] = rssi_acc

    out["timestamp"] = (target_t * 1_000_000).astype(np.int64)
    n_empty = int((counts == 0).sum())
    out["resampling_meta"] = {
        "status": "ok",
        "original_samples": n_orig,
        "resampled_samples": n_out,
        "actual_sr": round(actual_sr, 2),
        "target_sr": target_sr,
        "sr_ratio": round(actual_sr / target_sr, 4),
        "duration_sec": round(duration, 3),
        "empty_bins": n_empty,
        "empty_pct": round(100 * n_empty / n_out, 2),
    }
    # carry forward unprocessed keys
    for k, v in data.items():
        if k not in out:
            out[k] = v
    return out


# =========================================================================
# Subcarrier selection
# =========================================================================
def select_subcarriers(
    data: dict[str, np.ndarray],
    mask: np.ndarray = CSI_SUBCARRIER_MASK,
    keys: tuple[str, ...] = ("mag", "phase"),
) -> dict[str, np.ndarray]:
    """Apply boolean subcarrier mask (64 → 52) to specified keys."""
    out = dict(data)
    for k in keys:
        if k in out and out[k].ndim == 2 and out[k].shape[1] == len(mask):
            out[k] = out[k][:, mask]
    return out


# =========================================================================
# Rolling variance
# =========================================================================
def rolling_variance(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling variance per column using cumulative sums — O(N).

    Parameters
    ----------
    arr : (N, C) float array.
    window : sliding window size.

    Returns
    -------
    (N, C) float array (first window-1 values use partial windows).
    """
    if window <= 1:
        return np.zeros_like(arr)
    n = arr.shape[0]
    cs = np.cumsum(arr, axis=0)
    cs2 = np.cumsum(arr**2, axis=0)
    cs = np.vstack([np.zeros((1, arr.shape[1])), cs])
    cs2 = np.vstack([np.zeros((1, arr.shape[1])), cs2])
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - window, 0, None)
    cnt = (hi - lo).reshape(-1, 1)
    mean = (cs[hi] - cs[lo]) / cnt
    mean_sq = (cs2[hi] - cs2[lo]) / cnt
    return np.clip(mean_sq - mean**2, 0, None)


# =========================================================================
# Windowing
# =========================================================================
def window_array(
    arr: np.ndarray,
    length: int,
    stride: int | None = None,
    flatten: bool = True,
) -> np.ndarray | None:
    """Slice a 2-D array into fixed-length windows.

    Returns
    -------
    (n_windows, length*C) if flatten else (n_windows, length, C).
    None if too few samples.
    """
    stride = stride or length
    n, c = arr.shape
    n_win = (n - length) // stride + 1
    if n_win <= 0:
        return None
    wins = np.array([arr[i * stride : i * stride + length] for i in range(n_win)])
    if flatten:
        wins = wins.reshape(n_win, -1)
    return wins


# =========================================================================
# Dataset metadata helpers
# =========================================================================
METADATA_FILENAME = "dataset_metadata.json"


def load_metadata(root_dir: str | Path) -> dict:
    """Load dataset_metadata.json from a directory."""
    p = Path(root_dir) / METADATA_FILENAME
    if not p.is_file():
        raise FileNotFoundError(f"No {METADATA_FILENAME} in {root_dir}")
    with open(p) as f:
        return json.load(f)


# =========================================================================
# Reproducibility
# =========================================================================
def set_seed(seed: int = 42) -> None:
    """Set random seeds for numpy, random, and torch (if available)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
