"""Paper-faithful radar DSP for the MMW-HAT (BGT60TR13C) frame format.

Ports the per-frame view-map computation from the ICC multimodal
occupancy pipeline (``E2/occupancy_pipeline._radar_minute``) so that live
whispy payloads carry the exact same ``views`` (rd/ra/re/xy, 24x24,
log1p power) and clutter-masked ``snr_f`` the model was trained on.

Frame layout produced by the driver: ``(rx, chirps, samples) =
(3, 64, 128)`` uint12 ADC values packed 2-per-3-bytes behind a 12-byte
header (36864 payload bytes total).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

VIEW_NAMES = ("rd", "ra", "re", "xy")

# --- constants (identical to the pipeline) ----------------------------------
_HANN_R = np.hanning(128).astype(np.float32)     # range window (samples)
_HANN_D = np.hanning(64).astype(np.float32)      # doppler window (chirps)
_THETA = np.deg2rad(np.linspace(-60, 60, 25))
_PHASE = np.exp(1j * np.pi * np.sin(_THETA)).astype(np.complex64)
_RADIUS = np.arange(64) / 63
_IX = np.clip(((_RADIUS[:, None] * np.sin(_THETA) + 1) / 2 * 23)
              .round().astype(int), 0, 23)
_IY = np.clip((_RADIUS[:, None] * np.cos(_THETA) * 23)
              .round().astype(int), 0, 23)
_DC_ROWS = (10, 14)                              # masked doppler rows
_NEAR_RANGE = 2                                  # masked range cols


def read_uint12(raw: bytes) -> np.ndarray:
    """Packed 12-bit ADC words -> float32 samples."""
    data = np.frombuffer(raw, dtype=np.uint8)
    fst, mid, lst = np.reshape(
        data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    hi = (fst << 4) | (mid >> 4)
    lo = ((mid & 0x0F) << 8) | lst
    return np.reshape(np.concatenate(
        (hi[:, None], lo[:, None]), axis=1),
        2 * hi.shape[0]).astype(np.float32)


def decode_frame(raw: bytes) -> Optional[np.ndarray]:
    """Wire payload -> (rx, chirps, samples) float32 array or None."""
    if len(raw) < 12:
        return None
    if int.from_bytes(raw[0:4], "little") != 0:    # version
        return None
    data_len = int.from_bytes(raw[8:12], "little")
    if len(raw) - 12 != data_len:
        return None
    adc = read_uint12(raw[12:])
    if adc.shape[0] != 64 * 128 * 3:
        return None
    # ADC order is (chirp, sample, rx); handle shape is (rx, chirp, sample)
    return adc.reshape(64, 128, 3).transpose(2, 0, 1)


def _zoom1(m: np.ndarray, zy: float, zx: float) -> np.ndarray:
    """Separable bilinear resize — replicates scipy.ndimage.zoom(order=1)
    exactly (corner-aligned coords: out[i] samples in[i*(n-1)/(out-1)])."""
    ny, nx = int(round(m.shape[0] * zy)), int(round(m.shape[1] * zx))
    # scipy (grid_mode=False): in_coord = out_i * ((n_in-1)/(n_out-1));
    # the ratio is divided first, so the last index lands at n-1+eps.
    fy = (m.shape[0] - 1) / max(ny - 1, 1)
    fx = (m.shape[1] - 1) / max(nx - 1, 1)
    yi = np.arange(ny) * fy
    xi = np.arange(nx) * fx
    y0 = np.minimum(np.floor(yi).astype(int), m.shape[0] - 2)
    x0 = np.minimum(np.floor(xi).astype(int), m.shape[1] - 2)
    wy = np.clip(yi - y0, 0, 1)[:, None]        # (ny,1)
    wx = np.clip(xi - x0, 0, 1)[None, :]        # (1,nx)
    m64 = m.astype(np.float64)
    h = m64[y0] * (1 - wy) + m64[y0 + 1] * wy   # (ny, n_in_x)
    out = h[:, x0] * (1 - wx) + h[:, x0 + 1] * wx
    # mode='constant' cval=0: a coord strictly past the last sample -> 0
    # (the corner-aligned last index overshoots by ~1e-14 in fp64).
    out[yi > m.shape[0] - 1, :] = 0.0
    out[:, xi > m.shape[1] - 1] = 0.0
    return out.astype(np.float32)


def frame_views(arr: np.ndarray) -> Dict[str, Any]:
    """(3, 64, 128) ADC frame -> paper view maps + clutter-masked SNR.

    Returns {'views': (4,24,24) float32 [rd,ra,re,xy] log1p power,
             'snr': float dB}."""
    a = np.transpose(arr.astype(np.float32), (1, 2, 0))   # (ch,samp,rx)
    a = a - a.mean(axis=1, keepdims=True)                 # DC per chirp
    a = a * _HANN_R[None, :, None]
    R = np.fft.fft(a, axis=1)[:, :64, :]                  # (ch,rng,rx)
    D = np.fft.fftshift(np.fft.fft(R * _HANN_D[:, None, None],
                                 axis=0), axes=0)
    rd = np.mean(np.abs(D) ** 2, axis=2)                  # (64,64) d x r
    rd[:, :_NEAR_RANGE] = 0
    rd24 = _zoom1(np.log1p(rd), 24 / 64, 24 / 64)

    xy = np.zeros((24, 24), np.float32)
    maps = [rd24]
    for left in (0, 1):                                   # ra, re
        p = (np.abs(R[:, :, left]) ** 2
             + np.abs(R[:, :, 2]) ** 2).mean(axis=0)      # (64,)
        cross = (R[:, :, left] * R[:, :, 2].conj()).mean(axis=0)
        energy = np.maximum(
            p[:, None] + 2 * np.real(cross[:, None] * _PHASE), 0)
        energy[:_NEAR_RANGE, :] = 0
        maps.append(_zoom1(np.log1p(energy), 24 / 64, 24 / 25))
        if left == 0:                                     # xy from ra energy
            np.maximum.at(xy, (_IY.ravel(), _IX.ravel()),
                          energy.astype(np.float32).ravel())
    maps.append(np.log1p(xy).astype(np.float32))
    views = np.stack(maps).astype(np.float32)             # (4,24,24)

    # clutter-masked SNR proxy on the 24x24 RD power map
    mask = np.zeros((24, 24), bool)
    mask[_DC_ROWS[0]:_DC_ROWS[1], :] = True
    mask[:, :_NEAR_RANGE] = True
    pwr = np.expm1(rd24.astype(np.float32)).ravel()
    med = float(np.median(pwr)) + 1e-9
    peak = float(np.where(mask.ravel(), 0.0, pwr).max())
    snr = float(10.0 * np.log10(peak / med))
    return {"views": views, "snr": snr}


def payload_views(arr: np.ndarray) -> Dict[str, Any]:
    """Encode views for the wire payload (compact float16 base64)."""
    import base64
    out = frame_views(arr)
    v16 = out["views"].astype(np.float16)
    return {
        "views_b64": base64.b64encode(v16.tobytes()).decode("ascii"),
        "views_shape": list(v16.shape),
        "snr_f": round(out["snr"], 3),
    }


def decode_views(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    """Payload dict -> (4,24,24) float32 views or None."""
    import base64
    b = payload.get("views_b64")
    if not b:
        return None
    try:
        v = np.frombuffer(base64.b64decode(b), dtype=np.float16)
        return v.reshape(4, 24, 24).astype(np.float32)
    except Exception:
        return None
