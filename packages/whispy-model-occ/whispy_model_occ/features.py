"""5-second window features for the ICC RF-MoE occupancy model.

Exact reimplementation of ``occupancy_pipeline.minute_windows`` (E2):
the CSI block is verified against the paper's ``features_5s.csv`` to
float16 tolerance; the radar block consumes the ``views``
(rd/ra/re/xy 24x24 log1p power maps) + ``snr_f`` produced by
``whispy_sensor_mmwhat.dsp.frame_views`` — the same math the pipeline
applied to recorded raw frames.

Layout (positions match the pipeline's FUSION_COLS):
  CSI_COLS   = csi_amp_{mean,std,q90}, csi_rv_{mean,q90,max},
               csi_pcv{1..3}, csi_ok                      (10)
  RADAR_COLS = snr_{max,mean}, rad_pcv{1..3},
               <view>_{mean,p90,std90,delta,peak} x4, rad_ok (26)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

SEC_HZ = 100          # guaranteed CSI grid
N_SUB = 52            # CSI subcarriers kept by the paper's mask
WIN_S = 5             # prediction window seconds
ROLL_W = 20           # 0.2 s causal rolling variance
MIN_COV = 0.8         # min fraction of covered seconds per window
MIN_RADAR_FRAMES = 5  # min radar frames per 5-s window

CSI_AMP = ["csi_amp_mean", "csi_amp_std", "csi_amp_q90"]
CSI_RV = ["csi_rv_mean", "csi_rv_q90", "csi_rv_max"]
CSI_PCV = ["csi_pcv1", "csi_pcv2", "csi_pcv3"]
CSI_COLS = CSI_AMP + CSI_RV + CSI_PCV + ["csi_ok"]

VIEW_NAMES = ("rd", "ra", "re", "xy")
_DESC = ("mean", "p90", "std90", "delta", "peak")
RADAR_COLS = (["snr_max", "snr_mean", "rad_pcv1", "rad_pcv2", "rad_pcv3"]
              + [f"{v}_{s}" for v in VIEW_NAMES for s in _DESC]
              + ["rad_ok"])
FUSION_COLS = CSI_COLS + RADAR_COLS

# ESP32 CSI 64->52 subcarrier mask (common.CSI_SUBCARRIER_MASK)
CSI_MASK = np.array(
    [False] * 6 + [True] * 26 + [False] + [True] * 26 + [False] * 5,
    dtype=bool)


def csi_amplitude(sample) -> Optional[np.ndarray]:
    """SensorSample payload -> (52,) amplitude vector or None."""
    p = sample.payload if isinstance(sample.payload, dict) else {}
    iq = p.get("iq")
    if iq is None:
        return None
    iq = np.asarray(iq, dtype=np.float64)
    if iq.size < 128:
        return None
    real = iq[0::2][:64][CSI_MASK]
    imag = iq[1::2][:64][CSI_MASK]
    return np.hypot(real, imag)


def rollvar(x: np.ndarray, w: int = ROLL_W) -> np.ndarray:
    """Causal trailing variance along axis 0 — pipeline's _rollvar
    verbatim ((T,S) -> (T,S))."""
    z = np.zeros((1, x.shape[1]), dtype=np.float64)
    cs = np.concatenate([z, np.cumsum(x, axis=0)], axis=0)
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=0)], axis=0)
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float64)[:, None]
    mean = (cs[hi] - cs[lo]) / cnt
    msq = (cs2[hi] - cs2[lo]) / cnt
    return np.clip(msq - mean * mean, 0, None)


def interp_seconds(mat: np.ndarray, ok: np.ndarray) -> Optional[np.ndarray]:
    """Linear interpolation of invalid seconds — pipeline's
    _interp_seconds (clamps at the context edges)."""
    mat = np.asarray(mat, np.float32)
    ok = np.asarray(ok, bool)
    if not ok.any():
        return None
    out = mat.copy()
    vi = np.flatnonzero(ok)
    for col in range(mat.shape[1]):
        out[:, col] = np.interp(np.arange(len(mat)), vi,
                                mat[vi, col])
    return out


def ffill_rows(rows: np.ndarray) -> Optional[np.ndarray]:
    """Pipeline's _ffill_rows: each row takes the *next* fully-finite row
    (searchsorted picks the nearest valid index at-or-after), edge rows
    clamp to the nearest valid."""
    valid = np.isfinite(rows).all(axis=1)
    if not valid.any():
        return None
    vi = np.flatnonzero(valid)
    fill = np.clip(np.searchsorted(vi, np.arange(len(rows))),
                   0, len(vi) - 1)
    return rows[vi[fill]]


# ---------------------------------------------------------------------------
# CSI 5-s window block.
#   blocks_ctx: (100,52) amplitude blocks or None per invalid second,
#               ending at the window's last second; the final n_win
#               entries form the window, earlier ones are context for
#               the paper's (forward-looking) fill.
#   dots_ctx:   (M,104) per-second mean||std dot rows, NaN where invalid.
#   ok_ctx:     (M,) bool — same seconds as dots_ctx.
#   pca_mean, pca_comp: fitted CSI PCA (mean_, components_).
# ---------------------------------------------------------------------------
def csi_window_features(blocks_ctx: List[Optional[np.ndarray]],
                        dots_ctx: np.ndarray, ok_ctx: np.ndarray,
                        pca_mean: np.ndarray, pca_comp: np.ndarray,
                        n_win: int = WIN_S) -> np.ndarray:
    x = np.full(len(CSI_COLS), np.nan, np.float32)
    secs = blocks_ctx[-n_win:]
    valid = np.array([s is not None for s in secs])
    csi_ok = bool(valid.mean() >= MIN_COV)
    x[9] = float(csi_ok)
    if not csi_ok:
        return x

    blocks = [s if s is not None else np.full((SEC_HZ, N_SUB), np.nan,
                                              np.float32)
              for s in secs]
    a = np.concatenate(blocks, axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        x[0] = np.nanmean(a)
        x[1] = np.nanstd(a)
        x[2] = np.nanquantile(a, 0.9)

    ctx_blocks = [s if s is not None else np.full((SEC_HZ, N_SUB), np.nan,
                                                  np.float32)
                  for s in blocks_ctx]
    rows_f = ffill_rows(np.concatenate(ctx_blocks, axis=0)
                        .astype(np.float32))
    if rows_f is not None:
        w_rows = rows_f[-n_win * SEC_HZ:]
        rv = np.log1p(rollvar(np.log1p(w_rows.astype(np.float64))))
        x[3] = rv.mean()
        x[4] = np.quantile(rv, 0.9)
        x[5] = rv.max()

    dots_f = interp_seconds(np.asarray(dots_ctx, np.float32), ok_ctx)
    if dots_f is not None and len(dots_f) >= n_win:
        proj = (dots_f - pca_mean) @ pca_comp.T
        x[6:9] = np.var(proj[-n_win:], axis=0)
    return x


# ---------------------------------------------------------------------------
# Radar 5-s window block — views semantics identical to the pipeline:
#   sel = frames whose timestamp falls in the window
#   per view: mean_img.mean / q90(mean_img) / q90(std_img) /
#             mean |frame-to-frame diff| / mean_img.max
#   rad_pcv: variance of per-second mean-RD PCA projections
# ---------------------------------------------------------------------------
def radar_window_features(views: np.ndarray, snr: np.ndarray,
                          rd_sec_ctx: np.ndarray, sec_has_radar_ctx,
                          pca_rad=None, n_win: int = WIN_S,
                          min_frames: int = MIN_RADAR_FRAMES) -> np.ndarray:
    """views (n,4,24,24), snr (n,) for frames *inside* the window;
    rd_sec_ctx (M,576) per-second mean-RD rows + sec_has_radar_ctx (M,)
    covering >= n_win seconds ending at the window's last second
    (context mimics the paper's minute-level interpolation)."""
    x = np.full(len(RADAR_COLS), np.nan, np.float32)
    rad_ok = bool(len(snr) >= min_frames)
    x[25] = float(rad_ok)
    if not rad_ok:
        return x
    snr = np.asarray(snr, np.float32)
    x[0] = float(np.nanmax(snr))
    x[1] = float(np.nanmean(snr))
    off = 5
    for vi in range(4):
        mp = views[:, vi].astype(np.float32)
        mean_img = mp.mean(axis=0)
        std_img = mp.std(axis=0)
        delta = (float(np.abs(np.diff(mp, axis=0)).mean())
                 if len(mp) > 1 else 0.0)
        x[off:off + 5] = [float(mean_img.mean()),
                          float(np.quantile(mean_img, 0.9)),
                          float(np.quantile(std_img, 0.9)),
                          delta,
                          float(mean_img.max())]
        off += 5
    ok_ctx = np.asarray(sec_has_radar_ctx, bool)
    if (pca_rad is not None and len(ok_ctx) >= n_win
            and ok_ctx[-n_win:].mean() >= MIN_COV):
        rds_f = interp_seconds(np.asarray(rd_sec_ctx, np.float32), ok_ctx)
        if rds_f is not None:
            proj = (rds_f - np.asarray(pca_rad.mean_, np.float32)) \
                @ np.asarray(pca_rad.components_, np.float32).T
            x[2:5] = np.var(proj[-n_win:], axis=0)
    return x


def fusion_window(csi_x: np.ndarray, rad_x: np.ndarray) -> np.ndarray:
    """CSI_COLS ++ RADAR_COLS (FUSION_COLS order)."""
    return np.concatenate([csi_x, rad_x]).astype(np.float32)
