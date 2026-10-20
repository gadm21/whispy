"""``occ-rf-hier`` — the icc_multimodal2 paper's best model, ported.

Fifteen-second windows; each window carries the paper's descriptor set:

* physical descriptors — CSI ``csi_amp_{mean,std,q90},
  csi_rv_{mean,q90,max}, csi_tv, csi_dop_frac`` (8) and radar
  ``snr_{mean,max}`` + ``{rd,ra,re,xy}_{mean,p90,std90,delta,peak}``
  (22), each computed per five-second block then averaged across the
  window's three blocks;
* hierarchical PCA (10 per modality) — level 1 projects each second
  through the training-fitted bases (CSI 104-d dots, radar 576-d mean
  RD map), level 2 is a two-component PCA on five-second level-1
  trajectories, level 3 a two-component PCA on the window's three
  level-2 pairs; the window gets level-1 and level-2 mean/std plus
  level-3 coordinates.

Late fusion averages the seed-matched CSI and radar RF probabilities
(paper's best view: .975 multilink test minute accuracy); ``view``
config also exposes ``csi``, ``radar`` and ``early`` (GBM early
fusion). Weights ship in ``models/hier_late.joblib`` — trained on the
balanced train split, thresholds tuned on validation — and
``calibrate()`` re-tunes the cutoff on new data (labeled → balanced-
accuracy sweep; unlabeled → the paper's kmeans/median rules).
"""
from __future__ import annotations

import base64
import getpass
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from whispy.contracts import Prediction, SensorWindow
from whispy.models.runner import bound_samples
from whispy.processors.base import Processor, ProcessorMeta

logger = logging.getLogger(__name__)

_BUNDLE = Path(__file__).resolve().parent / "models" / "hier_late.joblib"
_CAL_FILE = Path.home() / ".whispy" / "occ_hier_calibration.json"
SEED = 42

# ---------------------------------------------------------------------------
# Feature constants (E1/common.py vocabulary)
# ---------------------------------------------------------------------------
N_SUB = 52
CSI_SUBCARRIER_MASK = np.array(
    [False] * 6 + [True] * 26 + [False] + [True] * 26 + [False] * 5,
    dtype=bool)
MIN_COV = 0.8
MIN_SEC_FRAMES = 8              # CSI frames needed to form a second
MIN_RADAR_FRAMES_5S = 5         # radar frames needed per 5-s block
RV_W_FRAC = 0.2
BLOCK, WIN = 5, 15
BLOCKS_PER_WIN = WIN // BLOCK
MOD_COLS = {"csi": slice(1, 3), "rad": slice(4, 6)}

CSI_PHYS = ["csi_amp_mean", "csi_amp_std", "csi_amp_q90", "csi_rv_mean",
            "csi_rv_q90", "csi_rv_max", "csi_tv", "csi_dop_frac"]
VIEW_NAMES = ("rd", "ra", "re", "xy")
_DESC = ("mean", "p90", "std90", "delta", "peak")
RADAR_PHYS = (["snr_mean", "snr_max"]
              + [f"{v}_{s}" for v in VIEW_NAMES for s in _DESC])
assert len(CSI_PHYS) == 8 and len(RADAR_PHYS) == 22


def hier_cols(mod):
    return ([f"{mod}_l{lvl}_pc{j}_{s}" for lvl in (1, 2)
             for s in ("mean", "std") for j in (1, 2)]
            + [f"{mod}_l3_pc{j}" for j in (1, 2)])


CSI_ALL = CSI_PHYS + hier_cols("csi")          # 18
RAD_ALL = RADAR_PHYS + hier_cols("rad")        # 32


# ---------------------------------------------------------------------------
# payload decoding
# ---------------------------------------------------------------------------
def csi_amplitude(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    """csi_raw payload -> complex64 (52,) amplitude-capable IQ vector."""
    iq = payload.get("iq")
    if iq is not None and len(iq) >= 128:
        b = np.asarray(iq, dtype=np.float32)[:128]
    else:
        data = payload.get("data")
        if not data:
            return None
        b = np.frombuffer(base64.b64decode(data), np.int8) \
            .astype(np.float32)[:128]
        if b.size != 128:
            return None
    imag = b[0::2][CSI_SUBCARRIER_MASK]
    real = b[1::2][CSI_SUBCARRIER_MASK]
    return real + 1j * imag


def radar_views(payload: Dict[str, Any]):
    """radar payload -> (views (4,24,24) f32, snr) or None."""
    b = payload.get("views_b64")
    if not b:
        return None
    views = np.frombuffer(base64.b64decode(b), np.float16) \
        .reshape(4, 24, 24).astype(np.float32)
    return views, float(payload.get("snr_f", np.nan))


# ---------------------------------------------------------------------------
# per-second products (extract_multilink semantics)
# ---------------------------------------------------------------------------
def rollvar(x: np.ndarray, w: int) -> np.ndarray:
    z = np.zeros((1, x.shape[1]), dtype=np.float64)
    cs = np.concatenate([z, np.cumsum(x, axis=0)], axis=0)
    cs2 = np.concatenate([z, np.cumsum(x * x, axis=0)], axis=0)
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = (hi - lo).astype(np.float64)[:, None]
    mean = (cs[hi] - cs[lo]) / cnt
    msq = (cs2[hi] - cs2[lo]) / cnt
    return np.clip(msq - mean * mean, 0, None)


def _flush_second(rows: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    if len(rows) < MIN_SEC_FRAMES:
        return None
    a = np.abs(np.asarray(rows, np.complex64)).astype(np.float64)
    mean = a.mean(axis=0)
    dot = np.r_[mean, a.std(axis=0)]
    a90 = float(np.quantile(a, 0.9))
    w = max(4, int(round(len(rows) * RV_W_FRAC)))
    rv = np.log1p(rollvar(np.log1p(a), w))
    tr = a.mean(axis=1)
    tv = float(np.abs(np.diff(tr)).sum())
    tr = tr - tr.mean()
    ps = np.abs(np.fft.rfft(tr)) ** 2
    hf = (ps[len(ps) // 3:].sum() / (ps[1:].sum() + 1e-12)
          if len(ps) > 3 else 0.0)
    return {"dot": dot, "a90": a90,
            "rv": np.array([rv.mean(), np.quantile(rv, 0.9), rv.max()],
                           np.float32),
            "tv": tv, "dopf": float(hf)}


class SecondGrid:
    """Per-second products for one capture/window span.

    csi:   ok, dots (104), a90, rv (3), tv, dopf  — index = second
    radar: rsec, vsec (4,24,24), plus frame lists (ts, views, snr) for
           the 5-s physical descriptors.
    """

    def __init__(self):
        self.csi_secs: Dict[int, Dict[str, Any]] = {}
        self.rd_sum: Dict[int, np.ndarray] = {}
        self.rd_n: Dict[int, int] = {}
        self.frames: List[tuple] = []      # (t_rel, views, snr)

    def add_csi(self, sec: int, rec: Dict[str, Any]):
        self.csi_secs[sec] = rec

    def add_radar(self, sec: int, views: np.ndarray, snr: float,
                  t_rel: float):
        if sec not in self.rd_sum:
            self.rd_sum[sec] = np.zeros((4, 24, 24), np.float64)
            self.rd_n[sec] = 0
        self.rd_sum[sec] += views
        self.rd_n[sec] += 1
        self.frames.append((t_rel, views, snr))

    def csi_arrays(self, nsec: int):
        ok = np.zeros(nsec, bool)
        dots = np.full((nsec, 104), np.nan, np.float32)
        a90 = np.full(nsec, np.nan, np.float32)
        rv = np.full((nsec, 3), np.nan, np.float32)
        tv = np.full(nsec, np.nan, np.float32)
        dopf = np.full(nsec, np.nan, np.float32)
        for s, r in self.csi_secs.items():
            if 0 <= s < nsec:
                ok[s] = True
                dots[s] = r["dot"]
                a90[s] = r["a90"]
                rv[s] = r["rv"]
                tv[s] = r["tv"]
                dopf[s] = r["dopf"]
        return {"ok": ok, "dots": dots, "a90": a90, "rv": rv,
                "tv": tv, "dopf": dopf}

    def radar_arrays(self, nsec: int):
        rsec = np.zeros(nsec, bool)
        vsec = np.zeros((nsec, 4, 24, 24), np.float32)
        for s, tot in self.rd_sum.items():
            if 0 <= s < nsec:
                vsec[s] = (tot / self.rd_n[s]).astype(np.float32)
                rsec[s] = True
        return {"rsec": rsec, "vsec": vsec}


def build_seconds(csi_samples, radar_samples, t0: float,
                  nsec: int) -> SecondGrid:
    """Bucket bound samples into per-second products.

    csi_samples/radar_samples are SensorSample-like objects exposing
    ``.timestamp`` and ``.payload``; plain (timestamp, payload) tuples
    work too for replay tooling.
    """
    grid = SecondGrid()
    rows: Dict[int, list] = defaultdict(list)
    for s in csi_samples:
        ts = s[0] if isinstance(s, tuple) else s.timestamp
        pl = s[1] if isinstance(s, tuple) else s.payload
        if not isinstance(pl, dict):
            continue
        a = csi_amplitude(pl)
        if a is None:
            continue
        rel = int(float(ts) - t0)
        if 0 <= rel < nsec:
            rows[rel].append(a)
    for sec, rr in rows.items():
        rec = _flush_second(rr)
        if rec is not None:
            grid.add_csi(sec, rec)
    for s in radar_samples:
        ts = s[0] if isinstance(s, tuple) else s.timestamp
        pl = s[1] if isinstance(s, tuple) else s.payload
        if not isinstance(pl, dict):
            continue
        d = radar_views(pl)
        if d is None:
            continue
        rel_f = float(ts) - t0
        sec = int(rel_f)
        if 0 <= sec < nsec:
            grid.add_radar(sec, d[0], d[1], rel_f)
    return grid


# ---------------------------------------------------------------------------
# level-1 projection + hierarchy (frozen_v3 parity)
# ---------------------------------------------------------------------------
def _project(sc, pca, X):
    X = np.asarray(X, np.float32)
    X = np.where(np.isfinite(X), X, sc.mean_[None, :])
    return pca.transform(sc.transform(X))


def joint_dots(csi, radar, nsec: int, level1) -> np.ndarray:
    """(nsec, 7) rows [ok, c_pc1..3, r_pc1..3]; ok = csi AND radar."""
    sc_c, pca_c = level1["csi"]
    sc_r, pca_r = level1["rad"]
    pc_c = _project(sc_c, pca_c, csi["dots"])[:, :3]
    pc_r = _project(sc_r, pca_r,
                    radar["vsec"][:, 0].reshape(nsec, -1))[:, :3]
    ok = csi["ok"] & radar["rsec"]
    return np.c_[ok.astype(np.float32), pc_c, pc_r]


def _block5(a, cols, start):
    seg = a[start:start + BLOCK]
    if len(seg) < BLOCK:
        return None
    good = (seg[:, 0] > .5) & np.isfinite(seg[:, cols]).all(axis=1)
    if good.sum() < 2:
        return None
    v = seg[:, cols].astype(float)
    return np.column_stack(
        [np.interp(np.arange(BLOCK), np.flatnonzero(good), v[good, j])
         for j in range(v.shape[1])])


def _level2(blocks, h):
    out = np.full((BLOCKS_PER_WIN, 2), np.nan)
    present = [i for i, b in enumerate(blocks) if b is not None]
    if present:
        x = np.stack([blocks[i].reshape(-1) for i in present])
        out[present] = h["p5"].transform(h["sc5"].transform(x))
    return out


def _level3_input(l2):
    present = np.flatnonzero(np.isfinite(l2).all(axis=1))
    if len(present) < 2:
        return None
    return np.column_stack(
        [np.interp(np.arange(BLOCKS_PER_WIN), present, l2[present, j])
         for j in range(2)]).reshape(-1)


def _mean_std(points, min_points):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < min_points:
        return np.full(4, np.nan)
    return np.r_[pts.mean(axis=0), pts.std(axis=0)]


def window_hier(dots: np.ndarray, cols, start: int, h) -> np.ndarray:
    """Ten hierarchical values for one 15-s window (frozen_v3
    window_vector)."""
    blocks = [_block5(dots, cols, start + BLOCK * b)
              for b in range(BLOCKS_PER_WIN)]
    l1 = (np.concatenate([b for b in blocks if b is not None])
          if any(b is not None for b in blocks)
          else np.full((0, 2), np.nan))
    l2 = _level2(blocks, h)
    x3 = _level3_input(l2)
    l3 = (h["p15"].transform(h["sc15"].transform(x3[None]))[0]
          if x3 is not None else np.full(2, np.nan))
    return np.r_[_mean_std(l1, 2), _mean_std(l2, 2), l3]


# ---------------------------------------------------------------------------
# 5-s physical descriptor rows (E1 window math, averaged per 15 s)
# ---------------------------------------------------------------------------
def _csi_5s(csi, s0: int, s1: int) -> np.ndarray:
    x = np.full(len(CSI_PHYS), np.nan, np.float32)
    ok = csi["ok"][s0:s1]
    if len(ok) == 0 or ok.mean() < MIN_COV:
        return x
    d = csi["dots"][s0:s1][ok]
    m2 = d[:, :N_SUB] ** 2 + d[:, N_SUB:] ** 2
    x[0] = d[:, :N_SUB].mean()
    x[1] = np.sqrt(max(m2.mean() - d[:, :N_SUB].mean() ** 2, 0))
    x[2] = csi["a90"][s0:s1][ok].mean()
    x[3] = csi["rv"][s0:s1][ok, 0].mean()
    x[4] = csi["rv"][s0:s1][ok, 1].mean()
    x[5] = csi["rv"][s0:s1][ok, 2].max()
    x[6] = csi["tv"][s0:s1][ok].sum()
    x[7] = csi["dopf"][s0:s1][ok].mean()
    return x


def _radar_5s(frames, t_lo: float, t_hi: float) -> np.ndarray:
    """22 radar descriptors for frames inside [t_lo, t_hi)."""
    x = np.full(len(RADAR_PHYS), np.nan, np.float32)
    sel = [(v, snr) for (t, v, snr) in frames if t_lo <= t < t_hi]
    if len(sel) < MIN_RADAR_FRAMES_5S:
        return x
    vs = np.stack([v for v, _ in sel])
    snrs = np.asarray([s for _, s in sel], np.float32)
    x[0] = float(np.nanmean(snrs))          # snr_mean
    x[1] = float(np.nanmax(snrs))           # snr_max
    off = 2
    for vi in range(4):
        mp = vs[:, vi].astype(np.float32)
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
    return x


def window_features(grid: SecondGrid, nsec: int, level1,
                    hierarchy, start: int) -> Dict[str, np.ndarray]:
    """One 15-s window starting at second ``start`` ->
    {'csi': 18 feats, 'radar': 32 feats, 'early': 50 feats}."""
    csi = grid.csi_arrays(nsec)
    radar = grid.radar_arrays(nsec)
    dots = joint_dots(csi, radar, nsec, level1)

    csi_rows = [_csi_5s(csi, start + b * BLOCK,
                        min(start + (b + 1) * BLOCK, nsec))
                for b in range(BLOCKS_PER_WIN)]
    rad_rows = [_radar_5s(grid.frames, start + b * BLOCK,
                          start + (b + 1) * BLOCK)
                for b in range(BLOCKS_PER_WIN)]

    with np.errstate(invalid="ignore"):
        csi_phys = np.nanmean(np.stack(csi_rows), axis=0).astype(np.float32)
        rad_phys = np.nanmean(np.stack(rad_rows), axis=0).astype(np.float32)

    csi_x = np.r_[csi_phys,
                  window_hier(dots, MOD_COLS["csi"], start,
                              hierarchy["csi"])]
    rad_x = np.r_[rad_phys,
                  window_hier(dots, MOD_COLS["rad"], start,
                              hierarchy["rad"])]
    return {"csi": csi_x, "radar": rad_x,
            "early": np.r_[rad_x, csi_x]}      # RAD_ALL + CSI_ALL order


# ---------------------------------------------------------------------------
# the Processor
# ---------------------------------------------------------------------------
class OccHierModel(Processor):
    """15-s hierarchical-PCA occupancy model (icc_multimodal2 v3).

    ``view`` config picks the reported view: ``late`` (default — RF
    late probability fusion, the paper's best), ``csi``, ``radar`` or
    ``early`` (GBM early fusion).
    """

    VIEWS = ("late", "csi", "radar", "early")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self.csi_input = str(self._config.get("csi_input", "csi"))
        self.radar_input = str(self._config.get("radar_input", "radar"))
        self.view = str(self._config.get("view", "late"))
        if self.view not in self.VIEWS:
            raise ValueError(f"view must be one of {self.VIEWS}")
        self._cal_file = Path(self._config.get(
            "calibration_file", str(_CAL_FILE)))
        self._threshold_override = self._config.get("threshold")
        self.threshold: Optional[float] = None
        self.threshold_source = "bundle"
        self.calibration: Dict[str, Any] = {}
        self._models = None

    # -- contract -------------------------------------------------------------
    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="occ-rf-hier",
            version="0.1.0",
            processor_type="model",
            sensor="wifi_csi+radar",
            task="occupancy",
            inputs=["csi", "radar"],
            outputs=["occupied", "empty"],
            hardware_reqs={"sensors": ["wifi_csi", "radar"]},
            config_schema={
                "type": "object",
                "properties": {
                    "view": {"type": "string",
                             "enum": list(self.VIEWS)},
                    "csi_input": {"type": "string"},
                    "radar_input": {"type": "string"},
                    "threshold": {"type": "number"},
                    "calibration_file": {"type": "string"},
                },
            },
        )

    def configure(self, config: Dict[str, Any]) -> None:
        self._config.update(config)
        if "threshold" in config:
            self._threshold_override = float(config["threshold"])
            self.threshold = self._threshold_override
            self.threshold_source = "override"
        if "view" in config:
            v = str(config["view"])
            if v not in self.VIEWS:
                raise ValueError(f"view must be one of {self.VIEWS}")
            self.view = v

    # -- bundle ---------------------------------------------------------------
    def _load(self):
        if self._models is None:
            import joblib
            bundle_path = Path(self._config.get("bundle", str(_BUNDLE)))
            self._models = joblib.load(bundle_path)
            self.threshold = float(
                self._models["thresholds"][self.view]["mean"])
            self.threshold_source = "bundle"
        return self._models

    def _resolve_threshold(self, csi_sid: str, radar_sid: str) -> float:
        if self._threshold_override is not None:
            self.threshold = float(self._threshold_override)
            self.threshold_source = "override"
            return self.threshold
        saved = self._load_calibration(csi_sid, radar_sid)
        if saved is not None:
            self.threshold = float(saved["threshold"])
            self.threshold_source = f"calibrated:{saved.get('method')}"
            self.calibration = saved
            return self.threshold
        self.threshold = float(
            self._load()["thresholds"][self.view]["mean"])
        self.threshold_source = "bundle"
        return self.threshold

    # -- feature extraction on a SensorWindow ---------------------------------
    def features(self, window: SensorWindow):
        """SensorWindow -> per-view feature vectors + detail."""
        m = self._load()
        csi = bound_samples(window, self.csi_input)
        rad = bound_samples(window, self.radar_input)
        t_all = [s.timestamp for s in (*csi, *rad)]
        if not t_all:
            return None, {"reason": "no sensor data in window",
                          "csi_samples": len(csi),
                          "radar_samples": len(rad)}
        # anchor the 15-s window at the last full second with any data
        s_end = int(np.floor(max(t_all)))
        t0 = float(s_end - WIN + 1)
        grid = build_seconds(
            [(s.timestamp, s.payload) for s in csi],
            [(s.timestamp, s.payload) for s in rad],
            t0, WIN)
        feats = window_features(grid, WIN, m["level1"], m["hierarchy"],
                                start=0)
        detail = {"csi_samples": len(csi), "radar_samples": len(rad),
                  "csi_secs": len(grid.csi_secs),
                  "radar_secs": len(grid.rd_n),
                  "t0": t0}
        return feats, detail

    def _proba_view(self, feats, view) -> float:
        """Mean seed probability for one view; NaN-safe via per-view
        fill/scaler."""
        m = self._load()
        probs = []
        for obj in m["views"][view]:
            x = np.asarray(feats[view], float)
            x[~np.isfinite(x)] = np.nan
            x = np.where(np.isnan(x), obj["fill"], x)
            z = obj["scaler"].transform(x[None])
            probs.append(float(obj["model"].predict_proba(z)[0, 1]))
        return float(np.mean(probs))

    def proba(self, window: SensorWindow):
        feats, detail = self.features(window)
        if feats is None:
            return None, detail
        csi_sid = rad_sid = "?"
        csi = bound_samples(window, self.csi_input)
        rad = bound_samples(window, self.radar_input)
        if csi:
            csi_sid = csi[0].sensor_id
        if rad:
            rad_sid = rad[0].sensor_id
        self._resolve_threshold(csi_sid, rad_sid)
        p_csi = self._proba_view(feats, "csi")
        p_rad = self._proba_view(feats, "radar")
        if self.view == "late":
            p = (p_csi + p_rad) / 2
        elif self.view == "early":
            p = self._proba_view(feats, "early")
        else:
            p = p_csi if self.view == "csi" else p_rad
        detail.update(p=p, p_csi=p_csi, p_rad=p_rad, view=self.view)
        return p, detail

    def predict(self, window: SensorWindow) -> Prediction:
        p, detail = self.proba(window)
        thr = float(self.threshold
                    if self.threshold is not None else 0.62)
        if p is None:
            return Prediction(
                label="unknown", confidence=0.0, task="occupancy",
                scores={"occupied": 0.0, "empty": 0.0},
                metadata={**detail, "threshold": thr,
                          "threshold_source": self.threshold_source})
        label = "occupied" if p >= thr else "empty"
        return Prediction(
            label=label,
            confidence=float(p if label == "occupied" else 1 - p),
            task="occupancy",
            scores={"occupied": float(p), "empty": float(1 - p)},
            metadata={**detail, "threshold": thr,
                      "threshold_source": self.threshold_source})

    # -- threshold calibration -------------------------------------------------
    def calibrate(self, probs, labels=None, method: str = "auto",
                  persist: bool = True,
                  csi_sid: Optional[str] = None,
                  radar_sid: Optional[str] = None) -> Dict[str, Any]:
        """Re-tune the decision threshold on new data.

        ``labels`` given (0/1 per probability) → supervised sweep for
        max balanced accuracy (the frozen_v3 ``fit_threshold`` rule —
        ties break toward 0.5). Without labels the paper's unsupervised
        rules apply: exact 1-D two-means ('kmeans') or the median.
        Persists per user + CSI/radar sensor pair.
        """
        probs = np.asarray(list(probs), np.float64)
        probs = probs[np.isfinite(probs)]
        if probs.size < 2:
            raise ValueError("need >=2 probabilities to calibrate")
        if labels is not None:
            y = np.asarray(list(labels), int)
            if len(y) != probs.size:
                raise ValueError("labels must match probabilities")
            candidates = np.unique(probs)
            scored = []
            for t in candidates:
                pred = (probs >= t).astype(int)
                tp = int(((pred == 1) & (y == 1)).sum())
                tn = int(((pred == 0) & (y == 0)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                tpr = tp / max(tp + fn, 1)
                tnr = tn / max(tn + fp, 1)
                scored.append(((tpr + tnr) / 2,
                               -abs(float(t) - .5), float(t)))
            method = "labeled:balanced_accuracy"
            thr = max(scored)[2]
        elif method == "kmeans" or (
                method == "auto" and np.unique(probs).size > 2):
            x = np.sort(probs)
            n = len(x)
            s, q = np.cumsum(x), np.cumsum(x * x)
            cl, cr = np.arange(1, n), n - np.arange(1, n)
            sl, sr = s[:-1], s[-1] - s[:-1]
            ql, qr = q[:-1], q[-1] - q[:-1]
            cost = ql - sl ** 2 / cl + qr - sr ** 2 / cr
            cost[x[:-1] == x[1:]] = np.inf
            i = int(np.argmin(cost))
            method = "kmeans"
            thr = float((sl[i] / cl[i] + sr[i] / cr[i]) / 2)
        else:
            method = "median"
            thr = float(np.median(probs))
        self.threshold = thr
        self.threshold_source = f"calibrated:{method}"
        rec = {"threshold": thr, "method": method,
               "view": self.view, "n": int(probs.size),
               "p_min": float(probs.min()), "p_max": float(probs.max()),
               "p_mean": float(probs.mean()), "ts": time.time()}
        if persist:
            self._save_calibration(rec, csi_sid, radar_sid)
        self.calibration = rec
        return rec

    # -- persistence -----------------------------------------------------------
    def _cal_key(self, csi_sid=None, radar_sid=None) -> str:
        csi_sid = csi_sid or self._config.get("csi_sensor", "?")
        radar_sid = radar_sid or self._config.get("radar_sensor", "?")
        return f"{getpass.getuser()}|{self.view}|{csi_sid}|{radar_sid}"

    def _load_calibration(self, csi_sid, radar_sid):
        try:
            db = json.loads(self._cal_file.read_text())
        except Exception:
            return None
        return db.get(self._cal_key(csi_sid, radar_sid))

    def _save_calibration(self, rec, csi_sid, radar_sid) -> None:
        try:
            self._cal_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                db = json.loads(self._cal_file.read_text())
            except Exception:
                db = {}
            db[self._cal_key(csi_sid, radar_sid)] = rec
            self._cal_file.write_text(json.dumps(db, indent=2))
        except Exception as exc:                          # noqa: BLE001
            logger.warning("calibration persist failed: %s", exc)


__all__ = ["OccHierModel", "window_features", "build_seconds",
           "SecondGrid", "CSI_ALL", "RAD_ALL", "hier_cols"]
