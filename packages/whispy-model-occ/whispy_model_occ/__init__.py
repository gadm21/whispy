"""ICC RF-MoE occupancy model plugin (``occ-rf-moe``).

Reproduces the ICC multimodal paper's best model — a 5-second-window
Random-Forest mixture-of-experts over Wi-Fi CSI and mmWave radar — as a
native whispy ``Processor``:

* the trained bundle (``models/icc_moe_5s.joblib``) ships with the
  package, so the model works right after ``pip install``;
* preprocessing is native too: the vendored 100 Hz CSI resampler
  (``resampler.py``), the paper's window features (``features.py``), and
  the radar view maps produced by ``whispy_sensor_mmwhat.dsp``
  (``views_b64``/``snr_f`` payload fields);
* deployment-time self-calibration matches the paper's unsupervised
  variants: 2-means clustering of the observed occupancy probabilities
  (threshold = midpoint of centroids) or the median rule
  (``calibrate()``); the last computed threshold is persisted per
  user + CSI/radar sensor pair and reloaded automatically.

Config keys: ``window_s`` (5), ``context_s`` (8 — seconds of context
before the window for fill/interp parity with the paper),
``csi_input``/``radar_input`` (bound input names), ``threshold``
(override), ``calibration_file``.
"""
from __future__ import annotations

import getpass
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from whispy.contracts import Prediction, SensorWindow
from whispy.models.runner import bound_samples
from whispy.processors.base import Processor, ProcessorMeta

from . import features as F
from . import resampler as R

logger = logging.getLogger(__name__)

_BUNDLE = Path(__file__).resolve().parent / "models" / "icc_moe_5s.joblib"
_CAL_FILE = Path.home() / ".whispy" / "occ_calibration.json"
SEED = 42


def _decode_views(payload: Dict[str, Any]):
    """payload -> (views (4,24,24) f32, snr float) or None."""
    import base64
    b = payload.get("views_b64")
    if b:
        try:
            v = np.frombuffer(base64.b64decode(b), dtype=np.float16)
            views = v.reshape(4, 24, 24).astype(np.float32)
        except Exception:
            return None
        return views, float(payload.get("snr_f", np.nan))
    # fallback: recompute from raw ADC frame when the adapter exposes it
    raw = payload.get("raw_b64")
    if raw:
        try:
            from whispy_sensor_mmwhat import dsp
            arr = dsp.decode_frame(base64.b64decode(raw))
            if arr is None:
                return None
            out = dsp.frame_views(arr)
            return out["views"], float(out["snr"])
        except Exception:
            return None
    return None


class OccMoeModel(Processor):
    """5-s CSI+radar RF mixture-of-experts occupancy classifier."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self.window_s = float(self._config.get("window_s", F.WIN_S))
        self.context_s = float(self._config.get("context_s", 8.0))
        self.csi_input = str(self._config.get("csi_input", "csi"))
        self.radar_input = str(self._config.get("radar_input", "radar"))
        self._cal_file = Path(self._config.get(
            "calibration_file", str(_CAL_FILE)))
        self._models = None            # loaded lazily
        self._threshold_override = self._config.get("threshold")
        self.threshold: Optional[float] = None
        self.threshold_source = "bundle"
        self.calibration: Dict[str, Any] = {}

    # -- Processor contract --------------------------------------------------
    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="occ-rf-moe",
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
                    "window_s": {"type": "number"},
                    "context_s": {"type": "number"},
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
        if "window_s" in config:
            self.window_s = float(config["window_s"])
        if "context_s" in config:
            self.context_s = float(config["context_s"])

    # -- bundle --------------------------------------------------------------
    def _load(self):
        if self._models is None:
            import joblib
            bundle_path = Path(self._config.get("bundle", str(_BUNDLE)))
            self._models = joblib.load(bundle_path)
            self.threshold = float(self._models["threshold"])
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
        self.threshold = float(self._load()["threshold"])
        self.threshold_source = "bundle"
        return self.threshold

    # -- inference -----------------------------------------------------------
    def proba(self, window: SensorWindow):
        """Featurize the window and return (p, detail) or (None, detail)
        when the window is deferred (trailing-second context unseen yet)."""
        m = self._load()
        n_win = int(round(self.window_s))
        n_ctx = int(round(self.context_s))
        csi = bound_samples(window, self.csi_input)
        rad = bound_samples(window, self.radar_input)
        csi_sid = csi[0].sensor_id if csi else self.csi_input
        rad_sid = rad[0].sensor_id if rad else self.radar_input
        self._resolve_threshold(csi_sid, rad_sid)
        detail: Dict[str, Any] = {"csi_samples": len(csi),
                                  "radar_samples": len(rad),
                                  "csi_sensor": csi_sid,
                                  "radar_sensor": rad_sid}

        # ---- CSI: packets -> 100 Hz grid -> per-second blocks --------------
        amps, times = [], []
        for s in csi:
            a = F.csi_amplitude(s)
            if a is not None and np.isfinite(a).all():
                amps.append(a)
                times.append(s.timestamp)
        sec_blk: Dict[int, np.ndarray] = {}
        sec_dot: Dict[int, np.ndarray] = {}
        s_end = None
        if len(amps) >= 2:
            ts = np.asarray(times, np.float64)
            order = np.argsort(ts)
            amp = np.stack(amps)[order]
            ts = ts[order]
            uniform, t, _, _ = R.resample(amp, ts)
            raw_bins = np.floor(ts).astype(int)
            bins = np.floor(t).astype(int)
            s_max = int(raw_bins[-1])
            for sec in range(s_max - n_ctx + 1, s_max + 1):
                observed = ts[raw_bins == sec]
                if (len(observed) < 5
                        or observed[-1] - observed[0] < .5
                        or np.diff(np.r_[sec, observed, sec + 1]).max()
                        > .5):
                    continue
                blk = uniform[bins == sec]
                if len(blk) != F.SEC_HZ:
                    continue
                blk = blk[:, :F.N_SUB].astype(np.float32)
                sec_blk[sec] = blk
                sec_dot[sec] = np.r_[blk.mean(axis=0), blk.std(axis=0)]
            # anchor at the most recent *valid* second — the paper's
            # interpolation borrows the next second, so the window must
            # end on a resolved one (an in-flight trailing second just
            # shifts the window back).
            if sec_blk:
                s_end = max(sec_blk)
        if s_end is None:
            # no valid CSI second — anchor on radar time (radar-only)
            if rad:
                s_end = int(np.floor(max(s.timestamp for s in rad)))
            else:
                detail["deferred"] = True
                detail["reason"] = "no sensor data in window"
                return None, detail

        ctx_secs = list(range(s_end - n_ctx + 1, s_end + 1))
        blocks_ctx = [sec_blk.get(sec) for sec in ctx_secs]
        dots_ctx = np.stack([sec_dot[sec] if sec in sec_dot
                             else np.full(104, np.nan)
                             for sec in ctx_secs])
        ok_ctx = np.array([sec in sec_blk for sec in ctx_secs])
        detail["csi_valid_ctx"] = int(ok_ctx.sum())

        t0, t1 = float(s_end - n_win + 1), float(s_end + 1)

        # ---- radar: per-frame views -> window + per-second RD dots ---------
        views_l, snr_l = [], []
        rd_ctx = np.full((n_ctx, 24 * 24), np.nan, np.float32)
        has_rad = np.zeros(n_ctx, bool)
        rd_acc: Dict[int, List[np.ndarray]] = {}
        for s in rad:
            d = _decode_views(s.payload if isinstance(s.payload, dict)
                              else {})
            if d is None:
                continue
            views, snr = d
            sec = int(np.floor(s.timestamp))
            rd_acc.setdefault(sec, []).append(views[0].reshape(-1))
            if t0 <= s.timestamp < t1:
                views_l.append(views)
                snr_l.append(snr)
        for sec, lst in rd_acc.items():
            j = sec - (s_end - n_ctx + 1)
            if 0 <= j < n_ctx:
                rd_ctx[j] = np.mean(lst, axis=0).astype(np.float32)
                has_rad[j] = True
        detail["radar_frames_win"] = len(views_l)

        # same next-second rule for the radar-PCA path: if rad_pcv would
        # be computed (>=80% second coverage) but the trailing second is
        # uncovered, the paper's minute interp would borrow the next —
        # defer instead.
        if (len(snr_l) >= int(self._config.get("min_radar_frames",
                                               F.MIN_RADAR_FRAMES))
                and has_rad[-n_win:].mean() >= F.MIN_COV
                and not has_rad[-1]):
            detail["deferred"] = True
            detail["reason"] = "trailing radar second uncovered"
            return None, detail

        csi_x = F.csi_window_features(blocks_ctx, dots_ctx, ok_ctx,
                                      np.asarray(m["csi_pca_mean"]),
                                      np.asarray(m["csi_pca_comp"]),
                                      n_win=n_win)
        views_arr = (np.stack(views_l).astype(np.float32)
                     if views_l else np.zeros((0, 4, 24, 24), np.float32))
        rad_x = F.radar_window_features(
            views_arr, np.asarray(snr_l, np.float32), rd_ctx, has_rad,
            pca_rad=m.get("pca_rad"), n_win=n_win,
            min_frames=int(self._config.get("min_radar_frames",
                                            F.MIN_RADAR_FRAMES)))
        detail.update(csi_ok=bool(csi_x[-1]), rad_ok=bool(rad_x[-1]),
                      deferred=False)

        pcs = [p.predict_proba(csi_x[None])[0, 1]
               for p in m["csi_models"]]
        prs = [p.predict_proba(rad_x[None])[0, 1]
               for p in m["radar_models"]]
        p_csi = float(np.mean(pcs))
        p_rad = float(np.mean(prs))
        w = float(m["w_csi"])
        p = w * p_csi + (1 - w) * p_rad
        detail.update(p=p, p_csi=p_csi, p_rad=p_rad, w_csi=w)
        return p, detail

    def predict(self, window: SensorWindow) -> Prediction:
        p, detail = self.proba(window)
        thr = float(self.threshold
                    if self.threshold is not None else 0.5)
        if p is None:
            return Prediction(
                label="unknown", confidence=0.0,
                task="occupancy",
                scores={"occupied": 0.0, "empty": 0.0},
                metadata={**detail, "threshold": thr,
                          "threshold_source": self.threshold_source,
                          "reason": "window deferred (trailing second "
                                    "unresolved)"})
        label = "occupied" if p >= thr else "empty"
        return Prediction(
            label=label,
            confidence=float(p if label == "occupied" else 1 - p),
            task="occupancy",
            scores={"occupied": float(p), "empty": float(1 - p)},
            metadata={**detail, "threshold": thr,
                      "threshold_source": self.threshold_source})

    # -- paper self-calibration ----------------------------------------------
    def calibrate(self, probs, method: str = "kmeans",
                  persist: bool = True,
                  csi_sid: Optional[str] = None,
                  radar_sid: Optional[str] = None) -> Dict[str, Any]:
        """Unsupervised deployment calibration (paper's ``acc_km`` /
        ``acc_cal``): 2-means clustering of observed occupancy
        probabilities -> threshold = midpoint of the two centroids
        ('mean of means'); ``method='median'`` selects the median
        probability instead. Persists per user+sensor pair."""
        probs = np.asarray(list(probs), np.float64)
        probs = probs[np.isfinite(probs)]
        if probs.size < 2:
            raise ValueError("need >=2 probabilities to calibrate")
        if method == "kmeans" and np.unique(probs).size > 2:
            from sklearn.cluster import KMeans
            ctr = KMeans(n_clusters=2, n_init=10, random_state=SEED) \
                .fit(probs.reshape(-1, 1)).cluster_centers_.ravel()
            thr = float(np.sort(ctr).mean())
        else:
            method = "median"
            thr = float(np.median(probs))
        self.threshold = thr
        self.threshold_source = f"calibrated:{method}"
        rec = {
            "threshold": thr, "method": method, "n": int(probs.size),
            "p_min": float(probs.min()), "p_max": float(probs.max()),
            "p_mean": float(probs.mean()), "ts": time.time(),
        }
        if persist:
            self._save_calibration(rec, csi_sid, radar_sid)
        self.calibration = rec
        return rec

    # -- persistence ----------------------------------------------------------
    def _cal_key(self, csi_sid=None, radar_sid=None) -> str:
        csi_sid = csi_sid or self._config.get("csi_sensor", "?")
        radar_sid = radar_sid or self._config.get("radar_sensor", "?")
        return f"{getpass.getuser()}|{csi_sid}|{radar_sid}"

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
        except Exception as exc:                       # noqa: BLE001
            logger.warning("calibration persist failed: %s", exc)


__all__ = ["OccMoeModel"]
