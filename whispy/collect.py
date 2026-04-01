"""Standardized CSI data collection from ESP32 serial port.

Produces 1-minute files (configurable) with:
  - Uniform resampling to guaranteed SR (default 150 Hz)
  - Guard subcarrier removal (64 → 52)
  - Per-file JSON metadata sidecar
  - Optional face detection (camera mode)
  - Optional deploy mode (inference only, no file saved)

Output .npz keys: mag (N,52), phase (N,52), rssi (N,),
    timestamp (N,), face_present (N,), model_pred (N,)
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from whispy.core import (
    CSI_SUBCARRIER_MASK,
    N_SUBCARRIERS_VALID,
    parse_csi_line,
    rolling_variance,
    window_array,
)

DEFAULT_LABELS = ["indoor", "laptopback", "myroom"]
DEFAULT_SR = 150
DEFAULT_DURATION = 60  # seconds


# =========================================================================
# Thread-safe raw CSI buffer
# =========================================================================
class _CSIBuffer:
    """Accumulates parsed CSI packets with face/model columns."""

    def __init__(self):
        self.timestamps: list[int] = []
        self.rssi_vals: list[float] = []
        self.real_rows: list[list[int]] = []
        self.imag_rows: list[list[int]] = []
        self.face_flags: list[int] = []      # 0 or 1
        self.model_preds: list[float] = []   # 0.0–1.0
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._count = 0
        self._face_flag = 0       # latest face state
        self._model_pred = 0.0    # latest model pred

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def set_face_flag(self, val: int) -> None:
        with self._lock:
            self._face_flag = val

    def set_model_pred(self, val: float) -> None:
        with self._lock:
            self._model_pred = val

    def add_line(self, text: str) -> None:
        parsed = parse_csi_line(text)
        if parsed is None:
            return
        with self._lock:
            self.timestamps.append(parsed["timestamp"])
            self.rssi_vals.append(parsed["rssi"])
            self.real_rows.append(parsed["real"].tolist())
            self.imag_rows.append(parsed["imag"].tolist())
            self.face_flags.append(self._face_flag)
            self.model_preds.append(self._model_pred)
            self._count += 1

    def to_arrays(self) -> dict[str, np.ndarray]:
        with self._lock:
            if not self.real_rows:
                e = np.empty((0, 64), dtype=np.float64)
                return {"real": e, "imag": e, "rssi": np.array([]),
                        "timestamp": np.array([], dtype=np.int64),
                        "face_present": np.array([], dtype=np.int32),
                        "model_pred": np.array([], dtype=np.float64)}
            return {
                "real": np.array(self.real_rows, dtype=np.float64),
                "imag": np.array(self.imag_rows, dtype=np.float64),
                "rssi": np.array(self.rssi_vals, dtype=np.float64),
                "timestamp": np.array(self.timestamps, dtype=np.int64),
                "face_present": np.array(self.face_flags, dtype=np.int32),
                "model_pred": np.array(self.model_preds, dtype=np.float64),
            }


# =========================================================================
# Serial reader thread
# =========================================================================
_stop = threading.Event()


def _serial_reader(port: str, baud: int, buf: _CSIBuffer, duration: float) -> None:
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial required — pip install whispy[collect]", file=sys.stderr)
        return

    end = time.time() + duration
    try:
        with serial.Serial(port, baudrate=baud, timeout=0.05) as ser:
            _ = ser.read(ser.in_waiting or 1)
            while not _stop.is_set() and time.time() < end:
                line = ser.readline()
                if not line:
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if text.startswith("CSI_DATA,"):
                    buf.add_line(text)
    except Exception as e:
        print(f"[serial error] {e}", file=sys.stderr)


# =========================================================================
# Camera face-detection thread
# =========================================================================
def _camera_thread(buf: _CSIBuffer, detector: str = "haar") -> None:
    """Run face detection and update buf.face_flag in real time."""
    try:
        import cv2
    except ImportError:
        print("[warn] opencv not installed — pip install whispy[camera]", file=sys.stderr)
        return

    if detector == "haar":
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        def detect(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)
            return len(faces) > 0
    else:  # dnn
        net = cv2.dnn.readNetFromCaffe(
            cv2.data.haarcascades + "../res10_300x300_ssd_iter_140000.caffemodel",
            cv2.data.haarcascades + "../deploy.prototxt",
        )
        def detect(frame):
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                if detections[0, 0, i, 2] > 0.5:
                    return True
            return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[warn] Cannot open camera", file=sys.stderr)
        return

    while not _stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        found = detect(frame)
        buf.set_face_flag(1 if found else 0)
        time.sleep(0.1)  # ~10 fps detection

    cap.release()


# =========================================================================
# Rate monitor thread
# =========================================================================
def _rate_monitor(buf: _CSIBuffer) -> None:
    last, t0 = 0, time.time()
    while not _stop.is_set():
        time.sleep(1.0)
        cur = buf.count
        print(f"  [rate] {cur - last:5d} CSI/s  total={cur:7d}  "
              f"elapsed={time.time() - t0:5.1f}s")
        last = cur


# =========================================================================
# Standardization (resample + guard removal)
# =========================================================================
def _standardize(arrays: dict[str, np.ndarray], sr: int) -> dict | None:
    """Resample to uniform SR, remove guard subcarriers."""
    ts = arrays["timestamp"]
    n = len(ts)
    if n < 10:
        return None

    ts_sec = ts.astype(np.float64) / 1_000_000
    start, end = ts_sec[0], ts_sec[-1]
    dur = end - start
    if dur < 0.1:
        return None

    actual_sr = n / dur
    n_out = max(2, int(np.ceil(dur * sr)))
    target_t = start + np.arange(n_out) / sr
    dt = 1.0 / sr

    bin_edges = target_t - dt / 2
    bin_idx = np.clip(np.searchsorted(bin_edges, ts_sec, side="right") - 1, 0, n_out - 1)
    counts = np.bincount(bin_idx, minlength=n_out).astype(np.float64)
    pop = counts > 0

    real, imag = arrays["real"], arrays["imag"]
    mag_raw = np.sqrt(real**2 + imag**2)
    phase_raw = np.arctan2(imag, real)

    out = {}
    for name, src in [("mag", mag_raw), ("phase", phase_raw)]:
        acc = np.zeros((n_out, 64), dtype=np.float64)
        np.add.at(acc, bin_idx, src)
        acc[pop] /= counts[pop, None]
        empty = ~pop
        if empty.any():
            vi = np.where(pop)[0]
            if len(vi) >= 2:
                for c in range(64):
                    acc[empty, c] = np.interp(target_t[empty], target_t[vi], acc[vi, c])
        out[name] = acc[:, CSI_SUBCARRIER_MASK]

    # 1-D arrays
    for key in ("rssi", "face_present", "model_pred"):
        src = arrays[key].astype(np.float64)
        acc = np.zeros(n_out, dtype=np.float64)
        np.add.at(acc, bin_idx, src)
        acc[pop] /= counts[pop]
        out[key] = acc
    # round face_present back to int
    out["face_present"] = np.round(out["face_present"]).astype(np.int32)
    out["model_pred"] = out["model_pred"].astype(np.float64)

    out["timestamp"] = (target_t * 1_000_000).astype(np.int64)
    out["meta"] = {
        "original_samples": n,
        "resampled_samples": n_out,
        "actual_sr": round(actual_sr, 2),
        "guaranteed_sr": sr,
        "sr_ratio": round(actual_sr / sr, 4),
        "duration_sec": round(dur, 3),
        "subcarriers": N_SUBCARRIERS_VALID,
    }
    return out


# =========================================================================
# Save
# =========================================================================
def _save(result: dict, out_dir: str, label: str, idx: int, wall_start: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = wall_start.replace(":", "").replace("-", "")[:15]
    base = f"{label}_{ts}_{idx:03d}"
    npz_path = os.path.join(out_dir, base + ".npz")
    meta_path = os.path.join(out_dir, base + ".meta.json")

    np.savez_compressed(
        npz_path,
        mag=result["mag"], phase=result["phase"],
        rssi=result["rssi"], timestamp=result["timestamp"],
        face_present=result["face_present"], model_pred=result["model_pred"],
    )

    meta = result["meta"].copy()
    meta.update(label=label, file_index=idx, wall_start=wall_start,
                wall_end=datetime.now(timezone.utc).isoformat(),
                shape=list(result["mag"].shape), version="1.0")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [saved] {npz_path}  shape={result['mag'].shape}")
    return npz_path


# =========================================================================
# Deploy mode (inference only — no file saved)
# =========================================================================
def deploy(
    port: str,
    model_path: str,
    baud: int = 115200,
    sr: int = DEFAULT_SR,
    var_window: int = 20,
    window_len: int = 500,
    # --- rolling cache ---
    cache_gb: float = 1.0,
    # --- MQTT / Home Assistant ---
    mqtt_broker: str | None = None,
    mqtt_port: int = 1883,
    mqtt_node: str = "office",
    mqtt_location: str = "Office",
    mqtt_user: str | None = None,
    mqtt_password: str | None = None,
    mqtt_task: str = "occupancy",
    mqtt_tls: bool = False,
    latitude: float = 0.0,
    longitude: float = 0.0,
    backend_url: str | None = None,
    # --- watchdog ---
    watchdog: bool = False,
    gpio_pin: int | None = None,
    labels: list[str] | None = None,
) -> None:
    """Run live inference on CSI stream with rolling cache, MQTT, and watchdog.

    The rolling cache continuously stores raw CSI packets in a fixed-size
    ring buffer (default 1 GB).  Use ``export_cache()`` to dump data.

    If *mqtt_broker* is set, predictions and diagnostics are published
    to MQTT with Home Assistant auto-discovery.  The device also
    self-registers with the central backend (via MQTT and optionally REST).

    If *watchdog* is True, health checks run every ~15 s and a systemd
    heartbeat is sent (requires ``sd-notify``).
    """
    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"[deploy] Model loaded from {model_path}")

    # ── rolling cache ───────────────────────────────────────
    from whispy.watchdog import RollingCache
    cache = RollingCache(max_bytes=int(cache_gb * 1024**3))
    print(f"[deploy] Rolling cache: {cache}")

    # ── MQTT ────────────────────────────────────────────────
    mqtt = None
    if mqtt_broker:
        from whispy.mqtt import WhispyMQTT
        mqtt = WhispyMQTT(
            broker=mqtt_broker, port=mqtt_port,
            node_id=mqtt_node, location=mqtt_location,
            latitude=latitude, longitude=longitude,
            username=mqtt_user, password=mqtt_password,
            tls=mqtt_tls,
            task=mqtt_task, labels=labels or [],
            backend_url=backend_url,
        )
        if not mqtt.connect():
            print("[deploy] WARNING: MQTT connection failed — continuing without it")
            mqtt = None

    # ── watchdog / health monitor ───────────────────────────
    health = None
    if watchdog:
        from whispy.watchdog import HealthMonitor, WatchdogConfig
        cfg = WatchdogConfig(gpio_reset_pin=gpio_pin)
        health = HealthMonitor(cfg)
        health.report_ready()
        health.report_status("Model loaded, listening for CSI")
        print("[deploy] Watchdog active (systemd heartbeat + health checks)")

    print(f"[deploy] Listening on {port} @ {baud} — Ctrl+C to stop")

    _stop.clear()
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    buf = _CSIBuffer()
    t = threading.Thread(target=_serial_reader, args=(port, baud, buf, 1e9), daemon=True)
    t.start()

    last_count = 0
    last_health_time = time.time()
    last_rate_count = 0
    last_rate_time = time.time()

    try:
        while not _stop.is_set():
            time.sleep(window_len / sr)  # wait one window's worth
            arrays = buf.to_arrays()
            n = len(arrays["timestamp"])
            if n < window_len + var_window:
                continue
            if n == last_count:
                continue

            # ── feed rolling cache with new packets ─────────
            new_start = last_count if last_count < n else 0
            for i in range(new_start, n):
                cache.push(
                    int(arrays["timestamp"][i]),
                    float(arrays["rssi"][i]),
                    arrays["real"][i],
                    arrays["imag"][i],
                )

            last_count = n

            # ── process latest window ───────────────────────
            real, imag = arrays["real"], arrays["imag"]
            mag = np.sqrt(real**2 + imag**2)
            mag = mag[:, CSI_SUBCARRIER_MASK]  # (N, 52)
            rv = rolling_variance(mag, var_window)
            chunk = rv[-window_len:]
            if chunk.shape[0] < window_len:
                continue
            x = chunk.reshape(1, -1)
            pred = model.predict(x)[0]
            prob = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
            conf_val = float(prob.max()) if prob is not None else 0.0
            conf = f"  conf={conf_val:.2f}" if prob is not None else ""
            print(f"  [pred] class={pred}{conf}  (n={n})  cache={cache.size}")
            buf.set_model_pred(conf_val if prob is not None else float(pred))

            # ── MQTT publish ────────────────────────────────
            if mqtt:
                label_str = None
                all_probs = None
                if labels and prob is not None:
                    label_str = labels[int(pred)] if int(pred) < len(labels) else str(pred)
                    all_probs = {labels[i]: float(prob[i]) for i in range(min(len(labels), len(prob)))}
                mqtt.publish_prediction(
                    pred=int(pred), confidence=conf_val,
                    label=label_str, all_probs=all_probs,
                )

            # ── health check + MQTT diagnostics (every ~15s) ──
            now = time.time()
            if now - last_health_time > 15:
                dt = now - last_rate_time
                csi_rate = (n - last_rate_count) / dt if dt > 0 else 0
                last_rate_count = n
                last_rate_time = now
                last_health_time = now

                cpu_temp = None
                if health:
                    results = health.tick(csi_rate=csi_rate)
                    # extract cpu temp from results
                    for r in results:
                        if r.check == "cpu_temp" and r.value > 0:
                            cpu_temp = r.value
                            break

                if mqtt:
                    mqtt.publish_diagnostics(
                        csi_rate=csi_rate,
                        cpu_temp=cpu_temp,
                        esp32_alive=csi_rate > 0,
                        cache_mb=cache.used_bytes / 1024**2,
                    )

    finally:
        _stop.set()
        if mqtt:
            mqtt.disconnect()
        print(f"\n[deploy] Stopped.  Cache: {cache}")


# =========================================================================
# Collect entry point
# =========================================================================
def collect(
    port: str,
    label: str | None = None,
    labels: list[str] | None = None,
    duration: float = DEFAULT_DURATION,
    total_duration: float | None = None,
    baud: int = 115200,
    sr: int = DEFAULT_SR,
    out_dir: str = "./whispy_data",
    camera: bool = False,
    detector: str = "haar",
    pause: float = 2.0,
    no_reset: bool = False,
) -> list[str]:
    """Collect standardized CSI data.

    Parameters
    ----------
    port : serial port of receiver ESP32.
    label : activity label. If None, uses first from labels list.
    labels : list of labels to cycle through. Defaults to DEFAULT_LABELS.
    duration : duration per file in seconds (default 60).
    total_duration : total collection time. Files = total_duration / duration.
    baud : serial baud rate.
    sr : guaranteed sampling rate.
    out_dir : output directory.
    camera : enable face detection.
    detector : 'haar' or 'dnn'.
    pause : seconds between files.
    no_reset : skip ESP32 reset.

    Returns
    -------
    List of saved .npz file paths.
    """
    labels = labels or DEFAULT_LABELS
    label = label or labels[0]
    repeats = max(1, int(total_duration / duration)) if total_duration else 1

    print(f"[collect] port={port}  label={label}  duration={duration}s  "
          f"repeats={repeats}  sr={sr}  camera={camera}")

    if not no_reset:
        try:
            import serial
            _reset_board(port, baud)
        except Exception:
            pass

    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    saved = []
    for i in range(repeats):
        _stop.clear()
        buf = _CSIBuffer()
        wall_start = datetime.now(timezone.utc).isoformat()

        print(f"\n  Recording {i+1}/{repeats}  label={label}  start={wall_start}")

        threads = [
            threading.Thread(target=_serial_reader, args=(port, baud, buf, duration), daemon=True),
            threading.Thread(target=_rate_monitor, args=(buf,), daemon=True),
        ]
        if camera:
            threads.append(threading.Thread(target=_camera_thread, args=(buf, detector), daemon=True))

        for t in threads:
            t.start()

        deadline = time.time() + duration
        try:
            while threads[0].is_alive():
                if time.time() >= deadline:
                    break
                time.sleep(0.2)
        finally:
            _stop.set()
            threads[0].join()
            time.sleep(0.1)

        arrays = buf.to_arrays()
        print(f"  [raw] {len(arrays['timestamp'])} packets, {len(buf.errors)} errors")

        if len(arrays["timestamp"]) == 0:
            print("  [warn] No data — skipping")
            continue

        result = _standardize(arrays, sr)
        if result is None:
            print("  [warn] Standardization failed — skipping")
            continue

        # copy face/model columns
        result.setdefault("face_present", np.zeros(result["mag"].shape[0], dtype=np.int32))
        result.setdefault("model_pred", np.zeros(result["mag"].shape[0], dtype=np.float64))

        path = _save(result, out_dir, label, i + 1, wall_start)
        saved.append(path)

        if i < repeats - 1 and pause > 0:
            print(f"  [pause] {pause}s ...")
            time.sleep(pause)

    print(f"\n[collect] Done — {len(saved)} files saved to {out_dir}")
    return saved


def _reset_board(port: str, baud: int) -> None:
    import serial
    try:
        with serial.Serial(port, baudrate=baud) as s:
            s.dtr = False; s.rts = True; time.sleep(0.05)
            s.dtr = True; s.rts = False; time.sleep(0.05)
        print(f"  [reset] {port}")
    except Exception as e:
        print(f"  [warn] reset: {e}", file=sys.stderr)
