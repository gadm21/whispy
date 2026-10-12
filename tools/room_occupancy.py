"""Collect ESP32 Wi-Fi CSI with Whispy and estimate room occupancy.

Run once with ``--calibrate-empty`` while the room is empty, then run
normally to append CSI features and baseline-relative occupancy estimates.
This is a transparent heuristic; it needs a working CSI firmware stream
and should be validated against the room before being used for automation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def numeric_iq(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        values = payload.get("iq")
        if values is None and isinstance(payload.get("data"), dict):
            values = payload["data"].get("iq")
    else:
        values = payload
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return []
    vals = []
    for i in range(0, len(values) - 1, 2):
        try:
            a, b = float(values[i]), float(values[i + 1])
            if math.isfinite(a) and math.isfinite(b):
                vals.append(math.hypot(a, b))
        except (TypeError, ValueError):
            continue
    return vals


def feature(sample: Any) -> tuple[float, float, int] | None:
    amps = numeric_iq(getattr(sample, "payload", None))
    if len(amps) < 2:
        return None
    return statistics.fmean(amps), statistics.pstdev(amps), len(amps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM10", help="ESP32 serial port (default: COM10)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--calibrate-empty", action="store_true", help="save an empty-room baseline; leave room empty while collecting")
    parser.add_argument("--baseline", type=Path, default=Path("room_csi_baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("room_csi_observations.csv"))
    parser.add_argument("--sensor", help="Whispy CSI sensor id (auto-select when omitted)")
    args = parser.parse_args()

    os.environ["WHISPY_CSI_SOURCES"] = f"serial:{args.port}@{args.baud}"
    repo = Path(__file__).resolve().parents[1] / "packages" / "whispy"
    sys.path.insert(0, str(repo))
    try:
        import whispy
        node = whispy.local()
        sensors = node.sensors()
        candidates = [s for s in sensors if "csi" in f"{s.id} {s.type}".lower()]
        if args.sensor:
            sensor = node.sensor(args.sensor)
        elif candidates:
            sensor = node.sensor(candidates[0].id)
        else:
            print("No Wi-Fi CSI sensor was discovered. Check ESP32 firmware, port, and baud rate.", file=sys.stderr)
            print("Discovered sensors:", ", ".join(f"{s.id} ({s.type})" for s in sensors) or "none", file=sys.stderr)
            return 2
        print(f"Whispy CSI sensor: {sensor.info.id} ({sensor.info.type})")
        print(f"Collecting for {args.seconds:g}s from {args.port} at {args.baud} baud")
        print("Do not interpret missing samples as an empty room.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + args.seconds
        features: list[tuple[float, float, int]] = []
        baseline = None
        if not args.calibrate_empty and args.baseline.exists():
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        with args.output.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["timestamp_utc", "sensor_id", "amplitude_mean", "amplitude_std", "subcarriers", "baseline_delta", "occupancy_estimate", "method"])
            stream = sensor.stream()
            try:
                for sample in stream:
                    if time.monotonic() >= deadline:
                        break
                    values = feature(sample)
                    if values is None:
                        continue
                    mean, std, count = values
                    features.append(values)
                    delta = abs(std - float(baseline["std_mean"])) if baseline else None
                    if args.calibrate_empty:
                        label, method = "calibration_empty", "empty-room baseline collection"
                    elif baseline and int(baseline.get("samples", 0)) >= 30:
                        threshold = max(float(baseline["std_mad"]) * 6.0, float(baseline["std_mean"]) * 0.12, 1e-6)
                        label = "occupied_candidate" if delta is not None and delta > threshold else "empty_candidate"
                        method = "CSI amplitude variation vs empty-room baseline (heuristic)"
                    else:
                        label, method = "unknown", "no valid empty-room baseline"
                    writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sample.sensor_id, f"{mean:.5f}", f"{std:.5f}", count, "" if delta is None else f"{delta:.5f}", label, method])
                    f.flush()
                    print(f"{label}: CSI std={std:.3f}" + (f", baseline Δ={delta:.3f}" if delta is not None else ""))
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()
        if args.calibrate_empty:
            if len(features) < 30:
                print(f"Only {len(features)} valid CSI frames; need at least 30 for a baseline. No baseline saved.", file=sys.stderr)
                return 3
            stds = [x[1] for x in features]
            med = statistics.median(stds)
            mad = statistics.median(abs(x - med) for x in stds)
            args.baseline.write_text(json.dumps({"method": "empty-room CSI amplitude variability", "sensor_id": sensor.info.id, "port": args.port, "created_at": time.time(), "samples": len(stds), "std_mean": statistics.fmean(stds), "std_mad": mad}, indent=2), encoding="utf-8")
            print(f"Saved empty-room baseline with {len(stds)} CSI frames to {args.baseline}")
        elif not features:
            print("No valid CSI frames arrived. Occupancy is unknown; check port access and CSI firmware.", file=sys.stderr)
            return 4
        print(f"CSI samples appended to {args.output}")
        return 0
    except Exception as exc:
        print(f"Whispy could not stream CSI: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Release the serial port, then rerun; the script does not infer occupancy without live frames.", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
