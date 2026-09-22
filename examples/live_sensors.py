"""Measure live sensor rates on Thoth nodes via the SDK.

    python examples/live_sensors.py 10.0.0.88 10.0.0.22

Opens a live session on each node (the collector switches to dedicated
streaming mode), then samples every sensor endpoint for a few seconds and
reports the achieved rate vs. what the node reports as configured.
"""
from __future__ import annotations

import sys
import threading
import time

import thothcraft
from thothcraft.errors import APIError

SAMPLE_SECONDS = 6.0


def _heartbeat(node: thothcraft.local.LocalDevice, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            node.live_session()
        except Exception:
            pass
        stop.wait(5.0)


def measure_radar(node) -> tuple[float, dict]:
    """Count distinct radar frames via the per-frame SNR sidecar."""
    seen: set = set()
    t0 = time.monotonic()
    last = {}
    while time.monotonic() - t0 < SAMPLE_SECONDS:
        try:
            last = node.radar_snr()
            if last.get("updated_at"):
                seen.add(last["updated_at"])
        except APIError:
            pass
        time.sleep(0.02)
    return len(seen) / SAMPLE_SECONDS, last


def measure_csi(node) -> tuple[float, int]:
    cursor, count = "", 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < SAMPLE_SECONDS:
        try:
            body = node.csi_tail(cursor)
        except APIError:
            time.sleep(0.2)
            continue
        cursor = body.get("cursor") or cursor
        count += len(body.get("samples") or [])
        time.sleep(0.05)
    return count / SAMPLE_SECONDS, count


def measure_sensehat(node) -> tuple[float, dict]:
    seen: set = set()
    t0 = time.monotonic()
    last = {}
    while time.monotonic() - t0 < SAMPLE_SECONDS:
        try:
            body = node.sensehat()
            latest = body.get("latest") or {}
            if latest.get("monotonic_ns"):
                seen.add(latest["monotonic_ns"])
                last = latest
        except APIError:
            pass
        time.sleep(0.15)
    return len(seen) / SAMPLE_SECONDS, last


def measure_camera(node) -> tuple[float, int]:
    count, size = 0, 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < SAMPLE_SECONDS:
        try:
            frame = node.camera_frame()
            count += 1
            size = len(frame)
        except APIError:
            time.sleep(0.3)
    return count / SAMPLE_SECONDS, size


def probe_node(host: str) -> None:
    node = thothcraft.local(host)
    print(f"\n=== {host} ===")
    try:
        sensors = {s["key"]: s for s in node.sensors().get("sensors", [])}
    except Exception as exc:
        print(f"  unreachable: {exc}")
        return
    for key, s in sensors.items():
        mark = "on " if s.get("online") or s.get("available") else "off"
        print(f"  [{mark}] {key:<16} {s.get('source') or ''}")

    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(node, stop), daemon=True)
    hb.start()
    try:
        time.sleep(2.0)  # let the collector switch to live mode

        if sensors.get("dreamhat_radar", {}).get("online") or \
                sensors.get("dreamhat_radar", {}).get("ever_seen"):
            hz, last = measure_radar(node)
            print(f"  radar:    {hz:6.1f} Hz  (snr={last.get('snr_db')} dB, "
                  f"detected={last.get('detected')})")
        else:
            print("  radar:    not present")

        if sensors.get("esp32_csi", {}).get("online") or \
                sensors.get("esp32_csi", {}).get("devices"):
            hz, total = measure_csi(node)
            print(f"  csi:      {hz:6.1f} Hz  ({total} samples)")
        else:
            print("  csi:      not present")

        if sensors.get("sense_hat", {}).get("online") or \
                sensors.get("sense_hat", {}).get("available"):
            hz, last = measure_sensehat(node)
            print(f"  sensehat: {hz:6.1f} Hz  (temp={last.get('temperature_c')}, "
                  f"accel={last.get('acceleration')})")
        else:
            print("  sensehat: not present")

        if sensors.get("usb_camera", {}).get("online") or \
                sensors.get("usb_camera", {}).get("available"):
            hz, size = measure_camera(node)
            print(f"  camera:   {hz:6.1f} fps ({size} B/frame)")
        else:
            print("  camera:   not present")
    finally:
        stop.set()
        hb.join(timeout=2)


if __name__ == "__main__":
    hosts = sys.argv[1:] or ["thoth.local"]
    for host in hosts:
        try:
            probe_node(host)
        except Exception as exc:
            print(f"{host}: failed — {exc}")
