"""Watch a Thoth device's live radar SNR and print when it crosses a threshold.

    python snr_watch.py                          # prompts for credentials
    THOTHCRAFT_USERNAME=you THOTHCRAFT_PASSWORD=secret python snr_watch.py
    python snr_watch.py --device thoth-chen --threshold 4

The device pushes per-second ``features`` (radar SNR, CSI amplitude) to
Brain's live-chunks feed; ``Device.stream()`` yields them as they arrive.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

import thothcraft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="Thoth-116c4bfa",
                        help="device name or UUID (default: Thoth-116c4bfa)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="SNR threshold in dB (default: 4)")
    parser.add_argument("--poll", type=float, default=1.0,
                        help="seconds between live-chunk polls (default: 1)")
    parser.add_argument("--no-start", action="store_true",
                        help="don't send start_collection (device must already be collecting)")
    parser.add_argument("--local", metavar="HOST",
                        help="poll a node on the LAN directly (e.g. thoth-chen.local), no cloud auth")
    args = parser.parse_args()

    if args.local:
        return watch_local(args)

    client = thothcraft.Client.login(username="gadgad", password="password")
    devices = client.devices()
    device = next(
        (d for d in devices if d.name == args.device or d.uuid == args.device),
        None)
    if device is None:
        names = ", ".join(d.name for d in devices) or "none"
        print(f"device '{args.device}' not found — account devices: {names}")
        return 1

    if not args.no_start:
        # live-chunks only carry features while the collector is running
        print(f"starting collection on {device.name}…")
        device.start()

    print(f"watching {device.name} for radar SNR > {args.threshold} dB "
          f"(Ctrl+C to stop)")
    try:
        for chunk in device.stream(poll_s=args.poll):
            features = chunk.get("features") or {}
            snr = features.get("radar_snr_mean_db")
            if snr is None:
                continue
            flag = f" > {args.threshold} — THRESHOLD EXCEEDED" if snr > args.threshold else ""
            print(f"[{chunk.get('captured_at', '?')}] "
                  f"SNR {snr:.2f} dB{flag} "
                  f"(second {chunk.get('second_index', '?')})")
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def watch_local(args) -> int:
    """Poll the node's own dashboard — no Brain account needed.

    On each threshold crossing the script injects a prediction on the node
    (``occupied`` above the threshold, ``empty`` below) which drives the
    device's linked Home Assistant actuator — e.g. the configured light.
    """
    import threading
    import time
    from datetime import datetime
    node = thothcraft.local(args.local, timeout=10)

    # Keep the node in dedicated live-streaming mode: the session file has a
    # 15s TTL, so heartbeat well under that or the collector flaps back to
    # minute collection (which is what makes the stream burst-then-stall).
    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.is_set():
            try:
                node.live_session("start")
                wait = 5.0
            except Exception:
                wait = 1.0  # retry fast — a missed beat costs a radar re-init
            stop_heartbeat.wait(wait)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    print(f"watching {args.local} for radar SNR > {args.threshold} dB "
          f"(Ctrl+C to stop)")
    occupied = None      # edge-trigger: only predict on crossings
    last_updated = None  # dedupe: the live state only changes per radar frame
    try:
        while True:
            try:
                snr = node.radar_snr()
            except Exception as exc:
                print(f"poll failed: {exc}")
                time.sleep(args.poll)
                continue
            if snr["updated_at"] == last_updated:
                time.sleep(args.poll)
                continue  # same frame — nothing new to report
            last_updated = snr["updated_at"]
            value = snr["snr_db"]
            if value is None:
                print(f"no frames yet (stale={snr['stale']})   ", end="\r")
            else:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                now_occupied = value > args.threshold
                flag = f" > {args.threshold}" if now_occupied else ""
                peak = snr.get("peak_power_db")
                floor = snr.get("noise_floor_db")
                age = snr.get("age_seconds")
                detail = (f"peak {peak:.1f} / floor {floor:.1f}, "
                          if peak is not None and floor is not None else "")
                detail += f"age {age:.1f}s" if age is not None else ""
                print(f"[{ts}] SNR {value:6.2f} dB{flag} ({detail})")
                if now_occupied != occupied:
                    occupied = now_occupied
                    label = "occupied" if occupied else "empty"
                    # high confidence either way: for 'empty' the occupancy
                    # probability is 1-confidence, so the light turns OFF
                    try:
                        result = node.predict(label, confidence=0.9)
                        print(f"  -> predicted '{label}': {result.get('ha', result)}")
                    except Exception as exc:
                        print(f"  -> predict('{label}') failed: {exc}")
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        stop_heartbeat.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
