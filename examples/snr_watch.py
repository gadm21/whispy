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
    parser.add_argument("--device", default="thoth-chen",
                        help="device name or UUID (default: thoth-chen)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="SNR threshold in dB (default: 4)")
    parser.add_argument("--poll", type=float, default=1.0,
                        help="seconds between live-chunk polls (default: 1)")
    args = parser.parse_args()

    username = os.environ.get("THOTHCRAFT_USERNAME")
    password = os.environ.get("THOTHCRAFT_PASSWORD")
    if username and password:
        client = thothcraft.Client.login(username=username, password=password)
    else:
        try:
            client = thothcraft.Client.login()  # token from `thothcraft login`
        except Exception:
            username = username or input("Username: ")
            password = password or getpass.getpass("Password: ")
            client = thothcraft.Client.login(username=username, password=password)
    devices = client.devices()
    device = next(
        (d for d in devices if d.name == args.device or d.uuid == args.device),
        None)
    if device is None:
        names = ", ".join(d.name for d in devices) or "none"
        print(f"device '{args.device}' not found — account devices: {names}")
        return 1

    print(f"watching {device.name} for radar SNR > {args.threshold} dB "
          f"(Ctrl+C to stop)")
    try:
        for chunk in device.stream(poll_s=args.poll):
            features = chunk.get("features") or {}
            snr = features.get("radar_snr_mean_db")
            if snr is None:
                continue
            if snr > args.threshold:
                print(f"[{chunk.get('captured_at', '?')}] "
                      f"SNR {snr:.2f} dB > {args.threshold} — "
                      f"second {chunk.get('second_index', '?')}")
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
