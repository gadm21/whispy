"""thothcraftd — the Thoth device daemon.

Handles sensor discovery, collection, local storage, prediction, cloud
synchronization, commands and heartbeats. The `thothcraft` CLI controls
it; on Linux it runs under systemd (`systemctl status thothcraftd`).

A Thoth device is any authenticated node implementing the device
protocol — this daemon is the reference implementation for computers
(Pi, laptop, Jetson) rather than dedicated hardware.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

CONFIG_DIR = Path(os.getenv("THOTHCRAFT_CONFIG_DIR", Path.home() / ".thothcraft"))
DEVICE_FILE = CONFIG_DIR / "device.json"
HEARTBEAT_SECONDS = int(os.getenv("THOTHCRAFT_HEARTBEAT_SECONDS", "30"))


def _device_uuid() -> str:
    """Stable per-machine device UUID, persisted locally."""
    if DEVICE_FILE.exists():
        try:
            return json.loads(DEVICE_FILE.read_text())["device_uuid"]
        except (ValueError, KeyError):
            pass
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    new_uuid = str(uuid.uuid4())
    DEVICE_FILE.write_text(json.dumps({"device_uuid": new_uuid}))
    return new_uuid


def _load_device_token() -> str:
    try:
        return json.loads(DEVICE_FILE.read_text())["device_token"]
    except (OSError, ValueError, KeyError):
        return ""


def run(config_path: str = None) -> int:
    """Main daemon loop: register → heartbeat → sync → commands."""
    from thothcraft.client import Client, DEFAULT_BASE_URL
    from . import probe

    device_uuid = _device_uuid()
    token = _load_device_token()
    if not token:
        print("[thothcraftd] No device credential — run `thothcraft pair` first",
              file=sys.stderr)
        return 2

    base_url = Client.load_base_url() or DEFAULT_BASE_URL
    client = Client(base_url, token=token)
    capabilities = {k: v[0] for k, v in probe.scan().items()}

    print(f"[thothcraftd] device={device_uuid} brain={base_url}")
    print(f"[thothcraftd] capabilities: {capabilities}")

    stop = False

    def _term(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    while not stop:
        try:
            client._http.post_json("/api/device/heartbeat", body={
                "device_id": device_uuid,
                "capabilities": capabilities,
                "daemon": "thothcraftd",
            })
        except Exception as e:
            print(f"[thothcraftd] heartbeat failed: {e}", file=sys.stderr)
        # TODO: collection, local storage, prediction, sync, command poll
        time.sleep(HEARTBEAT_SECONDS)

    print("[thothcraftd] stopped")
    return 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="thothcraftd")
    parser.add_argument("--config", default=None, help="device.json path")
    args = parser.parse_args()
    raise SystemExit(run(args.config))
