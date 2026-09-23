"""Measure live sensor sample rates on this machine via Whispy.

    python examples/live_sensors.py

Enumerates the local node's sensors, opens a bounded SampleStream on each,
and reports the achieved sample rate over a few seconds.
"""
from __future__ import annotations

import time

import whispy
from whispy.streams import SampleStream

SAMPLE_SECONDS = 5.0


def measure(stream: SampleStream) -> float:
    count = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < SAMPLE_SECONDS:
        count += len(stream.drain())
        time.sleep(0.1)
    return count / SAMPLE_SECONDS


def main() -> int:
    node = whispy.local()
    sensors = node.sensors()
    if not sensors:
        print("no sensors detected on this machine")
        return 1
    print(f"{node.info.id}: {len(sensors)} sensor(s)")
    for sensor in sensors:
        handle = node.sensor(sensor.id)
        stream = SampleStream(handle.stream(), name=sensor.id).start()
        try:
            hz = measure(stream)
            print(f"  {sensor.id:<20} {sensor.type:<10} {hz:6.1f} Hz "
                  f"(dropped={stream.dropped})")
        except Exception as exc:
            print(f"  {sensor.id:<20} stream failed: {exc}")
        finally:
            stream.close()
    node.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
