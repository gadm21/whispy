"""Watch a local sensor stream and fire a rule processor on each crossing.

    python examples/snr_watch.py --sensor system-0 --threshold 50

Demonstrates the Whispy Sensor->Model->Actuator pattern: a bounded
SampleStream feeds a WindowSynchronizer, a rule processor scores each
rolling window, and a prediction is printed whenever the label flips.
"""
from __future__ import annotations

import argparse
import time

import whispy
from whispy.streams import SampleStream
from whispy.synchronization import WindowSynchronizer
from whispy.processors import create_processor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sensor", default=None,
                        help="sensor id to watch (default: first detected)")
    parser.add_argument("--window", type=float, default=2.0,
                        help="rolling window seconds (default: 2)")
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="value threshold for the rule (default: 50)")
    args = parser.parse_args()

    node = whispy.local()
    sensors = node.sensors()
    if not sensors:
        print("no sensors detected on this machine")
        return 1
    sensor = next((s for s in sensors if s.id == args.sensor), sensors[0])
    print(f"watching {sensor.id} ({sensor.type}), window={args.window}s, "
          f"threshold={args.threshold} (Ctrl+C to stop)")

    handle = node.sensor(sensor.id)
    stream = SampleStream(handle.stream(), name=sensor.id).start()
    sync = WindowSynchronizer({sensor.id: stream}, expected=[sensor.id])

    # A rule processor over the window's mean feature. WindowFeatures exposes
    # <sensor>_mean|std|max|min|energy; the rule fires when the mean crosses
    # the threshold param.
    proc = create_processor({
        "processor": "rule",
        "name": "threshold-watch",
        "rules": [{"when": f"{sensor.type}_mean > threshold",
                   "label": "above", "confidence": 0.9}],
        "else": "below",
        "params": {"threshold": args.threshold},
        "sensor": sensor.type, "task": "threshold",
    })

    label = None
    try:
        while True:
            window = sync.rolling(args.window)
            pred = proc.predict(window)
            if pred.label != label:
                label = pred.label
                print(f"[{time.strftime('%H:%M:%S')}] -> {pred.label} "
                      f"(confidence={pred.confidence:.2f})")
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        stream.close()
        node.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
