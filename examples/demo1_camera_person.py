"""Demo 1 -- laptop camera -> installable OpenCV person model.

The simplest proof of the model plugin architecture: a local sensor
feeding a locally installed model. The camera adapter knows nothing
about person detection; the model knows nothing about which webcam
produced the frame.

    Windows camera hardware
        -> OpenCvCameraAdapter
        -> SensorHandle
        -> SensorSample(camera, jpeg)
        -> ModelRunner window builder
        -> opencv-haar-person plugin
        -> Prediction(person | no_person)

Run::

    pip install -e whispy/packages/whispy-sensor-opencv-camera
    pip install -e whispy/packages/whispy-model-opencv-person
    python whispy/examples/demo1_camera_person.py

Stages: A) empty frame -> no_person, B) person faces camera -> person,
C) person leaves -> no_person. Press Enter between stages.
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    import whispy

    laptop = whispy.local()

    # -- Step 1: discovery ------------------------------------------------------
    cameras = [s for s in laptop.sensors() if s.type == "camera"]
    if not cameras:
        print("No camera discovered. Is whispy-sensor-opencv-camera "
              "installed and a webcam attached?")
        return 2
    print("Discovered cameras:")
    for c in cameras:
        print(f"  {c.id:<16} {c.metadata.get('name', '')} "
              f"stable={c.metadata.get('stable')}")

    # ``sensor("camera")`` resolves only when unambiguous; otherwise use
    # the stable id printed above.
    camera = laptop.sensor("camera" if len(cameras) == 1 else cameras[0].id)

    # -- Step 3/4: install + bind the model --------------------------------------
    try:
        model = whispy.model("opencv-haar-person")
    except KeyError:
        print("Model not installed: "
              "pip install -e whispy/packages/whispy-model-opencv-person")
        return 2

    runner = model.bind(video=camera, window_seconds=0.5)
    runner.start()

    print("\nModel:", model.metadata().name, model.metadata().version)
    print("Bound input 'video' ->", camera.info.id)
    print("\nStage A: empty frame -- step out of view, then press Enter.")
    input()

    stages = [("A", "no_person"), ("B", "person"), ("C", "no_person")]
    results = []
    for stage, expected in stages:
        if stage != "A":
            prompt = ("Stage B: face the camera clearly, then press Enter."
                      if stage == "B" else
                      "Stage C: leave the frame, then press Enter.")
            input(prompt)
        pred = runner.predict(warmup_s=0.6)
        ok = pred.label == expected
        results.append(ok)
        print(f"  [{stage}] expected={expected:<10} got={pred.label:<10} "
              f"conf={pred.confidence:.2f} count={pred.people_count} "
              f"{'PASS' if ok else 'FAIL'}")

    runner.stop()

    print("\nProvenance")
    print(f"  sensor:   {laptop.info.id} / {camera.info.id}")
    print(f"  model:    {model.metadata().name} on {laptop.info.id}")
    passed = all(results)
    print(f"\n{'PASS' if passed else 'FAIL'} -- "
          f"{sum(results)}/{len(results)} stages matched")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
