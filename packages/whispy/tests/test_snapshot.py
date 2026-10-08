"""whispy.snapshot — one-call local physical + textual descriptors."""
import json

import whispy
from whispy.devices.local import LocalDevice
from whispy.sensors import FixtureDriver


class _Describer:
    def describe(self, window, physical, now):
        return {sid: {"text": f"{sid}: ok", "cues": {"n": d.get("n")}}
                for sid, d in physical.items()}

    def summary(self, described):
        return " | ".join(v["text"] for v in described.values())


def _device():
    return LocalDevice(device_id="t", drivers={"fixture": FixtureDriver()})


def test_snapshot_returns_descriptors_text_and_scene():
    snap = whispy.snapshot(0.6, device=_device(), describer=_Describer())
    assert snap["sensors"], "fixture sensor should produce samples"
    sid, desc = next(iter(snap["sensors"].items()))
    assert desc["n"] > 0
    assert desc["text"] == f"{sid}: ok"
    assert snap["scene"].split(" | ") == [f"{s}: ok" for s in snap["sensors"]]
    json.dumps(snap)


def test_snapshot_without_text_and_sensor_filter():
    snap = whispy.snapshot(0.3, device=_device(), text=False,
                           sensors=["does-not-exist"])
    assert snap["sensors"] == {} and snap["scene"] is None
