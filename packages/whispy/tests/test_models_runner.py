"""Model plugins, ModelRunner bindings, and capture sessions."""
import itertools
import time

import pytest

import whispy
from whispy.contracts import (
    ActuatorCommand, ActuatorDescriptor, ModelInput, Prediction,
    SensorSample, SensorWindow,
)
from whispy.devices.local import LocalDevice
from whispy.models import ModelRunner, bound_samples, bound_sensor_id
from whispy.processors.base import Processor, ProcessorMeta
from whispy.sensors import FixtureDriver, SensorDriverAdapter


class _EchoProcessor(Processor):
    """Test model: labels the window with its bound sensor id."""

    def __init__(self, config=None):
        self._config = dict(config or {})

    def metadata(self):
        return ProcessorMeta(name="echo", processor_type="echo",
                             inputs=("camera",), task="echo")

    def predict(self, window):
        sid = bound_sensor_id(window, "audio") or "none"
        n = len(bound_samples(window, "audio"))
        return Prediction(label=f"heard-{sid}", confidence=1.0,
                          task="echo",
                          metadata={"samples": n, "sensor": sid})


def _fixture_device(sensor_type="camera", payloads=None):
    drv = FixtureDriver()
    dev = LocalDevice(device_id="dev-1", drivers={"fixture": drv})
    dev.open({"fixture": {"sensor_type": sensor_type,
                          "payloads": payloads or [[0.0]],
                          "sample_rate": 100}})
    return dev


def test_model_registry_lists_builtins():
    names = {m["name"] for m in whispy.models()}
    assert {"rule", "torchscript", "fusion"} <= names


def test_model_factory_builtin():
    handle = whispy.model("rule", config={
        "rules": [{"when": "1 > 0", "label": "yes"}], "else": "no"})
    assert handle.metadata().name
    w = SensorWindow(start_timestamp=0, end_timestamp=1, samples={})
    assert handle.predict(w).label == "yes"


def test_model_factory_unknown():
    with pytest.raises(KeyError):
        whispy.model("does-not-exist")


def test_model_input_capability_form():
    inp = ModelInput.from_dict({
        "name": "audio", "modality": "camera",
        "capabilities": ["pcm_audio"],
        "constraints": {"sample_rate": 16000}})
    assert inp.name == "audio" and inp.modality == "camera"
    from whispy.contracts import Sensor
    good = Sensor(id="mic-1", type="camera",
                  capabilities=["pcm_audio"], sample_rate=16000)
    bad = Sensor(id="cam-1", type="camera")
    assert inp.matches(good)
    assert not inp.matches(bad)


def test_model_input_legacy_form():
    inp = ModelInput.from_dict({"sensor": "radar", "window_seconds": 2.0})
    assert inp.modality == "radar" and inp.name == "radar"
    assert inp.window_seconds == 2.0


def test_runner_binds_named_input():
    dev = _fixture_device()
    try:
        mic = dev.sensor("camera")
        handle = whispy.model("rule", config={
            "rules": [{"when": "microphone_mean >= 0", "label": "ok"}],
            "else": "none"})
        runner = handle.bind(audio=mic, window_seconds=0.2)
        pred = runner.predict(warmup_s=0.3)
        assert pred.label in ("ok", "none")
        assert pred.source_window["bindings"]["audio"] == mic.info.id
        runner.stop()
    finally:
        dev.close()


def test_runner_positional_single_input():
    dev = _fixture_device()
    try:
        mic = dev.sensor("camera")
        proc = _EchoProcessor()
        from whispy.models.registry import ModelHandle
        handle = ModelHandle("echo", proc,
                             config={"inputs": [{"name": "audio",
                                                 "modality": "camera"}]})
        runner = handle.bind(mic)
        pred = runner.predict(warmup_s=0.3)
        assert pred.label == f"heard-{mic.info.id}"
        assert pred.metadata["samples"] > 0
        runner.stop()
    finally:
        dev.close()


def test_runner_window_has_binding_map():
    dev = _fixture_device()
    try:
        mic = dev.sensor("camera")
        proc = _EchoProcessor()
        runner = ModelRunner(proc, {"audio": mic}, window_seconds=0.2)
        runner.start()
        time.sleep(0.3)
        window = runner._window()
        assert bound_sensor_id(window, "audio") == mic.info.id
        assert bound_samples(window, "audio")
        runner.stop()
    finally:
        dev.close()


def test_capture_window_session():
    dev = _fixture_device(payloads=[[1.0], [2.0]])
    try:
        mic = dev.sensor("camera")
        session = whispy.capture_window(mic, pre_roll=0.05)
        session.start()
        window = session.finish(post_roll=0.1)
        assert mic.info.id in window.samples
        assert len(window.samples[mic.info.id]) > 0
        assert window.modalities[mic.info.id].state == "ok"
        assert window.timing["pre_roll"] == 0.05
    finally:
        dev.close()


def test_capture_window_requires_start():
    dev = _fixture_device()
    try:
        mic = dev.sensor("camera")
        session = whispy.capture_window(mic)
        with pytest.raises(RuntimeError):
            session.finish()
    finally:
        dev.close()

