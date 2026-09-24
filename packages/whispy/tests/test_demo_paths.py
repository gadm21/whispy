"""Demo-path tests — the two acceptance demos with fixture/fake hardware.

Demo 1: camera adapter → opencv-haar-person (real cv2, synthetic frame).
Demo 2: speaker actuator → capture_window → whisper-stt (fake backend).

These validate the *architecture* — plugin discovery, binding, capture
orchestration, model independence — without requiring a webcam, a Pi,
or a Whisper download.
"""
import base64
import itertools
import sys
import time
from pathlib import Path

import pytest

import whispy
from whispy.contracts import (
    ActionResult, ActionStatus, ActuatorCommand, ActuatorDescriptor,
    Prediction, SensorDescriptor, SensorSample,
)
from whispy.devices.local import LocalDevice
from whispy.sensors import FixtureDriver, SensorDriverAdapter
from whispy.sensors.base import SensorAdapter, SensorMeta
from whispy.actuators.base import (
    ActuatorAdapter, ActuatorHandle, ActuatorMeta,
)

PACKAGES = Path(__file__).resolve().parents[2]


def _import_pkg(pkg_dir: str, module: str):
    """Import a sibling plugin package without installing it."""
    path = str(PACKAGES / pkg_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    __import__(module)


# -- Demo 1: camera → person model ------------------------------------------------

def _jpeg_frame(draw_face: bool = False):
    """A synthetic 320×240 frame; optionally with a face-like pattern."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    if draw_face:
        cv2.circle(frame, (160, 120), 60, (200, 200, 200), -1)
        cv2.circle(frame, (135, 100), 10, (40, 40, 40), -1)
        cv2.circle(frame, (185, 100), 10, (40, 40, 40), -1)
        cv2.ellipse(frame, (160, 145), (25, 12), 0, 0, 180, (60, 60, 60), 3)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class _CameraAdapter(SensorAdapter):
    """Fixture camera emitting synthetic JPEG frames."""

    def __init__(self, face: bool = False):
        self._face = face

    def metadata(self):
        return SensorMeta(name="test-camera", modalities=("camera",))

    def discover(self):
        return [SensorDescriptor(
            id="camera-test", modality="camera", adapter="test-camera",
            name="Test Camera", hardware_id="test-cam-0", stable=True)]

    def connect(self, descriptor, config=None):
        face = self._face

        class _H:
            @property
            def info(self):
                return descriptor.to_sensor()

            def stream(self, max_samples=None):
                for i in itertools.count():
                    yield SensorSample.now(
                        "dev", descriptor.id, "camera",
                        {"encoding": "jpeg", "width": 320, "height": 240,
                         "data": _jpeg_frame(face)},
                        sequence=i, payload_type="jpeg")
                    time.sleep(0.01)
                    if max_samples and i + 1 >= max_samples:
                        return

            def close(self):
                pass

        return _H()


def _haar_model():
    _import_pkg("whispy-model-opencv-person", "whispy_model_opencv_person")
    from whispy_model_opencv_person import HaarPersonModel
    return HaarPersonModel()


def _has_cascade() -> bool:
    try:
        import cv2
        return hasattr(cv2, "CascadeClassifier")
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cascade(),
                    reason="cv2.CascadeClassifier unavailable")
def test_demo1_pipeline_no_person():
    """Camera adapter → runner → HaarPersonModel on a blank frame."""
    dev = LocalDevice(device_id="laptop-01", drivers={}, adapters={
        "test-camera": _CameraAdapter(face=False)})
    try:
        camera = dev.sensor("camera")
        assert camera.info.id == "camera-test"
        proc = _haar_model()
        from whispy.models.registry import ModelHandle
        handle = ModelHandle("opencv-haar-person", proc, config={
            "inputs": [{"name": "video", "modality": "camera"}]})
        runner = handle.bind(video=camera, window_seconds=0.2)
        pred = runner.predict(warmup_s=0.3)
        assert pred.label == "no_person"
        assert pred.task == "person_presence"
        assert pred.metadata["frame_sensor"] == "camera-test"
        runner.stop()
    finally:
        dev.close()


def test_demo1_pipeline_binding_without_cascade():
    """The binding/window path works regardless of cascade availability —
    the model reports 'error' explicitly rather than crashing."""
    dev = LocalDevice(device_id="laptop-01", drivers={}, adapters={
        "test-camera": _CameraAdapter(face=False)})
    try:
        camera = dev.sensor("camera")
        proc = _haar_model()
        from whispy.models.registry import ModelHandle
        handle = ModelHandle("opencv-haar-person", proc, config={
            "inputs": [{"name": "video", "modality": "camera"}]})
        runner = handle.bind(video=camera, window_seconds=0.2)
        pred = runner.predict(warmup_s=0.3)
        assert pred.label in ("no_person", "error")
        assert pred.task == "person_presence"
        runner.stop()
    finally:
        dev.close()


@pytest.mark.skipif(not _has_cascade(),
                    reason="cv2.CascadeClassifier unavailable")
def test_demo1_model_decodes_jpeg():
    proc = _haar_model()
    sample = SensorSample.now("dev", "camera-test", "camera",
                              {"encoding": "jpeg", "width": 320,
                               "height": 240, "data": _jpeg_frame()},
                              payload_type="jpeg")
    frame = proc._decode_frame(sample)
    assert frame is not None and frame.shape == (240, 320, 3)


# -- Demo 2: speaker → remote mic → STT --------------------------------------------

class _SpeakerAdapter(ActuatorAdapter):
    """Records spoken text instead of producing sound."""

    def __init__(self):
        self.spoken = []

    def metadata(self):
        return ActuatorMeta(name="test-speaker", kinds=("speaker",))

    def discover(self):
        return [ActuatorDescriptor(
            id="speaker-test", kind="speaker", adapter="test-speaker",
            operations=["speak", "stop"], stable=True)]

    def connect(self, descriptor, config=None):
        spoken = self.spoken

        class _H(ActuatorHandle):
            @property
            def info(self):
                return descriptor

            def execute(self, command):
                if command.operation == "speak":
                    spoken.append(command.params.get("text"))
                    return ActionResult(status=ActionStatus.SUCCEEDED,
                                        action_type="speaker")
                return ActionResult(status=ActionStatus.UNSUPPORTED,
                                    action_type="speaker")

        return _H()


def _pcm_samples(text_marker: float = 0.5, n: int = 20):
    """Fake 100 ms PCM chunks (constant tone) as microphone samples."""
    import struct
    pcm = struct.pack("<1600h", *([int(1000 * text_marker)] * 1600))
    return base64.b64encode(pcm).decode("ascii")


class _MicAdapter(SensorAdapter):
    def metadata(self):
        return SensorMeta(name="test-mic", modalities=("microphone",))

    def discover(self):
        return [SensorDescriptor(
            id="microphone-test", modality="microphone", adapter="test-mic",
            hardware_id="test-mic-0", capabilities=["pcm_audio"],
            stable=True)]

    def connect(self, descriptor, config=None):
        class _H:
            @property
            def info(self):
                return descriptor.to_sensor()

            def stream(self, max_samples=None):
                for i in itertools.count():
                    yield SensorSample.now(
                        "rpi1", descriptor.id, "microphone",
                        {"encoding": "pcm_s16le", "sample_rate": 16000,
                         "channels": 1, "data": _pcm_samples()},
                        sequence=i, payload_type="pcm_s16le")
                    time.sleep(0.01)
                    if max_samples and i + 1 >= max_samples:
                        return

            def close(self):
                pass

        return _H()


def _whisper_model():
    _import_pkg("whispy-model-whisper-stt", "whispy_model_whisper_stt")
    from whispy_model_whisper_stt import WhisperSttModel
    return WhisperSttModel()


def test_demo2_orchestration_capture_and_stt():
    """Speaker.execute → capture_window → whisper-stt (stubbed backend)."""
    speaker_adapter = _SpeakerAdapter()
    dev = LocalDevice(device_id="laptop-01", drivers={}, adapters={
        "test-mic": _MicAdapter()},
        actuator_adapters={"test-speaker": speaker_adapter})
    try:
        speaker = dev.actuator("speaker")
        mic = dev.sensor("microphone")

        session = whispy.capture_window(mic, pre_roll=0.05)
        session.start()
        result = speaker.execute(whispy.Speak("test seven"))
        assert result.status is ActionStatus.SUCCEEDED
        window = session.finish(post_roll=0.1)

        assert speaker_adapter.spoken == ["test seven"]
        samples = window.samples["microphone-test"]
        assert len(samples) > 0
        assert all(s.payload["encoding"] == "pcm_s16le" for s in samples)

        stt = _whisper_model()
        # Stub the backend: transcribe returns a fixed transcript.
        class _Seg:
            text = "test seven"

        class _Info:
            language_probability = 0.99

        class _FakeModel:
            def transcribe(self, audio, **kw):
                return [_Seg()], _Info()

        stt._model = _FakeModel()
        stt._backend = "faster"
        pred = stt.predict(window)
        assert pred.attributes["text"] == "test seven"
        assert pred.task == "speech_to_text"
        assert pred.metadata["audio_sensor"] == "microphone-test"
    finally:
        dev.close()


def test_demo2_stt_empty_window_explicit():
    stt = _whisper_model()
    from whispy.contracts import SensorWindow
    pred = stt.predict(SensorWindow(start_timestamp=0, end_timestamp=1,
                                    samples={}))
    assert pred.attributes["text"] == ""
    assert "error" in pred.attributes
