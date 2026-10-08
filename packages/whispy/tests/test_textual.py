"""Textual descriptors — rule sentences + model-backed cues (fake models)."""
from whispy.contracts import Prediction, SensorSample, SensorWindow
from whispy.textual import (TextualDescriber, face_provider, person_provider,
                            physical_sentence, speech_provider)


def _s(sid, stype, ts, payload, seq=0):
    return SensorSample(device_id="d", sensor_id=sid, sensor_type=stype,
                        timestamp=ts, sequence=seq, payload_type="auto",
                        payload=payload)


def _window():
    return SensorWindow(start_timestamp=100.0, end_timestamp=102.0, samples={
        "radar-1": [_s("radar-1", "radar", 100 + i * 0.1,
                       {"snr_db": 10.0 + (i % 2) * 6}, i) for i in range(10)],
        "mic-1": [_s("mic-1", "microphone", 101.0, [0, 1000, -1000, 500])],
        "cam-1": [_s("cam-1", "camera", 101.0, {"jpeg_b64": "xx"})],
        "radio-1": [_s("radio-1", "ble_scan", 101.0, {"kind": "ble_scan",
                       "devices": [{"mac": "aa", "rssi": -45, "name": "watch"},
                                   {"mac": "bb", "rssi": -80}]})],
    })


class FakeModel:
    def __init__(self, pred):
        self.pred = pred
        self.calls = 0
        self.seen = []

    def predict(self, window):
        self.calls += 1
        self.seen.append(dict(window.preprocessing))
        return self.pred(window) if callable(self.pred) else self.pred


def _factory(models):
    def make(name, config):
        if name not in models:
            raise ImportError(f"no plugin {name}")
        return models[name]
    return make


def test_physical_sentences_per_type():
    radar = physical_sentence({"type": "radar", "n": 10, "age_s": 0.1,
                               "fields": {"snr_db": {"mean": 13, "std": 3.0}}})
    assert radar["text"].startswith("radar: high motion")
    assert radar["cues"]["motion"] == "high"
    imu = physical_sentence({"type": "imu", "n": 5,
                             "fields": {"magnitude": {"std": 0.1}}})
    assert imu["cues"]["motion"] == "still"
    scan = physical_sentence({"type": "ble_scan", "n": 1, "scan": {
        "emitters": 2, "top": [{"id": "aa", "rssi": -45, "name": "watch"}]}})
    assert scan["text"] == "ble scan: 2 emitters; strongest watch -45 dBm"
    stale = physical_sentence({"type": "radar", "n": 3, "age_s": 60,
                               "fields": {"snr_db": {"std": 0.1}}})
    assert "stale" in stale["text"] and stale["cues"]["stale"]
    assert physical_sentence({"type": "camera", "n": 0, "state": "missing"}) \
        == {"text": "camera: missing", "cues": {"state": "missing"}}


def test_describer_adds_speech_person_face_cues():
    whisper = FakeModel(Prediction(label="lights off", confidence=0.9,
                                   metadata={"text": "lights off"}))
    person = FakeModel(Prediction(label="person", confidence=0.8,
                                  people_count=2))
    det = FakeModel(Prediction(label="face", confidence=0.9,
                               metadata={"detections": [[1, 2, 3, 4]]}))
    rec = FakeModel(Prediction(label="person:gad", confidence=0.7,
                               metadata={"distance": 6.8}))
    fac = _factory({"whisper-stt": whisper, "opencv-haar-person": person,
                    "opencv-haar-face": det, "pca-face-recognizer": rec})
    d = TextualDescriber.default(face_recognizer={"gallery": {}}, factory=fac)
    out = d.describe(_window(), now=200.0)
    assert 'speech: "lights off"' in out["mic-1"]["text"]
    assert out["mic-1"]["cues"]["speech"]["text"] == "lights off"
    assert "2 people visible" in out["cam-1"]["text"]
    assert "face: gad (distance 6.8)" in out["cam-1"]["text"]
    assert out["cam-1"]["cues"]["identity"] == "gad"
    # detector detections are handed to the recognizer
    assert rec.seen[-1]["detections"] == [[1, 2, 3, 4]]
    assert "strongest watch" in out["radio-1"]["text"]
    assert d.summary(out).count("|") == 3


def test_model_providers_are_rate_limited_and_cached():
    whisper = FakeModel(Prediction(label="hi", metadata={"text": "hi"}))
    d = TextualDescriber([speech_provider(factory=_factory(
        {"whisper-stt": whisper}), min_interval_s=10)])
    a = d.describe(_window(), now=100.0)
    b = d.describe(_window(), now=105.0)
    assert whisper.calls == 1 and a["mic-1"]["text"] == b["mic-1"]["text"]
    d.describe(_window(), now=111.0)
    assert whisper.calls == 2


def test_missing_plugin_degrades_to_rules_with_error_cue():
    d = TextualDescriber([person_provider(factory=_factory({}))])
    out = d.describe(_window(), now=1.0)
    assert out["cam-1"]["text"].startswith("camera:")
    assert "no plugin" in out["cam-1"]["cues"]["person"]["error"]


def test_no_face_and_unknown_face():
    det_none = FakeModel(Prediction(label="no_face", metadata={"detections": []}))
    d = TextualDescriber([face_provider(factory=_factory(
        {"opencv-haar-face": det_none}))])
    assert d.describe(_window(), now=1.0)["cam-1"]["cues"]["faces"] == 0
    det = FakeModel(Prediction(label="face", metadata={"detections": [[0, 0, 1, 1]]}))
    rec = FakeModel(Prediction(label="person:unknown", metadata={}))
    d = TextualDescriber([face_provider({"x": 1}, factory=_factory(
        {"opencv-haar-face": det, "pca-face-recognizer": rec}))])
    out = d.describe(_window(), now=1.0)["cam-1"]
    assert out["cues"]["identity"] == "unknown"
    assert "not recognized" in out["text"]
