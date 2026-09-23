"""Contract round-trip and validation tests (Phase 0 exit gate)."""
import pytest

from whispy.contracts import (
    Action, ActionResult, ActionStatus, Capture, Deployment,
    DeploymentState, Device, ModelManifest, Prediction, Sensor,
    SensorSample, SensorWindow, ModalityState,
)


def test_sensor_sample_roundtrip():
    s = SensorSample.now("dev-1", "radar-0", "radar", [1.0, 2.0],
                         sequence=7, payload_type="ndarray")
    s2 = SensorSample.from_dict(s.to_dict())
    assert s2.device_id == "dev-1"
    assert s2.sensor_id == "radar-0"
    assert s2.sequence == 7
    assert s2.payload == [1.0, 2.0]


def test_device_roundtrip_with_sensors():
    d = Device(id="d1", stable_uuid="uuid-1", name="thoth-pi-a",
               online=True, sensors=[Sensor(id="radar-0", type="radar")])
    d2 = Device.from_dict(d.to_dict())
    assert d2.name == "thoth-pi-a"
    assert d2.sensors[0].type == "radar"


def test_window_missing_stale_markers():
    w = SensorWindow(
        start_timestamp=0.0, end_timestamp=1.0,
        samples={"radar": []},
        modalities={
            "radar": ModalityState(sensor_id="radar", state="missing"),
            "env": ModalityState(sensor_id="env", state="stale"),
        })
    assert w.missing() == ["radar"]
    assert w.stale() == ["env"]
    assert not w.is_complete()


def test_prediction_roundtrip():
    p = Prediction(label="occupied", confidence=0.9, device_id="d1",
                   runtime_model_id="rm-1")
    p2 = Prediction.from_dict(p.to_dict())
    assert p2.label == "occupied"
    assert p2.runtime_model_id == "rm-1"


def test_action_roundtrip():
    a = Action(type="webhook", config={"url": "https://x"},
               min_confidence=0.8, cooldown_seconds=30)
    a2 = Action.from_dict(a.to_dict())
    assert a2.type == "webhook"
    assert a2.min_confidence == 0.8
    assert a2.cooldown_seconds == 30


def test_manifest_validation():
    good = ModelManifest(
        name="radar-occupancy-v2", processor="torchscript",
        inputs=[{"sensor": "radar", "window_seconds": 2.0}],
        outputs=["empty", "occupied"])
    # inputs accept dicts via from_dict path
    good = ModelManifest.from_dict(good.to_dict())
    assert good.validate() == []

    bad = ModelManifest.from_dict({
        "format": "whispy-model/v1", "name": "", "processor": "onnx",
        "inputs": []})
    errors = bad.validate()
    assert any("name" in e for e in errors)
    assert any("processor" in e for e in errors)
    assert any("input" in e for e in errors)


def test_manifest_legacy_format_accepted():
    """Pre-rename ``thoth-model/v1`` manifests still parse and validate,
    and normalize to the canonical ``whispy-model/v1`` name."""
    m = ModelManifest.from_dict({
        "format": "thoth-model/v1", "name": "legacy", "processor": "rule",
        "inputs": [{"sensor": "radar"}]})
    assert m.format == "whispy-model/v1"
    assert m.validate() == []
    # Serialized form always emits the canonical name.
    assert m.to_dict()["format"] == "whispy-model/v1"


def test_manifest_unknown_format_rejected():
    m = ModelManifest.from_dict({
        "format": "whispy-model/v0", "name": "x", "processor": "rule",
        "inputs": [{"sensor": "radar"}]})
    assert any("format" in e for e in m.validate())


def test_manifest_hash_verify():
    blob = b"fake-model-bytes"
    import hashlib
    m = ModelManifest(name="m", processor="torchscript",
                      inputs=[{"sensor": "radar"}],
                      artifact_sha256=hashlib.sha256(blob).hexdigest())
    m = ModelManifest.from_dict(m.to_dict())
    assert m.verify_artifact(blob)
    assert not m.verify_artifact(b"tampered")


def test_deployment_state_machine():
    dep = Deployment(deployment_id="dep-1", device_id="d1", model_id="m1")
    assert dep.state is DeploymentState.QUEUED
    dep.transition(DeploymentState.RECEIVED)
    dep.transition(DeploymentState.VALIDATED)
    dep.transition(DeploymentState.INSTALLED, runtime_model_id="rm-1")
    dep.transition(DeploymentState.ACKNOWLEDGED)
    dep.transition(DeploymentState.ACTIVE)
    assert dep.runtime_model_id == "rm-1"
    assert dep.state.terminal
    assert len(dep.history) == 5


def test_deployment_illegal_transition():
    dep = Deployment(deployment_id="dep-1", device_id="d1", model_id="m1")
    with pytest.raises(ValueError):
        dep.transition(DeploymentState.ACTIVE)  # must walk the chain


def test_deployment_failure_path():
    dep = Deployment(deployment_id="dep-1", device_id="d1", model_id="m1")
    dep.transition(DeploymentState.RECEIVED)
    from whispy.contracts import DeploymentFailure
    dep.transition(DeploymentState.FAILED,
                   failure=DeploymentFailure(stage="validated", code="bad_hash",
                                             message="sha mismatch"))
    assert dep.state is DeploymentState.FAILED
    assert dep.failure.code == "bad_hash"


def test_capture_roundtrip():
    c = Capture(id="cap-1", device_id="d1", started_at=100.0,
                sensors=["radar"], sample_counts={"radar": 42})
    c2 = Capture.from_dict(c.to_dict())
    assert c2.sample_counts == {"radar": 42}
