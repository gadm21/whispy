"""Processor tests — rules, fusion, torchscript contract."""
import pytest

from whispy.contracts import SensorSample, SensorWindow, ModalityState
from whispy.processors import (
    FusionProcessor, RuleProcessor, TorchScriptProcessor, create_processor,
)


def _window(samples_by_sensor, modalities=None):
    samples = {}
    for sid, payloads in samples_by_sensor.items():
        samples[sid] = [
            SensorSample(device_id="d1", sensor_id=sid,
                         sensor_type=sid.split("-")[0],
                         timestamp=float(i), sequence=i,
                         payload_type="list", payload=p)
            for i, p in enumerate(payloads)
        ]
    mods = modalities or {sid: ModalityState(sensor_id=sid, state="ok")
                          for sid in samples}
    return SensorWindow(start_timestamp=0, end_timestamp=1,
                        samples=samples, modalities=mods)


# -- RuleProcessor -----------------------------------------------------------

def test_rule_simple_threshold():
    proc = RuleProcessor({
        "name": "snr-occupancy",
        "rules": [{"when": "snr_mean > 12", "label": "occupied",
                   "confidence": 0.9}],
        "else": "empty"})
    w = _window({"snr": [[15.0], [13.0]]})
    pred = proc.predict(w)
    assert pred.label == "occupied"
    assert pred.confidence == 0.9


def test_rule_compound_and_or():
    proc = RuleProcessor({
        "rules": [
            {"when": "snr_mean > 12 and radar_energy > 0.4",
             "label": "occupied"},
            {"when": "snr_mean > 100 or radar_max > 50", "label": "burst"},
        ],
        "else": "empty"})
    # and-path fails (energy too low), or-path fails → else
    w = _window({"snr": [[13.0]], "radar": [[0.1]]})
    assert proc.predict(w).label == "empty"
    # or-path matches
    w2 = _window({"snr": [[5.0]], "radar": [[60.0]]})
    assert proc.predict(w2).label == "burst"


def test_rule_params_and_legacy_single_rule():
    proc = RuleProcessor({
        "rule": "snr_mean > snr_threshold",
        "target_label": "occupied",
        "params": {"snr_threshold": 12.0},
        "else": "empty"})
    w = _window({"snr": [[15.0]]})
    assert proc.predict(w).label == "occupied"


def test_rule_missing_feature_does_not_match():
    proc = RuleProcessor({
        "rules": [{"when": "csi_variance > 5", "label": "motion"}],
        "else": "still"})
    w = _window({"radar": [[1.0]]})  # no csi sensor
    assert proc.predict(w).label == "still"


def test_rule_invalid_expression_fails_at_deploy():
    with pytest.raises(Exception):
        RuleProcessor({"rules": [{"when": "snr_mean >> 12 !!"}]})


def test_rule_incomplete_expression_fails_at_deploy():
    """'x >' tokenizes fine but is not a valid expression — must raise."""
    with pytest.raises(Exception):
        RuleProcessor({"rules": [{"when": "x >", "label": "y"}]})
    with pytest.raises(Exception):
        RuleProcessor({"rules": [{"when": "(a > 1", "label": "y"}]})
    with pytest.raises(Exception):
        RuleProcessor({"rules": [{"when": "a > 1 b", "label": "y"}]})


def test_rule_not_and_parens():
    proc = RuleProcessor({
        "rules": [{"when": "not (snr_mean > 12)", "label": "quiet"}],
        "else": "loud"})
    assert proc.predict(_window({"snr": [[5.0]]})).label == "quiet"
    assert proc.predict(_window({"snr": [[20.0]]})).label == "loud"


# -- FusionProcessor -----------------------------------------------------------

def test_fusion_abstains_on_missing_required():
    proc = FusionProcessor({
        "name": "hvac",
        "inputs": [{"sensor": "radar"}, {"sensor": "env"}],
        "rules": [{"when": "radar_mean > 0", "label": "occupied"}],
        "else": "empty"})
    w = _window(
        {"radar": [[1.0]]},
        modalities={
            "radar": ModalityState(sensor_id="radar", state="ok"),
            "env": ModalityState(sensor_id="env", state="missing"),
        })
    pred = proc.predict(w)
    assert pred.label == "unknown"
    assert "env" in pred.metadata["unavailable"]


def test_fusion_runs_when_complete():
    proc = FusionProcessor({
        "inputs": [{"sensor": "radar"}, {"sensor": "env"}],
        "rules": [{"when": "radar_mean > 0", "label": "occupied"}],
        "else": "empty"})
    w = _window({"radar": [[1.0]], "env": [[22.0]]})
    assert proc.predict(w).label == "occupied"


def test_fusion_invalid_on_missing():
    with pytest.raises(ValueError):
        FusionProcessor({"on_missing": "bogus"})


def test_fusion_abstains_when_required_never_opened():
    """A required modality with no marker AND no samples is unavailable."""
    proc = FusionProcessor({
        "inputs": [{"sensor": "radar"}, {"sensor": "env"}],
        "rules": [{"when": "radar_mean > 0", "label": "occupied"}],
        "else": "empty"})
    # Only radar produced data; env never opened → no marker, no samples.
    w = _window({"radar": [[1.0]]},
                modalities={"radar": ModalityState(sensor_id="radar",
                                                   state="ok")})
    pred = proc.predict(w)
    assert pred.label == "unknown"
    assert "env" in pred.metadata["unavailable"]


def test_fusion_resolves_modality_to_sensor_id():
    """Manifest modality 'radar' resolves to sensor id 'radar-0'."""
    proc = FusionProcessor({
        "inputs": [{"sensor": "radar"}, {"sensor": "env"}],
        "rules": [{"when": "radar_mean > 0", "label": "occupied"}],
        "else": "empty"})
    w = _window({"radar-0": [[1.0]], "env-0": [[22.0]]})
    assert proc.predict(w).label == "occupied"


def test_fusion_abstains_when_required_id_absent():
    proc = FusionProcessor({
        "inputs": [{"sensor": "radar"}, {"sensor": "env"}],
        "rules": [{"when": "radar_mean > 0", "label": "occupied"}],
        "else": "empty"})
    # env-0 absent entirely → required 'env' unresolved → abstain.
    w = _window({"radar-0": [[1.0]]})
    assert proc.predict(w).label == "unknown"


# -- TorchScriptProcessor -------------------------------------------------------

def test_torchscript_unloaded_returns_error():
    proc = TorchScriptProcessor({
        "name": "m", "inputs": [{"sensor": "radar"}],
        "outputs": ["empty", "occupied"]})
    pred = proc.predict(_window({"radar": [[1.0]]}))
    assert pred.label == "error"


def test_torchscript_missing_input_explicit():
    proc = TorchScriptProcessor({
        "name": "m", "inputs": [{"sensor": "radar"}],
        "outputs": ["empty", "occupied"]})
    proc._model = object()  # pretend loaded
    w = _window({"csi": [[1.0]]},
                modalities={"radar": ModalityState(sensor_id="radar",
                                                   state="missing")})
    pred = proc.predict(w)
    assert pred.label == "unknown"
    assert "radar" in pred.metadata["missing"]


# -- factory ------------------------------------------------------------------

def test_create_processor_dispatch():
    assert isinstance(create_processor({"processor": "rule", "rules": []}),
                      RuleProcessor)
    assert isinstance(create_processor({"processor": "fusion", "inputs": []}),
                      FusionProcessor)
    assert isinstance(
        create_processor({"processor": "torchscript", "inputs": []}),
        TorchScriptProcessor)
    with pytest.raises(ValueError):
        create_processor({"processor": "onnx"})
