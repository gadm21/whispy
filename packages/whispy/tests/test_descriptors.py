"""Physical descriptors — compact per-sensor summaries of a SensorWindow."""
import json

import pytest

from whispy.contracts import ModalityState, SensorSample, SensorWindow
from whispy.descriptors import sensor_descriptor, window_descriptors


def _s(sid, stype, ts, payload, seq=0, **meta):
    return SensorSample(device_id="d", sensor_id=sid, sensor_type=stype,
                        timestamp=ts, sequence=seq, payload_type="auto",
                        payload=payload, metadata=meta)


def _window():
    radar = [_s("radar-1", "radar", 100 + i * 0.1,
                {"snr_db": 10.0 + i, "shape": [3, 64, 128],
                 "range_profile": [1.0, 2.0, 3.0],
                 "views_b64": "AAAA"}, i) for i in range(10)]
    imu = [_s("imu-1", "imu", 100 + i * 0.5, [0.0, 0.0, 9.81], i)
           for i in range(4)]
    ble = [_s("radio-1", "ble_scan", 100.5,
              {"kind": "ble_scan", "devices": [
                  {"mac": "aa", "rssi": -50, "name": "watch"},
                  {"mac": "bb", "rssi": -80},
                  {"mac": "aa", "rssi": -45}]})]
    return SensorWindow(
        start_timestamp=100.0, end_timestamp=101.0,
        samples={"radar-1": radar, "imu-1": imu, "radio-1": ble},
        modalities={"cam-1": ModalityState("cam-1", "missing")})


def test_descriptors_summarise_fields_without_raw_payloads():
    d = window_descriptors(_window(), now=101.0)
    radar = d["radar-1"]
    assert radar["type"] == "radar" and radar["n"] == 10
    assert radar["rate_hz"] == pytest.approx(10.0, rel=0.01)
    assert radar["fields"]["snr_db"]["mean"] == pytest.approx(14.5)
    assert "range_profile_mean" in radar["fields"]
    assert "views_b64" not in json.dumps(d)
    assert d["imu-1"]["fields"]["magnitude"]["mean"] == pytest.approx(9.81)
    assert d["cam-1"] == {"type": None, "n": 0, "state": "missing"}


def test_scan_sensor_reports_emitter_count_and_full_top():
    d = window_descriptors(_window(), detail="descriptors", now=101.0)
    assert d["radio-1"]["scan"] == {"emitters": 2}
    full = window_descriptors(_window(), detail="full", now=101.0)
    top = full["radio-1"]["scan"]["top"]
    assert top[0] == {"id": "aa", "rssi": -45.0, "name": "watch"}


def test_minimal_detail_has_no_fields():
    d = window_descriptors(_window(), detail="minimal", now=101.0)
    assert set(d["radar-1"]) == {"type", "n", "rate_hz", "age_s"}


def test_descriptors_are_json_safe_and_reject_bad_detail():
    json.dumps(window_descriptors(_window(), detail="full", now=101.0))
    with pytest.raises(ValueError):
        window_descriptors(_window(), detail="raw")


def test_empty_samples():
    assert sensor_descriptor([]) == {"type": None, "n": 0}
