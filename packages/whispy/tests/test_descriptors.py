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


def test_feature_metadata_promoted_to_fields():
    """Radar/CSI drivers put PCA/FFT/doppler features in sample
    metadata — they must surface as descriptor fields, not be dropped."""
    s = [_s("csi-1", "wifi_csi", 100 + i * 0.1, {"amp": 20.0 + i},
            seq=i, pca1_ratio=0.4 + i * 0.01, fft_low_db=-30.0 - i,
            doppler_hz=1.2, internal_state="x" * 4)
         for i in range(6)]
    d = sensor_descriptor(s, now=100.6)
    f = d["fields"]
    assert f["pca1_ratio"]["mean"] == pytest.approx(0.425)
    assert f["fft_low_db"]["min"] == pytest.approx(-35.0)
    assert f["doppler_hz"]["mean"] == pytest.approx(1.2)
    assert "internal_state" not in f


def test_vector_payload_emits_mean_and_std():
    """A per-subcarrier CSI array must keep its spread — mean alone is
    invisible to occupancy judgements."""
    s = [_s("csi-1", "wifi_csi", 100 + i * 0.1,
            {"subcarriers": [float(i), float(i + 1), float(i + 5)]},
            seq=i) for i in range(3)]
    d = sensor_descriptor(s, now=100.3)
    f = d["fields"]
    assert "subcarriers_mean" in f and "subcarriers_std" in f
    # per-sample std of [i, i+1, i+5] ≈ 2.16 — present and non-zero
    assert f["subcarriers_std"]["mean"] > 1.0


def test_field_cap_keeps_radar_feature_sets():
    """Rich radar payloads (snr + map stats + pca/fft features) must not
    be silently truncated at the old 12-field cap."""
    payload = {f"f{i}": float(i) for i in range(10)}
    payload.update({"pca1": 0.9, "pca2": 0.05, "fft_band_a": -20.0,
                    "fft_band_b": -25.0, "doppler_rms": 0.8,
                    "range_max_m": 3.2, "snr_db": 9.1})
    s = [_s("radar-1", "radar", 100.0 + i * 0.1, payload, seq=i)
         for i in range(5)]
    d = sensor_descriptor(s, now=100.5)
    f = d["fields"]
    for key in ("pca1", "pca2", "fft_band_a", "fft_band_b",
                "doppler_rms", "range_max_m"):
        assert key in f
