"""whispy-sensor-mmwhat: frame decode + adapter conformance."""
import os

import numpy as np

from whispy.conformance import check_sensor_adapter
from whispy_sensor_mmwhat import (
    MmwhatRadarAdapter, _decode_frame, _frame_payload, _parse_radar_cfg,
)


def _cfg(chirps, samples, rx_mask):
    return {"num_chirps_per_frame": chirps,
            "num_samples_per_chirp": samples,
            "num_antennas": bin(rx_mask).count("1"),
            "rx_mask": rx_mask}


def _pack12(values):
    """Pack a list of 12-bit ints the way the FIFO does (3 bytes/2 samples)."""
    out = bytearray()
    for i in range(0, len(values), 2):
        a, b = values[i], values[i + 1]
        out += bytes([a >> 4, ((a & 0xF) << 4) | (b >> 8), b & 0xFF])
    return bytes(out)


def _full_frame(values, seq=7):
    data = _pack12(values)
    header = (0).to_bytes(4, "little") + seq.to_bytes(4, "little") \
        + len(data).to_bytes(4, "little")
    return header + data


def test_decode_frame_3rx_layout():
    chirps, samples, rx_mask = 2, 4, 7          # 3 antennas
    n = chirps * samples * 3
    values = list(range(n))
    arr = _decode_frame(_full_frame(values), _cfg(chirps, samples, rx_mask))
    assert arr is not None
    assert arr.shape == (3, 2, 4)             # [rx, chirps, samples]
    # wire order: adc[chirp][sample][rx] → arr[rx][chirp][sample]
    assert arr[1, 0, 2] == values[0 * 4 * 3 + 2 * 3 + 1]
    assert arr[2, 1, 3] == values[1 * 4 * 3 + 3 * 3 + 2]


def test_decode_frame_rejects_garbage():
    assert _decode_frame(b"", _cfg(1, 2, 1)) is None
    assert _decode_frame(b"\x01" * 20, _cfg(1, 2, 1)) is None
    good = _full_frame(list(range(6)))
    assert _decode_frame(good[:-1], _cfg(1, 2, 3)) is None


def test_frame_payload_keys():
    arr = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
    p = _frame_payload(arr)
    assert p["encoding"] == "radar_frame"
    assert p["shape"] == [3, 4, 8]
    assert len(p["xy_map"]) == 24 and len(p["xy_map"][0]) == 24
    assert len(p["range_profile"]) == 8       # per-range-bin energy
    assert p["snr_db"] > 0 and p["energy"] > 0


def test_parse_radar_cfg_three_rx():
    setting = {"sequence": [{"repetition_time_s": 0.1, "sequence": [
        {"type": "loop", "num_repetitions": 64, "repetition_time_s": 0.001,
         "sequence": [{"type": "chirp", "num_samples": 128, "rx_mask": 7,
                       "sample_rate_Hz": 2_000_000,
                       "start_frequency_Hz": 58e9,
                       "end_frequency_Hz": 60e9}]}]}]}
    cfg = _parse_radar_cfg(setting)
    assert cfg["num_antennas"] == 3
    assert cfg["num_chirps_per_frame"] == 64
    assert abs(cfg["frame_rate"] - 10.0) < 0.1


def test_vendored_config_is_3rx():
    """Zero-config path: no release tree → vendored radar_config, rx_mask=7."""
    import whispy_sensor_mmwhat as m
    cfg_dir = m._resolve_config_dir(None, None)
    assert cfg_dir is not None
    assert m._PKG_CONFIG_BASE in os.path.abspath(cfg_dir)
    import json
    with open(m._find_one(cfg_dir, m._SETTINGS_RE)) as fh:
        cfg = m._parse_radar_cfg(json.load(fh))
    assert cfg["num_antennas"] == 3


def test_conformance():
    report = check_sensor_adapter(MmwhatRadarAdapter())
    assert report["passed"], report
