import base64
import pytest
from whispy_sensor_csi import CsiSensorAdapter, _parse_line
from whispy_sensor_csi.observations import normalize_observation

MAC = "1a:00:00:00:00:00"
LINE = b'CSI_DATA,15,1a:00:00:00:00:00,-73,11,-99,223,69,6,15,47,0,4,0,"[-12,-27,-11,-26]"'


def test_health_counters_are_distinct_from_csi():
    parsed = _parse_line(b"HEALTH_DATA,5000,2,17,800")
    assert parsed["data"]["output_dropped"] == 17
    result = normalize_observation(parsed["type"], parsed["data"], {},
                                   component_id="rx", received_at=9)
    assert result["sensor_type"] == "radio.health"
    assert result["firmware_tick"] == 5000
    assert _parse_line(b"HEALTH_DATA,5000,2,-1,800") is None
    assert _parse_line(b"HEALTH_DATA,broken,2,17,800") is None


@pytest.mark.parametrize("record", [LINE, b'SELF_DATA,rx,11:22:33:44:55:66,"rx","-"',
                                   b'BLE_DATA,2,11:22:33:44:55:66,1,-71,127,"-","-"'])
def test_discovery_survives_large_boot_log(monkeypatch, record):
    import whispy_sensor_csi as mod
    class Port:
        def __init__(self, *args, **kwargs):
            self.chunks = iter([b"boot diagnostic\n" * 900, record + b"\n"])
        def read(self, size):
            return next(self.chunks, b"")
        def close(self):
            pass
    monkeypatch.setattr(mod, "_serial_module", lambda: type("SerialModule", (), {"Serial": Port}))
    assert mod._probe_serial("fake")


def test_c6_metadata_and_legacy_iq():
    data = _parse_line(LINE)["data"]
    assert data["seq"] == data["firmware_timestamp_us"] == 15
    assert data["channel"] == 6
    assert data["declared_length"] == 4
    assert data["first_word_invalid"] is False


@pytest.mark.parametrize("iq", ["[]", "[1]", "[1,256]", "[true,0]", '["1",0]', "[1.5,0]"])
def test_malformed_iq_is_rejected(iq):
    assert _parse_line(('CSI_DATA,1,' + MAC + ',-70,"' + iq + '"').encode()) is None


def test_unsigned_tick_wrap_is_preserved():
    data = _parse_line(LINE.replace(b",6,15,47,", b",6,-1,47,"))["data"]
    assert data["firmware_timestamp_us"] == 0xFFFFFFFF


def normalize(config):
    return normalize_observation("csi_raw", _parse_line(LINE)["data"], config,
                                 component_id="rx-usb-id", received_at=100.0)


def test_source_classification_requires_explicit_configuration():
    unknown = normalize({})
    direct = normalize({"direct_source_mac": MAC.upper(), "direct_source_component": "esp1"})
    ap = normalize({"ap_bssid": MAC})
    ambiguous = normalize({"ap_bssid": MAC, "direct_source_mac": MAC})
    assert unknown["stream"] == ambiguous["stream"] == "wifi_csi_unclassified/v1"
    assert direct["stream"] == "wifi_csi_direct/v1"
    assert ap["stream"] == "wifi_csi_ap/v1"
    assert direct["source_component"] == "esp1"
    assert direct["transmit_sequence"] is None


def test_quoted_scan_names():
    data = _parse_line(b'BLE_DATA,999,11:22:33:44:55:66,1,-71,127,"a,b","abcd"')["data"]
    assert data["name"] == "a,b"
    assert data["mfg"] == "abcd"


def test_invalid_prefix_does_not_gain_source_identity():
    result = normalize({"direct_source_mac": "invalid"})
    assert result["stream"] == "wifi_csi_unclassified/v1"


def test_truncated_csi_is_explicitly_limited():
    data = _parse_line(LINE.replace(b",4,0,", b",128,1,"))["data"]
    result = normalize_observation("csi_raw", data, {}, component_id="rx", received_at=1)
    assert "csi_length_mismatch" in result["quality"]["issues"]
    assert "first_word_invalid" in result["quality"]["issues"]


def test_real_adapter_separates_radio_records(monkeypatch):
    import whispy_sensor_csi as mod
    class Port:
        in_waiting = 1
        def __init__(self, *args, **kwargs):
            self.lines = iter([
                b'WIFI_DATA,1,aa:bb:cc:dd:ee:ff,bcn,-62,6,"Home"\n',
                b'BLE_DATA,2,11:22:33:44:55:66,1,-71,127,"-","-"\n',
                b'SELF_DATA,rx,11:22:33:44:55:66,"rx","-"\n', LINE + b"\n"])
        def read(self, n):
            return next(self.lines)
        def close(self):
            pass
    monkeypatch.setattr(mod, "_serial_module", lambda: type("SerialModule", (), {"Serial": Port}))
    adapter = CsiSensorAdapter(sources=[{"serial_port": "fake", "id": "stable-usb",
        "node_id": "node", "component_id": "esp2", "direct_source_mac": MAC,
        "direct_source_component": "esp1"}])
    samples = list(adapter.connect(adapter.discover()[0]).stream(max_samples=4))
    assert [s.sensor_type for s in samples] == ["radio.wifi_rssi", "radio.ble_rssi", "radio.self", "radio.wifi_csi_direct"]
    assert samples[-1].payload_type == "csi_raw"
    assert base64.b64decode(samples[-1].payload["data"]) == bytes([244, 229, 245, 230])
    assert samples[-1].payload["observation"]["node_id"] == "node"
    assert samples[-1].metadata["component_id"] == "esp2"
    assert samples[0].payload["observation"]["firmware_tick_unit"] == "ms"
