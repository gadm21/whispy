"""whispy-sensor-radio: parser tests + conformance."""
import whispy_sensor_radio  # noqa: F401  (import smoke)
from whispy_sensor_radio import (
    RadioSensorAdapter, _parse_iw_dump, _chan_from_freq,
)

NETSH = """
Interface name : Wi-Fi
There are 2 networks currently visible.

SSID 1 : HomeNet
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption              : CCMP
    BSSID 1                 : 24:4b:fe:aa:bb:cc
         Signal             : 62%
         Radio type         : 802.11ax
         Channel            : 6
         Basic rates (Mbps) : 1 2 5.5 11
         Other rates (Mbps) : 6 9 12 18 24 36 48 54
    BSSID 2                 : 24:4b:fe:aa:bb:cd
         Signal             : 40%
         Channel            : 1
SSID 2 : CafeGuest
    BSSID 1                 : aa:bb:cc:dd:ee:ff
         Signal             : 90%
         Channel            : 11
"""

IW_DUMP = """
BSS 24:4b:fe:aa:bb:cc(on wlan0)
        freq: 2437.0
        signal: -58.00 dBm
        SSID: HomeNet
BSS aa:bb:cc:dd:ee:ff(on wlan0)
        freq: 2462
        signal: -80.50 dBm
        SSID: CafeGuest
BSS de:ad:be:ef:00:11(on wlan0)
        freq: 5220
        signal: -90.00 dBm
"""


def test_parse_netsh_windows(monkeypatch):
    monkeypatch.setattr(whispy_sensor_radio, "_IS_WIN", True)
    monkeypatch.setattr(whispy_sensor_radio, "_run",
                        lambda cmd, timeout=15: NETSH)
    aps = whispy_sensor_radio._wifi_scan_windows()
    assert len(aps) == 3
    a0 = aps[0]
    assert a0["mac"] == "24:4b:fe:aa:bb:cc"
    assert a0["ssid"] == "HomeNet"
    assert a0["rssi"] == 62 // 2 - 100
    assert a0["channel"] == 6
    assert aps[1]["ssid"] == "HomeNet"
    assert aps[2]["mac"] == "aa:bb:cc:dd:ee:ff"
    assert aps[2]["channel"] == 11


def test_parse_iw_dump():
    aps = _parse_iw_dump(IW_DUMP)
    assert len(aps) == 3
    a0 = aps[0]
    assert a0["mac"] == "24:4b:fe:aa:bb:cc"
    assert a0["freq_mhz"] == 2437
    assert a0["channel"] == 6
    assert a0["rssi"] == -58
    assert a0["ssid"] == "HomeNet"
    assert aps[1]["channel"] == 11
    assert aps[2]["freq_mhz"] == 5220        # 5 GHz kept, chan mapped


def test_chan_from_freq():
    assert _chan_from_freq(2437) == 6
    assert _chan_from_freq(2462) == 11
    assert _chan_from_freq(2484) == 14
    assert _chan_from_freq(5180) == 36
    assert _chan_from_freq(915) is None


def test_conformance(monkeypatch):
    """Fake Wi-Fi scan + no bleak -> adapter still discovers and streams."""
    monkeypatch.setattr(whispy_sensor_radio, "_IS_WIN", False)
    monkeypatch.setattr(whispy_sensor_radio, "_wifi_ifaces_linux",
                        lambda: ["wlan0"])
    monkeypatch.setattr(whispy_sensor_radio, "wifi_scan",
                        lambda: [{"mac": "aa:bb:cc:dd:ee:ff", "rssi": -60,
                                  "channel": 6, "ssid": "HomeNet",
                                  "freq_mhz": 2437}])
    monkeypatch.setattr(whispy_sensor_radio, "ble_scan",
                        lambda s: [{"addr": "11:22:33:44:55:66",
                                    "addr_type": 0, "rssi": -70,
                                    "tx_power": None, "name": "watch",
                                    "mfg": "ffff676164"}])
    monkeypatch.setattr(whispy_sensor_radio, "host_mac",
                        lambda: "d8:3a:dd:11:22:33")
    adapter = RadioSensorAdapter()
    descs = adapter.discover()
    assert len(descs) == 1
    handle = adapter.connect(descs[0], {"wifi_period_s": 0.01,
                                        "ble_period_s": 0.01,
                                        "ble_window_s": 0.01})
    # BLE window starts after a ~2s settle; wifi emits once per loop,
    # so collect enough samples for BLE to appear.
    samples = list(handle.stream(max_samples=15))
    types = [s.payload_type for s in samples]
    assert types[0] == "self"
    assert "wifi_scan" in types and "ble_scan" in types
    ws = next(s for s in samples if s.payload_type == "wifi_scan")
    assert ws.payload["rssi"] == -60
    assert ws.payload["ssid"] == "HomeNet"
    adapter.close()
