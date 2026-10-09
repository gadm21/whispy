"""whispy-sensor-csi: source parsing + CSI_DATA line decode + conformance."""
import base64

from whispy.conformance import check_sensor_adapter
from whispy_sensor_csi import (
    CsiSensorAdapter, _configured_sources, _parse_line,
)

LINE = (b'CSI_DATA,227064187,1a:00:00:00:00:00,-73,11,-99,223,69,6,'
        b'227064187,47,0,128,0,"[0,0,0,0,0,0,0,0,-12,-27,-11,-26,-10,-27]'
        b'"')


def test_parse_csi_line():
    out = _parse_line(LINE)
    assert out is not None
    assert out["type"] == "csi_raw"
    out = out["data"]
    assert out["seq"] == 227064187
    assert out["mac"] == "1a:00:00:00:00:00"
    assert out["rssi"] == -73
    assert out["iq"] == [0, 0, 0, 0, 0, 0, 0, 0, -12, -27, -11, -26,
                         -10, -27]


def test_parse_csi_line_rejects_non_csi():
    assert _parse_line(b"hello") is None
    assert _parse_line(b"CSI_DATA,1,2") is None      # no quoted array
    assert _parse_line(b"") is None


def test_parse_wifi_scan_line():
    out = _parse_line(
        b'WIFI_DATA,12345,aa:bb:cc:dd:ee:ff,bcn,-62,6,"MyHome"')
    assert out == {"type": "wifi_scan", "data": {
        "ms": "12345", "mac": "aa:bb:cc:dd:ee:ff", "kind": "bcn",
        "rssi": -62, "channel": 6, "ssid": "MyHome"}}


def test_parse_ble_scan_line():
    out = _parse_line(
        b'BLE_DATA,999,11:22:33:44:55:66,1,-71,127,"thoth-csi-tx","ffff6761"')
    assert out["type"] == "ble_scan"
    d = out["data"]
    assert d["addr"] == "11:22:33:44:55:66"
    assert d["rssi"] == -71
    assert d["tx_power"] is None          # 127 = absent
    assert d["name"] == "thoth-csi-tx"


def test_parse_self_line():
    out = _parse_line(
        b'SELF_DATA,rx,80:65:99:aa:bb:cc,"thoth-csi-rx","gad21"')
    assert out == {"type": "self", "data": {
        "role": "rx", "mac": "80:65:99:aa:bb:cc",
        "name": "thoth-csi-rx", "owner": "gad21"}}


def test_parse_net_line():
    out = _parse_line(b"NET_DATA,10.0.0.57,thoth-esp32-bbcc.local,5000,6")
    assert out == {"type": "net", "data": {
        "ip": "10.0.0.57", "hostname": "thoth-esp32-bbcc.local",
        "port": 5000, "channel": 6}}
    assert _parse_line(b"NET_DATA,10.0.0.57,h,notaport") is None


def test_configured_sources_serial_syntax(monkeypatch):
    monkeypatch.setenv("WHISPY_CSI_SOURCES",
                       "serial:/dev/ttyACM0@115200,esp=0.0.0.0:5500")
    srcs = _configured_sources()
    # only the serial source is valid — UDP syntax is ignored
    assert len(srcs) == 1
    assert srcs[0]["serial_port"] == "/dev/ttyACM0"
    assert srcs[0]["baud"] == 115200


class _FakeSerial:
    def __init__(self, port, baudrate=None, timeout=None,
                 chunks=()):
        self._chunks = list(chunks)

    @property
    def in_waiting(self):
        return sum(len(c) for c in self._chunks)

    def read(self, n):
        if not self._chunks:
            return b""
        head = self._chunks.pop(0)
        if len(head) > n:
            self._chunks.insert(0, head[n:])
            return head[:n]
        return head

    def close(self):
        pass


def _fake_serial_factory(chunks):
    class FakeSerialModule:
        class Serial(_FakeSerial):
            def __init__(self, port, baudrate=None, timeout=None):
                super().__init__(port, baudrate, timeout, chunks)
    return FakeSerialModule


def _serial_adapter():
    return CsiSensorAdapter(
        sources=[{"type": "serial", "id": "serial:t",
                  "serial_port": "/dev/fake", "baud": 921600}])


def test_serial_handle_streams(monkeypatch):
    """Feed a fake pyserial: CSI_DATA lines -> csi_raw samples."""
    import whispy_sensor_csi as mod

    monkeypatch.setattr(mod, "_serial_module",
                        lambda: _fake_serial_factory(
                            [LINE + b"\n" + LINE + b"\n"]))
    adapter = _serial_adapter()
    desc = [d for d in adapter.discover()
            if d.metadata.get("serial_port")][0]
    handle = adapter.connect(desc)
    samples = list(handle.stream(max_samples=2))
    assert len(samples) == 2
    s = samples[0]
    assert s.payload_type == "csi_raw"
    assert s.payload["encoding"] == "csi_raw"
    assert s.payload["rssi"] == -73
    assert s.payload["n_subcarriers"] == 7
    raw = base64.b64decode(s.payload["data"])
    assert len(raw) == 14
    adapter.close()


def test_serial_handle_streams_scan_lines(monkeypatch):
    """Scan-extension lines surface as their own payload types."""
    import whispy_sensor_csi as mod
    blob = (b'WIFI_DATA,1,aa:bb:cc:dd:ee:ff,bcn,-62,6,"Home"\n'
            b'BLE_DATA,2,11:22:33:44:55:66,1,-71,127,"-","-"\n' +
            LINE + b"\n")
    monkeypatch.setattr(mod, "_serial_module",
                        lambda: _fake_serial_factory([blob]))
    adapter = _serial_adapter()
    desc = [d for d in adapter.discover()
            if d.metadata.get("serial_port")][0]
    handle = adapter.connect(desc)
    types = [s.payload_type for s in handle.stream(max_samples=3)]
    assert types == ["wifi_scan", "ble_scan", "csi_raw"]
    adapter.close()


def test_conformance(monkeypatch):
    """Serial source first so the stream check exercises it; the auto
    UDP listener is second and stays untouched."""
    import whispy_sensor_csi as mod
    monkeypatch.setattr(mod, "_serial_module",
                        lambda: _fake_serial_factory([LINE + b"\n"] * 8))
    report = check_sensor_adapter(_serial_adapter())
    assert report["passed"], report
