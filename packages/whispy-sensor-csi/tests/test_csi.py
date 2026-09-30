"""whispy-sensor-csi: source parsing + CSI_DATA line decode + conformance."""
import base64

from whispy.conformance import check_sensor_adapter
from whispy_sensor_csi import (
    CsiSensorAdapter, _configured_sources, _parse_csi_line,
)

LINE = (b'CSI_DATA,227064187,1a:00:00:00:00:00,-73,11,-99,223,69,6,'
        b'227064187,47,0,128,0,"[0,0,0,0,0,0,0,0,-12,-27,-11,-26,-10,-27]'
        b'"')


def test_parse_csi_line():
    out = _parse_csi_line(LINE)
    assert out is not None
    assert out["seq"] == 227064187
    assert out["mac"] == "1a:00:00:00:00:00"
    assert out["rssi"] == -73
    assert out["iq"] == [0, 0, 0, 0, 0, 0, 0, 0, -12, -27, -11, -26,
                         -10, -27]


def test_parse_csi_line_rejects_non_csi():
    assert _parse_csi_line(b"hello") is None
    assert _parse_csi_line(b"CSI_DATA,1,2") is None      # no quoted array
    assert _parse_csi_line(b"") is None


def test_configured_sources_serial_syntax(monkeypatch):
    monkeypatch.setenv("WHISPY_CSI_SOURCES",
                       "serial:/dev/ttyACM0@115200,esp=0.0.0.0:5500")
    srcs = _configured_sources()
    serial = [s for s in srcs if s.get("type") == "serial"]
    udp = [s for s in srcs if s.get("host")]
    assert serial and serial[0]["serial_port"] == "/dev/ttyACM0"
    assert serial[0]["baud"] == 115200
    assert udp and udp[0]["id"] == "esp"


class _FakeSerial:
    def __init__(self, port, baudrate=None, timeout=None,
                 chunks=()):
        self._chunks = list(chunks)

    def read(self, n):
        return self._chunks.pop(0) if self._chunks else b""

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


def test_conformance(monkeypatch):
    """Serial source first so the stream check exercises it; the auto
    UDP listener is second and stays untouched."""
    import whispy_sensor_csi as mod
    monkeypatch.setattr(mod, "_serial_module",
                        lambda: _fake_serial_factory([LINE + b"\n"] * 8))
    report = check_sensor_adapter(_serial_adapter())
    assert report["passed"], report
