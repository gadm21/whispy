"""Wi-Fi CSI sensor adapter — ESP32 UDP streams, serial links, NICs.

Source kinds:

- **UDP** (``{"host","port"}`` / ``name=host:port``): ESP32 CSI tools that
  publish frames over UDP. With no config the adapter registers a wildcard
  listener on ``0.0.0.0:5500`` so a LAN transmitter shows up immediately.
- **Serial** (``serial:<port>[@<baud>]`` or ``{"type":"serial",...}``):
  esp32-csi-tool firmware printing ``CSI_DATA,...,"[i,q,...]"`` CSV lines on
  a USB UART/JTAG port. Auto-detected: every ``/dev/serial/by-id/usb-*``
  port is probed for ``CSI_DATA`` lines when ``WHISPY_CSI_SOURCES`` doesn't
  declare serial sources, so a plugged-in ESP32 appears with zero config.

Samples carry the raw CSI frame::

    payload = {"encoding": "csi_raw", "data": "<base64>",
               "seq": ..., "rssi": ..., "n_subcarriers": ...}

Serial sources base64-encode the raw I/Q int8 pairs in ``data`` (so the
dashboard amplitude/variance views reflect real CSI) and additionally
expose ``iq`` (list), ``rssi``, ``mac``, ``seq`` fields.
"""

from __future__ import annotations

import base64
import glob
import itertools
import json
import logging
import os
import socket
import time
from typing import Any, Dict, Iterator, List, Optional

from whispy.contracts import SensorDescriptor, SensorSample
from whispy.devices.base import SensorHandle
from whispy.sensors.base import HealthReport, SensorAdapter, SensorMeta

logger = logging.getLogger(__name__)

ENV_SOURCES = "WHISPY_CSI_SOURCES"
_SERIAL_GLOB = "/dev/serial/by-id/usb-*"
_DEFAULT_BAUD = 921600


def _configured_sources() -> List[Dict[str, Any]]:
    raw = os.getenv(ENV_SOURCES, "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item.startswith("serial:"):
            port = item[len("serial:"):]
            baud = _DEFAULT_BAUD
            if "@" in port:
                port, _, b = port.rpartition("@")
                baud = int(b)
            out.append({"type": "serial", "id": f"serial:{port}",
                        "serial_port": port, "baud": baud})
        elif "=" in item and ":" in item:
            name, _, addr = item.partition("=")
            host, _, port = addr.partition(":")
            out.append({"id": name.strip(), "host": host.strip(),
                        "port": int(port)})
    return out


def _serial_candidates() -> List[str]:
    """Stable USB-serial device paths (auto-detect scope)."""
    return sorted(glob.glob(_SERIAL_GLOB))


def _serial_module():
    try:
        import serial  # pyserial
        return serial
    except Exception:
        return None


def _probe_serial(port: str, baud: int = _DEFAULT_BAUD,
                  window_s: float = 2.0) -> bool:
    """True when the port prints esp32-csi-tool ``CSI_DATA`` lines."""
    serial = _serial_module()
    if serial is None:
        return False
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.25)
    except Exception:
        return False
    try:
        buf = b""
        deadline = time.monotonic() + window_s
        while time.monotonic() < deadline and len(buf) < 8192:
            chunk = ser.read(2048)
            if not chunk:
                continue
            buf += chunk
            if b"CSI_DATA" in buf:
                return True
        return False
    except Exception:
        return False
    finally:
        try:
            ser.close()
        except Exception:
            pass


def _parse_csi_line(line: bytes) -> Optional[Dict[str, Any]]:
    """Parse one ``CSI_DATA`` CSV line (esp32-csi-tool).

    Layout is firmware-dependent; only the stable fields are pulled by
    position: seq, mac, rssi — and the trailing quoted ``[i,q,...]`` array.
    """
    try:
        text = line.decode("ascii", errors="ignore").strip()
    except Exception:
        return None
    if not text.startswith("CSI_DATA,"):
        return None
    head, sep, tail = text.partition(',"')
    if not sep or not tail.rstrip().endswith('"'):
        return None
    iq_text = tail.rstrip().rstrip('"').strip()
    try:
        iq = json.loads(iq_text)
        if not isinstance(iq, list):
            return None
    except Exception:
        return None
    fields = head.split(",")
    out: Dict[str, Any] = {"iq": iq}
    try:
        out["seq"] = int(fields[1])
    except (IndexError, ValueError):
        pass
    if len(fields) > 3:
        out["mac"] = fields[2].strip()
        try:
            out["rssi"] = int(fields[3])
        except ValueError:
            pass
    return out


class _CsiHandle(SensorHandle):
    """Streams CSI frames from a UDP source."""

    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        self._config = dict(config or {})
        self._seq = itertools.count()
        self._sock: Optional[socket.socket] = None

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _open(self) -> None:
        if self._sock is not None:
            return
        host = str(self._config.get("host")
                   or self._desc.metadata.get("host") or "0.0.0.0")
        port = int(self._config.get("port")
                   or self._desc.metadata.get("port") or 5500)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.settimeout(1.0)
        self._sock = sock

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        self._open()
        assert self._sock is not None
        count = 0
        while True:
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            yield SensorSample(
                device_id="",
                sensor_id=self._desc.id,
                sensor_type="wifi_csi",
                timestamp=time.time(),
                sequence=next(self._seq),
                payload_type="csi_raw",
                payload={
                    "encoding": "csi_raw",
                    "data": base64.b64encode(data).decode("ascii"),
                    "bytes": len(data),
                    "source": f"{addr[0]}:{addr[1]}",
                },
                metadata={"adapter": "csi",
                          "hardware_id": self._desc.hardware_id},
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                return

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class _SerialCsiHandle(SensorHandle):
    """Streams ``CSI_DATA`` lines from an ESP32 serial port."""

    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        self._config = dict(config or {})
        self._seq = itertools.count()
        self._ser = None

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _open(self) -> None:
        if self._ser is not None:
            return
        serial = _serial_module()
        if serial is None:
            raise RuntimeError("pyserial not installed")
        port = str(self._config.get("serial_port")
                   or self._desc.metadata.get("serial_port") or "")
        baud = int(self._config.get("baud")
                   or self._desc.metadata.get("baud") or _DEFAULT_BAUD)
        if not port:
            raise RuntimeError("no serial port configured")
        self._ser = serial.Serial(port, baudrate=baud, timeout=0.5)

    def stream(self, max_samples: Optional[int] = None
               ) -> Iterator[SensorSample]:
        self._open()
        assert self._ser is not None
        buf = b""
        count = 0
        try:
            while True:
                try:
                    buf += self._ser.read(4096)
                except Exception:
                    return
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    parsed = _parse_csi_line(line)
                    if parsed is None:
                        continue
                    iq = parsed.pop("iq")
                    iq_bytes = bytes(
                        (v & 0xFF) for v in iq)  # int8 pairs as bytes
                    yield SensorSample(
                        device_id="",
                        sensor_id=self._desc.id,
                        sensor_type="wifi_csi",
                        timestamp=time.time(),
                        sequence=next(self._seq),
                        payload_type="csi_raw",
                        payload={
                            "encoding": "csi_raw",
                            "data": base64.b64encode(iq_bytes)
                                    .decode("ascii"),
                            "bytes": len(iq_bytes),
                            "n_subcarriers": len(iq) // 2,
                            "iq": iq,
                            "source": "serial",
                            **parsed,
                        },
                        metadata={"adapter": "csi",
                                  "hardware_id": self._desc.hardware_id},
                    )
                    count += 1
                    if max_samples is not None and count >= max_samples:
                        return
        finally:
            self.close()

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None


class CsiSensorAdapter(SensorAdapter):
    """CSI sources declared via ``WHISPY_CSI_SOURCES`` or config.

    ``discover()`` returns one ``wifi_csi`` descriptor per configured
    source; with no configuration it returns ``[]`` (no fake hardware).
    """

    def __init__(self, sources: Optional[List[Dict[str, Any]]] = None):
        self._sources = sources
        self._handles: List[_CsiHandle] = []

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="csi",
            version="0.1.0",
            modalities=("wifi_csi",),
            description="Wi-Fi CSI sources (ESP32 UDP / monitor-mode NIC)",
            config_schema={
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer"},
                    "serial_port": {"type": "string"},
                    "baud": {"type": "integer"},
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[SensorDescriptor]:
        sources = self._sources if self._sources is not None \
            else _configured_sources()
        has_udp = any("port" in s and "serial_port" not in s
                      for s in sources)
        has_serial = any(s.get("type") == "serial" or s.get("serial_port")
                         for s in sources)
        if not sources or not has_udp:
            # CSI transmitters broadcast blindly over UDP — no handshake to
            # discover. Always expose the standard listen socket so a node
            # with a receiver on the LAN reports the sensor immediately.
            sources = list(sources) + [{"id": "csi-listener",
                                        "name": "Wi-Fi CSI receiver",
                                        "host": "0.0.0.0", "port": 5500}]
        if not has_serial:
            for port in _serial_candidates():
                name = os.path.basename(port)
                if _probe_serial(port):
                    sources.append({
                        "type": "serial", "id": f"serial:{name}",
                        "name": f"ESP32 CSI ({name.split('_')[1]}"
                                f" serial)",
                        "serial_port": os.path.realpath(port),
                        "baud": _DEFAULT_BAUD})
        out: List[SensorDescriptor] = []
        for src in sources:
            if src.get("type") == "serial" or src.get("serial_port"):
                sid = str(src.get("id") or src.get("serial_port"))
                out.append(SensorDescriptor(
                    id=SensorDescriptor.make_id("csi", sid),
                    modality="wifi_csi",
                    adapter="csi",
                    name=str(src.get("name") or sid),
                    hardware_id=sid,
                    capabilities=["csi_raw", "amplitude", "phase"],
                    config_schema=self.metadata().config_schema,
                    stable=True,
                    metadata={
                        "serial_port": src.get("serial_port"),
                        "baud": src.get("baud"),
                    },
                ))
                continue
            sid = str(src.get("id") or f"{src.get('host')}:{src.get('port')}")
            out.append(SensorDescriptor(
                id=SensorDescriptor.make_id("csi", sid),
                modality="wifi_csi",
                adapter="csi",
                name=str(src.get("name") or sid),
                hardware_id=sid,
                capabilities=["csi_raw", "amplitude", "phase"],
                config_schema=self.metadata().config_schema,
                stable=True,
                metadata={"host": src.get("host"), "port": src.get("port")},
            ))
        return out

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None):
        is_serial = (descriptor.metadata.get("serial_port")
                     or (config or {}).get("serial_port"))
        handle = (_SerialCsiHandle(descriptor, config) if is_serial
                  else _CsiHandle(descriptor, config))
        self._handles.append(handle)
        return handle

    def health(self) -> HealthReport:
        return HealthReport(status="ok")

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


__all__ = ["CsiSensorAdapter"]
