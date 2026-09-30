"""Wi-Fi CSI sensor adapter — ESP32 boards streaming CSI over serial.

esp32-csi-tool firmware prints ``CSI_DATA,...,\"[i,q,...]\"`` CSV lines on
its USB UART/JTAG serial port (~200 Hz). This adapter speaks only that
serial protocol — the ESP32 plugged into the node is the receiver.

- Explicit source: ``WHISPY_CSI_SOURCES`` entries ``serial:<port>[@<baud>]``
  or JSON ``{"serial_port": "...", "baud": ...}``.
- **Auto-detect**: every ``/dev/serial/by-id/usb-*`` port is probed for
  ``CSI_DATA`` lines when no serial source is declared, so a plugged-in
  ESP32 appears with zero config and no phantom sensors.

Samples carry the decoded CSI frame (``encoding csi_raw``): the raw int8
I/Q pairs base64'd in ``data`` (dashboard amp/variance views work) plus
parsed ``iq`` (list), ``rssi``, ``mac``, ``seq``, ``n_subcarriers``.
"""

from __future__ import annotations

import base64
import glob
import itertools
import json
import logging
import os
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
                    n = self._ser.in_waiting
                    buf += self._ser.read(n or 1)
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
    """ESP32 serial CSI sources — auto-detected or via WHISPY_CSI_SOURCES."""

    def __init__(self, sources: Optional[List[Dict[str, Any]]] = None):
        self._sources = sources
        self._handles: List[_SerialCsiHandle] = []

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="esp32_csi",
            version="0.2.0",
            modalities=("wifi_csi",),
            description="ESP32 CSI receiver on USB serial "
                        "(CSI_DATA lines, esp32-csi-tool)",
            config_schema={
                "type": "object",
                "properties": {
                    "serial_port": {"type": "string"},
                    "baud": {"type": "integer"},
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[SensorDescriptor]:
        sources = [s for s in (self._sources if self._sources is not None
                               else _configured_sources())
                   if s.get("serial_port") or s.get("type") == "serial"]
        if not sources:
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
            sid = str(src.get("id") or src.get("serial_port"))
            out.append(SensorDescriptor(
                id=SensorDescriptor.make_id("csi", sid),
                modality="wifi_csi",
                adapter="esp32_csi",
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
        return out

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None):
        handle = _SerialCsiHandle(descriptor, config)
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
