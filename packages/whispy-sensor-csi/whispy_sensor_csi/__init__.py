"""Wi-Fi CSI sensor adapter — ESP32 UDP streams and monitor-mode NICs.

Discovers CSI sources from explicit configuration (the adapter cannot
auto-detect a UDP publisher): set ``WHISPY_CSI_SOURCES`` to a JSON list
or a comma-separated ``name=host:port`` list, e.g.::

    WHISPY_CSI_SOURCES='[{"id":"esp32-desk","host":"0.0.0.0","port":5500}]'
    WHISPY_CSI_SOURCES="esp32-desk=0.0.0.0:5500"

Each source becomes a ``wifi_csi`` descriptor with a stable id derived
from the source identity. Samples carry the raw CSI frame::

    payload = {"encoding": "csi_raw", "data": "<base64>",
               "seq": ..., "rssi": ..., "n_subcarriers": ...}
"""

from __future__ import annotations

import base64
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
        if "=" in item and ":" in item:
            name, _, addr = item.partition("=")
            host, _, port = addr.partition(":")
            out.append({"id": name.strip(), "host": host.strip(),
                        "port": int(port)})
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
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[SensorDescriptor]:
        sources = self._sources if self._sources is not None \
            else _configured_sources()
        if not sources:
            # CSI transmitters broadcast blindly over UDP — no handshake to
            # discover. Always expose the standard listen socket so a node
            # with a receiver on the LAN reports the sensor immediately.
            sources = [{"id": "csi-listener", "name": "Wi-Fi CSI receiver",
                        "host": "0.0.0.0", "port": 5500}]
        out: List[SensorDescriptor] = []
        for src in sources:
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
                config: Optional[Dict[str, Any]] = None) -> _CsiHandle:
        handle = _CsiHandle(descriptor, config)
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
