"""Zigbee (802.15.4) environment sensor — device discovery + sightings.

Two backends, both config-driven (no phantom hardware):

- ``ha`` — Home Assistant REST. Polls ``/api/states`` for entities whose
  attributes carry Zigbee identity (``ieee``, ``lqi``, ``zha``…), plus
  zigbee2mqtt-style ``bridge/state``. Gives real device inventory and
  link quality without a dedicated coordinator on the node.
- ``serial`` — a coordinator that prints ``ZB_DATA`` CSV lines, matching
  the fleet's ``*_DATA`` serial convention::

      ZB_DATA,ms,addr16,ieee,cluster,rssi,lqi,"name"
      ZB_JOIN,ms,addr16,ieee,"name"
      ZB_LEAVE,ms,addr16,ieee

  Kept forward-compatible so a future ESP32-C6 802.15.4 scan role can
  feed the same stream (the C6 shares one RF frontend with Wi-Fi/BLE,
  so it must be a dedicated dongle, not the csi_recv role).

Configure via ``WHISPY_ZIGBEE`` (JSON object or array)::

    {"type": "ha", "ha_url": "http://localhost:8123", "ha_token": "..."}
    {"type": "serial", "serial_port": "/dev/ttyUSB1", "baud": 115200}

Payload types: ``zigbee_device`` (periodic inventory + LQI),
``zigbee_state`` (entity state changes), ``zigbee_join``/``zigbee_leave``.
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import os
import time
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from whispy import (HealthReport, SensorAdapter, SensorDescriptor,
                    SensorHandle, SensorMeta, SensorSample)

log = logging.getLogger(__name__)

ENV_SOURCES = "WHISPY_ZIGBEE"
_DEFAULT_BAUD = 115200
_ZIGBEE_ATTRS = ("ieee", "device_ieee", "lqi", "zigbee", "zha")


def _configured_sources() -> List[Dict[str, Any]]:
    raw = os.getenv(ENV_SOURCES, "").strip()
    if not raw:
        # Reuse the HA actuator's env when present — the Zigbee backend
        # then needs zero extra config on nodes already bridged to HA.
        ha_url, ha_token = os.getenv("HA_URL"), os.getenv("HA_TOKEN")
        if ha_url and ha_token:
            return [{"type": "ha", "ha_url": ha_url, "ha_token": ha_token}]
        return []
    try:
        data = json.loads(raw)
    except Exception:
        log.warning("%s is not valid JSON", ENV_SOURCES)
        return []
    if isinstance(data, dict):
        data = [data]
    return [d for d in data if isinstance(d, dict)]


def _unquote(field: str) -> str:
    return field.strip().strip('"').strip()


def _parse_line(line: bytes) -> Optional[Dict[str, Any]]:
    """Parse one coordinator serial line. Returns {type, data} or None."""
    try:
        text = line.decode("ascii", errors="ignore").strip()
    except Exception:
        return None
    if text.startswith("ZB_DATA,"):
        parts = next(csv.reader([text]))
        if len(parts) < 6:
            return None
        try:
            lqi = int(parts[6]) if len(parts) > 6 and parts[6] else None
            rssi = int(parts[5]) if parts[5] else None
        except ValueError:
            return None
        return {"type": "zigbee_device", "data": {
            "ms": parts[1], "addr16": parts[2], "ieee": parts[3],
            "cluster": parts[4] or None, "rssi": rssi, "lqi": lqi,
            "name": _unquote(parts[7]) if len(parts) > 7 else None}}
    if text.startswith("ZB_JOIN,") or text.startswith("ZB_LEAVE,"):
        parts = next(csv.reader([text]))
        if len(parts) < 4:
            return None
        kind = "zigbee_join" if "JOIN" in parts[0] else "zigbee_leave"
        return {"type": kind, "data": {
            "ms": parts[1], "addr16": parts[2], "ieee": parts[3],
            "name": _unquote(parts[4]) if len(parts) > 4 else None}}
    return None


def _ha_states(url: str, token: str, timeout: float = 10.0
               ) -> Optional[List[Dict[str, Any]]]:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/states",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
        return data if isinstance(data, list) else []
    except Exception as exc:
        log.debug("ha states failed: %s", exc)
        return None


def _ha_zigbee_entities(states: List[Dict[str, Any]]
                        ) -> List[Dict[str, Any]]:
    """Entities carrying Zigbee identity attributes."""
    out = []
    for s in states:
        attrs = s.get("attributes") or {}
        attr_keys = {str(k).lower() for k in attrs}
        ieee = attrs.get("ieee") or attrs.get("device_ieee")
        if not ieee and not (attr_keys & set(_ZIGBEE_ATTRS)):
            continue
        out.append({
            "entity_id": s.get("entity_id"),
            "ieee": ieee,
            "lqi": attrs.get("lqi"),
            "name": attrs.get("friendly_name"),
            "state": s.get("state"),
            "last_seen": s.get("last_changed"),
        })
    return out


class _ZigbeeHandle(SensorHandle):
    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        cfg = dict(config or {})
        meta = descriptor.metadata or {}
        self._src = {**meta, **cfg}
        self._period = float(self._src.get("period_s", 20.0))
        self._seq = itertools.count()
        self._ser = None
        self._last_state: Dict[str, str] = {}

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _sample(self, ptype: str, data: Dict[str, Any]) -> SensorSample:
        return SensorSample(
            device_id="",
            sensor_id=self._desc.id,
            sensor_type="zigbee",
            timestamp=time.time(),
            sequence=next(self._seq),
            payload_type=ptype,
            payload={"source": self._src.get("type", "serial"), **data},
            metadata={"adapter": "zigbee",
                      "hardware_id": self._desc.hardware_id},
        )

    # -- backends -----------------------------------------------------------

    def _open_serial(self) -> None:
        if self._ser is not None:
            return
        import serial  # pyserial
        self._ser = serial.Serial(
            self._src["serial_port"],
            baudrate=int(self._src.get("baud", _DEFAULT_BAUD)),
            timeout=0.5)

    def _stream_serial(self) -> Iterator[SensorSample]:
        self._open_serial()
        buf = b""
        while True:
            n = self._ser.in_waiting  # type: ignore[union-attr]
            buf += self._ser.read(n or 1)  # type: ignore[union-attr]
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                parsed = _parse_line(line)
                if parsed is not None:
                    yield self._sample(parsed["type"], parsed["data"])

    def _stream_ha(self) -> Iterator[SensorSample]:
        url, token = self._src["ha_url"], self._src["ha_token"]
        while True:
            states = _ha_states(url, token)
            if states is not None:
                for ent in _ha_zigbee_entities(states):
                    prev = self._last_state.get(ent["entity_id"])
                    self._last_state[ent["entity_id"]] = ent["state"]
                    if prev != ent["state"]:
                        yield self._sample("zigbee_state", ent)
                    else:
                        yield self._sample("zigbee_device", ent)
            time.sleep(self._period)

    def stream(self, max_samples: Optional[int] = None
               ) -> Iterator[SensorSample]:
        gen = (self._stream_serial() if self._src.get("serial_port")
               else self._stream_ha())
        for i, sample in enumerate(gen):
            yield sample
            if max_samples is not None and i + 1 >= max_samples:
                return

    def latest(self) -> Optional[SensorSample]:
        for s in self.stream(max_samples=1):
            return s
        return None

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None


class ZigbeeSensorAdapter(SensorAdapter):
    """Zigbee environment via HA entities or a serial coordinator."""

    def __init__(self, sources: Optional[List[Dict[str, Any]]] = None):
        self._sources = sources

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="zigbee",
            version="0.1.0",
            modalities=("zigbee",),
            description="Zigbee device search/sightings — HA entity "
                        "backend or serial ZB_DATA coordinator",
            config_schema={
                "type": "object",
                "properties": {
                    "type": {"enum": ["ha", "serial"]},
                    "ha_url": {"type": "string"},
                    "ha_token": {"type": "string"},
                    "serial_port": {"type": "string"},
                    "baud": {"type": "integer"},
                    "period_s": {"type": "number", "default": 20},
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[SensorDescriptor]:
        out: List[SensorDescriptor] = []
        for src in (self._sources if self._sources is not None
                    else _configured_sources()):
            if src.get("serial_port"):
                sid = f"serial:{src['serial_port']}"
                name = f"Zigbee coordinator ({src['serial_port']})"
                caps = ["zb_data", "join_leave"]
                meta = {"type": "serial", "serial_port": src["serial_port"],
                        "baud": src.get("baud", _DEFAULT_BAUD)}
            elif src.get("ha_url") and src.get("ha_token"):
                sid = f"ha:{src['ha_url']}"
                name = f"Zigbee via HA ({src['ha_url']})"
                caps = ["zigbee_device", "zigbee_state"]
                meta = {"type": "ha", "ha_url": src["ha_url"],
                        "ha_token": src["ha_token"],
                        "period_s": src.get("period_s", 20.0)}
            else:
                continue
            out.append(SensorDescriptor(
                id=SensorDescriptor.make_id("zigbee", sid),
                modality="zigbee",
                adapter="zigbee",
                name=name,
                hardware_id=sid,
                capabilities=caps,
                config_schema=self.metadata().config_schema,
                stable=True,
                metadata=meta,
            ))
        return out

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> SensorHandle:
        return _ZigbeeHandle(descriptor, config)

    def health(self) -> HealthReport:
        return HealthReport(status="ok")


__all__ = ["ZigbeeSensorAdapter"]
