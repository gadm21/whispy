"""BLE advertisement → beacon decoding shared by radio-scan sensors.

Both the host scanner (``whispy-sensor-radio`` via bleak) and the ESP32
firmware scan extension (``whispy-sensor-csi`` ``BLE_DATA`` lines) feed
``ble_scan`` payloads through :func:`decode_beacon`. When an advertisement
carries a recognised beacon record the decoded fields are attached to the
sample so consumers get structured beacon data instead of raw hex.

Recognised kinds:

- ``ibeacon``      — Apple/iBeacon mfg 0x004C, type 0x02 len 0x15.
- ``altbeacon``    — any company id, AD type 0xBEAC.
- ``eddystone_*``  — Google 0xFEAA service data (uid/url/tlm/eid).
- ``thoth_identity`` — fleet's owner tag: mfg 0xFFFF + ``th:<owner>``.
"""
from __future__ import annotations

import binascii
from typing import Any, Dict, Mapping, Optional

APPLE_COMPANY = "004c"
EDDYSTONE_SERVICE = "feaa"
THOTH_COMPANY = "ffff"


def _hex_bytes(value: Any) -> Optional[bytes]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return binascii.unhexlify(value.strip())
    except (ValueError, binascii.Error):
        return None


def _decode_mfg(company: int, data: bytes) -> Optional[Dict[str, Any]]:
    # iBeacon: 02 15 | 16-byte uuid | major | minor | tx
    if company == 0x004C and len(data) == 23 and data[0] == 0x02 and data[1] == 0x15:
        uuid = data[2:18].hex()
        return {
            "kind": "ibeacon",
            "uuid": (f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}"
                     f"-{uuid[16:20]}-{uuid[20:32]}"),
            "major": int.from_bytes(data[18:20], "big"),
            "minor": int.from_bytes(data[20:22], "big"),
            "measured_power": int.from_bytes(data[22:23], "big", signed=True),
        }
    # AltBeacon: BE AC | 20-byte id | ref rssi | mfg reserved
    if len(data) >= 24 and data[0] == 0xBE and data[1] == 0xAC:
        return {
            "kind": "altbeacon",
            "company": f"{company:04x}",
            "id": data[2:22].hex(),
            "reference_rssi": int.from_bytes(data[22:23], "big", signed=True),
        }
    # Thoth identity tag: 0xFFFF + b"th:<owner>"
    if company == 0xFFFF and data[:3] == b"th:":
        return {
            "kind": "thoth_identity",
            "owner": data[3:].decode("utf-8", errors="replace"),
        }
    return None


def _decode_eddystone(data: bytes) -> Optional[Dict[str, Any]]:
    if not data:
        return None
    frame = data[0]
    if frame == 0x00 and len(data) >= 17:
        return {
            "kind": "eddystone_uid",
            "namespace": data[2:12].hex(),
            "instance": data[12:18].hex() if len(data) >= 18 else "",
            "measured_power": int.from_bytes(data[1:2], "big", signed=True),
        }
    if frame == 0x10 and len(data) >= 3:
        url = _eddystone_url(data[2:])
        return {
            "kind": "eddystone_url",
            "url": url,
            "measured_power": int.from_bytes(data[1:2], "big", signed=True),
        }
    if frame == 0x20:
        return {"kind": "eddystone_tlm"}
    if frame == 0x30:
        return {"kind": "eddystone_eid", "eid": data[2:10].hex()}
    return None


_EST_PREFIXES = ["http://www.", "https://www.", "http://", "https://"]
_EST_SUFFIXES = {
    0x00: ".com/", 0x01: ".org/", 0x02: ".edu/", 0x03: ".net/",
    0x04: ".info/", 0x05: ".biz/", 0x06: ".gov/",
    0x07: ".com", 0x08: ".org", 0x09: ".edu", 0x0A: ".net",
    0x0B: ".info", 0x0C: ".biz", 0x0D: ".gov",
}


def _eddystone_url(raw: bytes) -> str:
    if not raw:
        return ""
    out = _EST_PREFIXES[raw[0]] if raw[0] < len(_EST_PREFIXES) else ""
    for b in raw[1:]:
        out += _EST_SUFFIXES.get(b, chr(b))
    return out


def decode_beacon(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Decode a ``ble_scan`` payload into a beacon record, or ``None``.

    Reads ``mfg`` (``<company_hex><payload_hex>`` — the whispy compact
    form), ``service_data`` (``{uuid: hex}``) and ``service_uuids``.
    """
    mfg = _hex_bytes(payload.get("mfg"))
    if mfg is not None and len(mfg) >= 3:
        company = int.from_bytes(mfg[:2], "little")
        found = _decode_mfg(company, mfg[2:])
        if found:
            return found

    service_data = payload.get("service_data")
    if isinstance(service_data, Mapping):
        for uuid, hexdata in service_data.items():
            uuid_s = str(uuid).lower()
            raw = _hex_bytes(hexdata)
            if (uuid_s == EDDYSTONE_SERVICE or uuid_s.startswith("0000feaa")) \
                    and raw is not None:
                found = _decode_eddystone(raw)
                if found:
                    return found

    uuids = payload.get("service_uuids") or []
    if not isinstance(service_data, Mapping) and any(
            str(u).lower().startswith("0000feaa") or
            str(u).lower() == EDDYSTONE_SERVICE for u in uuids):
        # UUID advertised but payload bytes unavailable — still a beacon.
        return {"kind": "eddystone"}
    return None
