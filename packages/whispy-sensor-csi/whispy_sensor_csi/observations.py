"""Canonical metadata for legacy ESP serial observations.

Classification uses explicit host source configuration. Unknown sources stay
unclassified; proximity, packet shape, and the existence of a TX are not proof
of a direct propagation path. Legacy payloads remain available to consumers.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


def _mac(value: Any) -> str | None:
    value = str(value or "").strip().lower().replace("-", ":")
    return value if re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", value) else None


def normalize_observation(payload_type: str, payload: Mapping[str, Any],
                          config: Mapping[str, Any], *, component_id: str,
                          received_at: float) -> dict[str, Any]:
    source = _mac(payload.get("mac") or payload.get("addr"))
    sensor_type, stream, radio, measurement = {
        "wifi_scan": ("radio.wifi_rssi", "wifi_rssi/v1", "wifi", "rssi"),
        "ble_scan": ("radio.ble_rssi", "ble_rssi/v1", "ble", "rssi"),
        "self": ("radio.self", "radio_self/v1", None, "identity"),
        "radio_health": ("radio.health", "radio_health/v1", None, "health"),
        "csi_raw": ("wifi_csi", "wifi_csi_unclassified/v1", "wifi", "csi"),
    }[payload_type]
    direct, ap = _mac(config.get("direct_source_mac")), _mac(config.get("ap_bssid"))
    issues: list[str] = []
    source_component = None
    if payload_type == "csi_raw":
        if source and source == direct == ap:
            issues.append("ambiguous_source_configuration")
        elif source and source == direct:
            sensor_type, stream = "radio.wifi_csi_direct", "wifi_csi_direct/v1"
            source_component = config.get("direct_source_component") or None
        elif source and source == ap:
            sensor_type, stream = "radio.wifi_csi_ap", "wifi_csi_ap/v1"
        else:
            issues.append("unclassified_csi_source")
        if payload.get("first_word_invalid"):
            issues.append("first_word_invalid")
        if (payload.get("declared_length") is not None
                and payload["declared_length"] != len(payload.get("iq", []))):
            issues.append("csi_length_mismatch")
    tick = payload.get("firmware_timestamp_us")
    tick_unit = "us" if tick is not None else None
    if payload_type in ("wifi_scan", "ble_scan", "radio_health"):
        try:
            tick, tick_unit = int(payload["ms"]), "ms"
        except (KeyError, ValueError, TypeError):
            issues.append("missing_firmware_timestamp")
    return {
        "format": "radio-observation/v1",
        "node_id": config.get("node_id") or None,
        "component_id": config.get("component_id") or component_id,
        "sensor_type": sensor_type,
        "stream": stream,
        "radio": radio,
        "measurement_type": measurement,
        "source_id": source,
        "source_component": source_component,
        "channel": payload.get("channel"),
        "rssi": payload.get("rssi"),
        "host_received_at": received_at,
        "firmware_tick": tick,
        "firmware_tick_unit": tick_unit,
        # Legacy CSV `seq` is kept in the parent payload for compatibility.
        # Do not use an unverified counter to calculate packet loss.
        "transmit_sequence": None,
        "quality": {"status": "limited" if issues else "reported", "issues": issues},
        "provenance": {"transport": "usb_serial", "protocol": "legacy_csv",
                       "classification": "configured_source_mac" if stream in
                       ("wifi_csi_direct/v1", "wifi_csi_ap/v1") else "record_type"},
    }
