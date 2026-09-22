"""Home Assistant MQTT bridge.

Publishes Thoth device state as HA MQTT-discovery entities:

    binary_sensor.thoth_<room>_occupancy
    sensor.thoth_<room>_activity
    sensor.thoth_<room>_people_count
    sensor.thoth_<room>_confidence
    sensor.thoth_<device>_health

Requires ``paho-mqtt`` (optional dependency).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class HomeAssistantBridge:
    """Publish device predictions to Home Assistant over MQTT."""

    def __init__(self, host: str = "localhost", port: int = 1883,
                 username: Optional[str] = None, password: Optional[str] = None,
                 discovery_prefix: str = "homeassistant"):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise ImportError("pip install paho-mqtt for Home Assistant integration") from e
        self._mqtt = mqtt.Client()
        if username:
            self._mqtt.username_pw_set(username, password)
        self._mqtt.connect(host, port)
        self._mqtt.loop_start()
        self.prefix = discovery_prefix

    def publish_discovery(self, device_name: str) -> None:
        """Announce HA entities for a device."""
        slug = _slug(device_name)
        entities = {
            f"binary_sensor/{slug}_occupancy/config": {
                "name": f"Thoth {device_name} Occupancy",
                "uniq_id": f"thoth_{slug}_occupancy",
                "stat_t": f"thoth/{slug}/occupancy",
                "dev_cla": "occupancy",
                "pl_on": "ON", "pl_off": "OFF",
            },
            f"sensor/{slug}_activity/config": {
                "name": f"Thoth {device_name} Activity",
                "uniq_id": f"thoth_{slug}_activity",
                "stat_t": f"thoth/{slug}/activity",
            },
            f"sensor/{slug}_people_count/config": {
                "name": f"Thoth {device_name} People Count",
                "uniq_id": f"thoth_{slug}_people_count",
                "stat_t": f"thoth/{slug}/people_count",
            },
            f"sensor/{slug}_confidence/config": {
                "name": f"Thoth {device_name} Confidence",
                "uniq_id": f"thoth_{slug}_confidence",
                "stat_t": f"thoth/{slug}/confidence",
                "unit_of_meas": "%",
            },
        }
        for topic, cfg in entities.items():
            self._mqtt.publish(f"{self.prefix}/{topic}", json.dumps(cfg), retain=True)

    def publish_state(self, device_name: str, prediction: dict) -> None:
        """Publish one prediction update for a device."""
        slug = _slug(device_name)
        label = str(prediction.get("label") or prediction.get("class") or "").lower()
        occupied = label not in {"", "empty", "unoccupied", "none"}
        self._mqtt.publish(f"thoth/{slug}/occupancy", "ON" if occupied else "OFF", retain=True)
        if "label" in prediction or "class" in prediction:
            self._mqtt.publish(f"thoth/{slug}/activity", label or "unknown", retain=True)
        if prediction.get("people_count") is not None:
            self._mqtt.publish(f"thoth/{slug}/people_count", str(prediction["people_count"]), retain=True)
        if prediction.get("confidence") is not None:
            conf = prediction["confidence"]
            pct = round(conf * 100, 1) if conf <= 1 else round(conf, 1)
            self._mqtt.publish(f"thoth/{slug}/confidence", str(pct), retain=True)

    def close(self) -> None:
        self._mqtt.loop_stop()
        self._mqtt.disconnect()
