"""MQTT publisher with Home Assistant auto-discovery.

Publishes CSI predictions, diagnostics, and device availability to an MQTT
broker.  Home Assistant discovers Whispy as a device with multiple entities
automatically — no YAML configuration needed on the HA side.

Usage
-----
    from whispy.mqtt import WhispyMQTT

    mqtt = WhispyMQTT(broker="192.168.1.100", node_id="office")
    mqtt.connect()
    mqtt.publish_prediction(pred=1, confidence=0.94, label="occupied")
    mqtt.publish_diagnostics(csi_rate=148, cpu_temp=52.3, esp32_alive=True)
    mqtt.disconnect()

CLI testing::

    whispy mqtt test --broker 192.168.1.100 --node office
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any


# =========================================================================
# Topic helpers
# =========================================================================

def _topic(node_id: str, *parts: str) -> str:
    return "/".join(["whispy", node_id] + list(parts))


def _discovery_topic(component: str, node_id: str, object_id: str) -> str:
    """HA MQTT discovery topic format."""
    return f"homeassistant/{component}/whispy_{node_id}/{object_id}/config"


# =========================================================================
# Main publisher
# =========================================================================

class WhispyMQTT:
    """MQTT publisher with Home Assistant auto-discovery.

    Parameters
    ----------
    broker : MQTT broker hostname or IP.
    port : MQTT broker port (default 1883, use 8883 for TLS).
    node_id : unique identifier for this Whispy node (e.g. "lab-toronto-01").
    location : human-friendly location name (e.g. "Office").
    latitude, longitude : GPS coordinates for the device.
    username, password : broker credentials (optional).
    tls : enable TLS encryption (required for cloud brokers).
    task : sensing task — "occupancy", "har", or "localization".
    labels : class label names for the task (optional).
    backend_url : REST API URL for the central backend (optional).
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        node_id: str = "office",
        location: str = "Office",
        latitude: float = 0.0,
        longitude: float = 0.0,
        username: str | None = None,
        password: str | None = None,
        tls: bool = False,
        task: str = "occupancy",
        labels: list[str] | None = None,
        backend_url: str | None = None,
    ):
        self.broker = broker
        self.port = port
        self.node_id = node_id
        self.location = location
        self.latitude = latitude
        self.longitude = longitude
        self.username = username
        self.password = password
        self.tls = tls
        self.task = task
        self.labels = labels or []
        self.backend_url = backend_url
        self._client: Any = None
        self._connected = False

    # ── connection ──────────────────────────────────────────
    def connect(self) -> bool:
        """Connect to MQTT broker, publish discovery configs, set LWT."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("[mqtt] ERROR: paho-mqtt required — pip install whispy[mqtt]")
            return False

        client_id = f"whispy_{self.node_id}_{int(time.time())}"
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        if self.username:
            self._client.username_pw_set(self.username, self.password)

        # TLS for cloud broker connections
        if self.tls:
            import ssl
            self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED,
                                 tls_version=ssl.PROTOCOL_TLSv1_2)

        # LWT — broker publishes "offline" if we disconnect unexpectedly
        avail_topic = _topic(self.node_id, "availability")
        self._client.will_set(avail_topic, "offline", qos=1, retain=True)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
            # wait for connection
            for _ in range(50):  # 5 seconds
                if self._connected:
                    break
                time.sleep(0.1)
            if not self._connected:
                print(f"[mqtt] WARNING: Connection to {self.broker}:{self.port} timed out")
                return False
        except Exception as e:
            print(f"[mqtt] ERROR: {e}")
            return False

        # publish availability
        self._client.publish(avail_topic, "online", qos=1, retain=True)

        # publish HA discovery configs
        self._publish_discovery()

        # publish device info for backend registration
        self._publish_device_info()

        # also register via REST if backend_url is set
        if self.backend_url:
            self._register_rest()

        print(f"[mqtt] Connected to {self.broker}:{self.port}  node={self.node_id}")
        return True

    def disconnect(self) -> None:
        """Publish offline status and disconnect cleanly."""
        if self._client:
            avail_topic = _topic(self.node_id, "availability")
            self._client.publish(avail_topic, "offline", qos=1, retain=True)
            time.sleep(0.2)
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            print("[mqtt] Disconnected")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self._connected = True

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ── discovery ───────────────────────────────────────────
    def _publish_discovery(self) -> None:
        """Publish HA MQTT auto-discovery config for all entities."""
        import whispy

        device = {
            "ids": [f"whispy_{self.node_id}"],
            "name": f"Whispy {self.location}",
            "mf": "Whispy",
            "mdl": "ESP32-C6 CSI Sensor",
            "sw": whispy.__version__,
            "sa": self.location,
        }

        origin = {
            "name": "whispy",
            "sw": whispy.__version__,
            "url": "https://github.com/gadm21/whispy",
        }

        avail = {
            "topic": _topic(self.node_id, "availability"),
            "payload_available": "online",
            "payload_not_available": "offline",
        }

        # --- Occupancy binary sensor ---
        if self.task == "occupancy":
            self._publish_config(
                "binary_sensor", "occupancy", {
                    "name": None,  # inherit device name
                    "device_class": "occupancy",
                    "state_topic": _topic(self.node_id, "occupancy", "state"),
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "unique_id": f"whispy_{self.node_id}_occupancy",
                    "device": device,
                    "origin": origin,
                    "availability": avail,
                })

        # --- Activity / prediction sensor ---
        self._publish_config(
            "sensor", "prediction", {
                "name": "Prediction",
                "icon": "mdi:wifi",
                "state_topic": _topic(self.node_id, "prediction", "state"),
                "value_template": "{{ value_json.label }}",
                "json_attributes_topic": _topic(self.node_id, "prediction", "state"),
                "unique_id": f"whispy_{self.node_id}_prediction",
                "device": device,
                "origin": origin,
                "availability": avail,
            })

        # --- Confidence sensor ---
        self._publish_config(
            "sensor", "confidence", {
                "name": "Confidence",
                "icon": "mdi:gauge",
                "unit_of_measurement": "%",
                "state_topic": _topic(self.node_id, "confidence", "state"),
                "unique_id": f"whispy_{self.node_id}_confidence",
                "device": device,
                "origin": origin,
                "availability": avail,
            })

        # --- CSI rate diagnostic sensor ---
        self._publish_config(
            "sensor", "csi_rate", {
                "name": "CSI Rate",
                "icon": "mdi:speedometer",
                "unit_of_measurement": "Hz",
                "state_topic": _topic(self.node_id, "diagnostics", "csi_rate"),
                "unique_id": f"whispy_{self.node_id}_csi_rate",
                "entity_category": "diagnostic",
                "device": device,
                "origin": origin,
                "availability": avail,
            })

        # --- CPU temperature diagnostic sensor ---
        self._publish_config(
            "sensor", "cpu_temp", {
                "name": "CPU Temperature",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
                "state_topic": _topic(self.node_id, "diagnostics", "cpu_temp"),
                "unique_id": f"whispy_{self.node_id}_cpu_temp",
                "entity_category": "diagnostic",
                "device": device,
                "origin": origin,
                "availability": avail,
            })

        # --- ESP32 connectivity diagnostic ---
        self._publish_config(
            "binary_sensor", "esp32_alive", {
                "name": "ESP32",
                "device_class": "connectivity",
                "state_topic": _topic(self.node_id, "diagnostics", "esp32_alive"),
                "payload_on": "ON",
                "payload_off": "OFF",
                "unique_id": f"whispy_{self.node_id}_esp32_alive",
                "entity_category": "diagnostic",
                "device": device,
                "origin": origin,
                "availability": avail,
            })

        # --- Cache usage diagnostic sensor ---
        self._publish_config(
            "sensor", "cache_usage", {
                "name": "Cache Usage",
                "icon": "mdi:database",
                "unit_of_measurement": "MB",
                "state_topic": _topic(self.node_id, "diagnostics", "cache_mb"),
                "unique_id": f"whispy_{self.node_id}_cache_usage",
                "entity_category": "diagnostic",
                "device": device,
                "origin": origin,
                "availability": avail,
            })

    def _publish_config(self, component: str, object_id: str,
                        payload: dict) -> None:
        topic = _discovery_topic(component, self.node_id, object_id)
        self._client.publish(topic, json.dumps(payload), qos=1, retain=True)

    # ── device registration ─────────────────────────────────
    def _publish_device_info(self) -> None:
        """Publish device info so the central backend can register this node."""
        from whispy.device import DeviceInfo
        dev = DeviceInfo.from_system(
            node_id=self.node_id,
            location=self.location,
            latitude=self.latitude,
            longitude=self.longitude,
            task=self.task,
        )
        dev.labels = self.labels
        topic = _topic(self.node_id, "device", "info")
        self._client.publish(topic, dev.to_json(), qos=1, retain=True)

    def _register_rest(self) -> None:
        """Register device with the central backend via REST API."""
        try:
            import urllib.request
            from whispy.device import DeviceInfo
            dev = DeviceInfo.from_system(
                node_id=self.node_id,
                location=self.location,
                latitude=self.latitude,
                longitude=self.longitude,
                task=self.task,
            )
            dev.labels = self.labels
            url = f"{self.backend_url.rstrip('/')}/devices/register"
            data = dev.to_json().encode("utf-8")
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[mqtt] REST registration: {resp.read().decode()}")
        except Exception as e:
            print(f"[mqtt] REST registration failed (non-fatal): {e}")

    # ── state publishing ────────────────────────────────────
    def publish_prediction(
        self,
        pred: int | str,
        confidence: float,
        label: str | None = None,
        all_probs: dict[str, float] | None = None,
    ) -> None:
        """Publish model prediction and confidence."""
        if not self._connected:
            return

        label_str = label or str(pred)

        # prediction topic (JSON with attributes)
        pred_payload = {
            "label": label_str,
            "class": int(pred) if isinstance(pred, (int, float, np.integer)) else pred,
            "confidence": round(confidence * 100, 1),
        }
        if all_probs:
            pred_payload["probabilities"] = {
                k: round(v, 4) for k, v in all_probs.items()
            }
        self._client.publish(
            _topic(self.node_id, "prediction", "state"),
            json.dumps(pred_payload), qos=0,
        )

        # confidence as standalone value
        self._client.publish(
            _topic(self.node_id, "confidence", "state"),
            str(round(confidence * 100, 1)), qos=0,
        )

        # occupancy binary sensor
        if self.task == "occupancy":
            is_occupied = bool(pred) if isinstance(pred, (int, float)) else label_str.lower() not in ("empty", "off", "0", "none")
            self._client.publish(
                _topic(self.node_id, "occupancy", "state"),
                "ON" if is_occupied else "OFF", qos=0,
            )

    def publish_diagnostics(
        self,
        csi_rate: float = 0,
        cpu_temp: float | None = None,
        esp32_alive: bool = True,
        cache_mb: float = 0,
    ) -> None:
        """Publish diagnostic entities."""
        if not self._connected:
            return

        self._client.publish(
            _topic(self.node_id, "diagnostics", "csi_rate"),
            str(round(csi_rate, 1)), qos=0,
        )
        if cpu_temp is not None:
            self._client.publish(
                _topic(self.node_id, "diagnostics", "cpu_temp"),
                str(round(cpu_temp, 1)), qos=0,
            )
        self._client.publish(
            _topic(self.node_id, "diagnostics", "esp32_alive"),
            "ON" if esp32_alive else "OFF", qos=0,
        )
        self._client.publish(
            _topic(self.node_id, "diagnostics", "cache_mb"),
            str(round(cache_mb, 1)), qos=0,
        )

    # ── convenience ─────────────────────────────────────────
    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"WhispyMQTT({self.broker}:{self.port}, node={self.node_id}, {status})"


# we need numpy for the np.integer check in publish_prediction
import numpy as np


# =========================================================================
# Connection test
# =========================================================================

def test_connection(
    broker: str = "localhost",
    port: int = 1883,
    node_id: str = "test",
    location: str = "Test Room",
    username: str | None = None,
    password: str | None = None,
    verbose: bool = True,
) -> bool:
    """Test MQTT connection and HA discovery end-to-end.

    Connects to the broker, publishes discovery configs, sends a few
    test predictions and diagnostics, then verifies the messages appear.
    Returns True if all steps succeed.

    Usage::

        whispy mqtt test --broker 192.168.1.100
    """
    results: list[tuple[str, bool, str]] = []

    def log(step: str, ok: bool, msg: str = ""):
        results.append((step, ok, msg))
        if verbose:
            icon = "✓" if ok else "✗"
            print(f"  {icon} {step}: {msg}")

    # Step 1: import paho-mqtt
    try:
        import paho.mqtt.client as mqtt
        log("Import paho-mqtt", True, "OK")
    except ImportError:
        log("Import paho-mqtt", False, "pip install whispy[mqtt]")
        return False

    # Step 2: connect to broker
    m = WhispyMQTT(
        broker=broker, port=port, node_id=node_id,
        location=location, username=username, password=password,
        task="occupancy",
    )
    connected = m.connect()
    log("Connect to broker", connected,
        f"{broker}:{port}" if connected else f"Failed to connect to {broker}:{port}")
    if not connected:
        return False

    # Step 3: verify discovery topics were published
    # subscribe to our own discovery topics and check they arrive
    received_topics: list[str] = []
    receive_event = threading.Event()

    def on_message(client, userdata, msg):
        received_topics.append(msg.topic)
        if len(received_topics) >= 3:
            receive_event.set()

    m._client.on_message = on_message
    m._client.subscribe(f"homeassistant/+/whispy_{node_id}/+/config", qos=1)
    receive_event.wait(timeout=3.0)
    n_discovery = len(received_topics)
    log("HA discovery configs", n_discovery > 0,
        f"{n_discovery} entity configs published")

    # Step 4: publish test prediction
    try:
        m.publish_prediction(pred=1, confidence=0.93, label="occupied")
        log("Publish prediction", True, "pred=occupied conf=93%")
    except Exception as e:
        log("Publish prediction", False, str(e))

    # Step 5: publish test diagnostics
    try:
        m.publish_diagnostics(
            csi_rate=148.5, cpu_temp=52.3,
            esp32_alive=True, cache_mb=256.7,
        )
        log("Publish diagnostics", True, "rate=148.5Hz temp=52.3°C cache=256.7MB")
    except Exception as e:
        log("Publish diagnostics", False, str(e))

    # Step 6: verify our state topics are published
    state_topics: list[str] = []
    state_event = threading.Event()

    def on_state(client, userdata, msg):
        state_topics.append(msg.topic)
        if len(state_topics) >= 3:
            state_event.set()

    m._client.on_message = on_state
    m._client.subscribe(f"whispy/{node_id}/#", qos=0)

    # re-publish to trigger
    m.publish_prediction(pred=0, confidence=0.85, label="empty")
    m.publish_diagnostics(csi_rate=150, cpu_temp=48.0, esp32_alive=True, cache_mb=100)
    state_event.wait(timeout=3.0)
    n_state = len(state_topics)
    log("State topics received", n_state > 0,
        f"{n_state} state messages verified")

    # Step 7: clean disconnect
    m.disconnect()
    log("Clean disconnect", True, "LWT offline published")

    time.sleep(0.5)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    if verbose:
        print(f"\n  {'='*40}")
        print(f"  Results: {passed}/{total} checks passed")
        if passed == total:
            print("  Home Assistant integration is working!")
            print(f"  Device will appear as: Whispy {location}")
            print(f"  Topics: whispy/{node_id}/#")
        else:
            failed = [s for s, ok, _ in results if not ok]
            print(f"  Failed: {', '.join(failed)}")
        print()

    return passed == total
