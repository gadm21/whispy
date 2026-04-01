# Exploration: Home Assistant Integration & Watchdog System

> **Context:** Whispy runs on a Raspberry Pi connected to one or more ESP32 CSI collectors via USB-serial. The Pi runs `whispy deploy` (live inference) or `whispy collect` (data recording) 24/7. This document explores the two extensions that turn Whispy from a research CLI into a production-grade smart-home sensing platform.

---

## Part 1 — Home Assistant Integration via MQTT

### 1.1 Why MQTT (not REST, not WebSocket)

| Approach | Pros | Cons |
|----------|------|------|
| **MQTT** | HA native, auto-discovery, retained state, LWT availability, pub/sub decoupling, ~200 KB lib | Requires broker (Mosquitto) |
| REST API | Simple, no broker | Polling-based, no push, no auto-discovery, HA long-lived tokens |
| WebSocket | Real-time | Complex, no HA native discovery, connection management |

**Verdict:** MQTT is the only option with zero-config auto-discovery in Home Assistant. The Mosquitto add-on is pre-installed in most HA setups. MQTT is also the standard for every IoT device (Zigbee2MQTT, Tasmota, ESPHome all use it).

### 1.2 Library Options

| Library | Stars | Approach | Verdict |
|---------|-------|----------|---------|
| **`ha-mqtt-discoverable`** | ~700 | High-level Python classes for every HA entity type. Handles discovery payloads, availability, LWT automatically | **Best choice** — purpose-built, actively maintained, clean API |
| **`paho-mqtt`** (raw) | ~2k | Low-level MQTT client. Must manually construct JSON discovery payloads | Good as fallback, more control, but verbose |
| **`mqtt-homeassistant-utils`** | ~30 | Dataclass-based discovery payload generator | Too niche, small community |

**Recommendation:** Use `ha-mqtt-discoverable` as the primary interface. It wraps `paho-mqtt` internally and handles:
- Discovery topic format: `homeassistant/<component>/<node_id>/<object_id>/config`
- Retained config messages (survives HA restarts)
- LWT (Last Will and Testament) for automatic "unavailable" on disconnect
- Device grouping (all Whispy entities under one device card in HA)

### 1.3 Entity Mapping — What Whispy Exposes to Home Assistant

A single Whispy Pi node maps to **one HA Device** with multiple entities:

```
Device: "Whispy Office Sensor"
├── binary_sensor.whispy_office_occupancy      # ON/OFF (from model prediction)
├── sensor.whispy_office_activity              # "walking", "sitting", etc. (HAR)
├── sensor.whispy_office_location              # "desk", "door", etc. (localization)
├── sensor.whispy_office_confidence            # 0.0–1.0 (model confidence)
├── sensor.whispy_office_csi_rate              # packets/sec from ESP32
├── sensor.whispy_office_cpu_temp              # Pi CPU temperature
├── sensor.whispy_office_uptime                # hours since last restart
├── binary_sensor.whispy_office_esp32_alive    # ESP32 serial health
├── button.whispy_office_recalibrate           # trigger recalibration from HA
└── select.whispy_office_model                 # switch active model from HA
```

#### Entity details:

**a) Occupancy (binary_sensor, device_class=occupancy)**
- State topic: `whispy/<node_id>/occupancy/state` → `ON` / `OFF`
- Derived from: `model.predict()` in deploy mode, thresholded
- HA uses this for: lighting automation, HVAC, alarm arming

**b) Activity (sensor, no device_class)**
- State topic: `whispy/<node_id>/activity/state` → `"walking"`, `"sitting"`, `"standing"`, `"empty"`
- Attributes: `{"confidence": 0.94, "all_probs": {"walking": 0.94, "sitting": 0.03, ...}}`
- Published as JSON, extracted with `value_template: {{ value_json.activity }}`

**c) Confidence (sensor, device_class=None, unit=%)**
- State topic: `whispy/<node_id>/confidence/state` → `94.2`
- Useful for HA automations: "if confidence < 70% for 5 minutes, send alert"

**d) CSI Rate (sensor, device_class=None, unit=Hz)**
- State topic: `whispy/<node_id>/diagnostics/csi_rate` → `148`
- Diagnostic entity — detects ESP32 performance issues

**e) CPU Temperature (sensor, device_class=temperature, unit=°C)**
- State topic: `whispy/<node_id>/diagnostics/cpu_temp` → `52.3`
- Read via `gpiozero.CPUTemperature` or `/sys/class/thermal/thermal_zone0/temp`

**f) ESP32 Alive (binary_sensor, device_class=connectivity)**
- State topic: `whispy/<node_id>/diagnostics/esp32_alive` → `ON` / `OFF`
- Based on: CSI rate > 0 in last 5 seconds

**g) Recalibrate Button (button)**
- Command topic: `whispy/<node_id>/recalibrate/set`
- When pressed in HA → Whispy collects 60s of labeled data for model fine-tuning

**h) Model Select (select)**
- Command topic: `whispy/<node_id>/model/set`
- Options: list of `.pkl` files in model directory
- Allows switching between HAR/occupancy/localization models from HA UI

### 1.4 MQTT Discovery — Exact Payloads

Using **device discovery** (single message, all components):

```python
DISCOVERY_TOPIC = "homeassistant/device/whispy_{node_id}/config"

DISCOVERY_PAYLOAD = {
    "dev": {
        "ids": f"whispy_{node_id}",
        "name": f"Whispy {location}",
        "mf": "Whispy",
        "mdl": "ESP32-C6 CSI Sensor",
        "sw": whispy.__version__,
        "sa": location,            # suggested_area → auto-assigns room in HA
    },
    "o": {
        "name": "whispy",
        "sw": whispy.__version__,
        "url": "https://github.com/gadm21/whispy",
    },
    "cmps": {
        "occupancy": {
            "p": "binary_sensor",
            "device_class": "occupancy",
            "state_topic": f"whispy/{node_id}/occupancy/state",
            "payload_on": "ON",
            "payload_off": "OFF",
            "unique_id": f"whispy_{node_id}_occupancy",
        },
        "activity": {
            "p": "sensor",
            "state_topic": f"whispy/{node_id}/activity/state",
            "value_template": "{{ value_json.activity }}",
            "json_attributes_topic": f"whispy/{node_id}/activity/state",
            "unique_id": f"whispy_{node_id}_activity",
        },
        "confidence": {
            "p": "sensor",
            "unit_of_measurement": "%",
            "state_topic": f"whispy/{node_id}/confidence/state",
            "unique_id": f"whispy_{node_id}_confidence",
        },
        "csi_rate": {
            "p": "sensor",
            "unit_of_measurement": "Hz",
            "state_topic": f"whispy/{node_id}/diagnostics/csi_rate",
            "unique_id": f"whispy_{node_id}_csi_rate",
            "entity_category": "diagnostic",
        },
        "cpu_temp": {
            "p": "sensor",
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_topic": f"whispy/{node_id}/diagnostics/cpu_temp",
            "unique_id": f"whispy_{node_id}_cpu_temp",
            "entity_category": "diagnostic",
        },
        "esp32_alive": {
            "p": "binary_sensor",
            "device_class": "connectivity",
            "state_topic": f"whispy/{node_id}/diagnostics/esp32_alive",
            "unique_id": f"whispy_{node_id}_esp32_alive",
            "entity_category": "diagnostic",
        },
    },
    "availability": {
        "topic": f"whispy/{node_id}/availability",
        "payload_available": "online",
        "payload_not_available": "offline",
    },
}
```

### 1.5 Architecture — Where MQTT Fits in Whispy

```
                    ┌─────────────────────────────────────────┐
                    │              Raspberry Pi                 │
                    │                                           │
  ESP32 ──serial──▶│  collect.py / deploy()                   │
                    │       │                                   │
                    │       ▼                                   │
                    │  _CSIBuffer → model.predict()            │
                    │       │                                   │
                    │       ▼                                   │
                    │  whispy.mqtt.Publisher ──MQTT──▶ Mosquitto│──▶ Home Assistant
                    │       │                                   │
                    │       ▼                                   │
                    │  whispy.watchdog (monitors everything)    │
                    └─────────────────────────────────────────┘
```

The `Publisher` runs as a **background thread** inside the deploy loop. Every N seconds (configurable, default 2s), it:
1. Reads latest prediction from `_CSIBuffer.model_pred`
2. Reads CSI rate from `_CSIBuffer.count` delta
3. Reads Pi diagnostics (CPU temp, RAM, disk)
4. Publishes all topics
5. Sends `WATCHDOG=1` to systemd (if running as service)

### 1.6 Implementation Plan — `whispy/mqtt.py`

```python
# whispy/mqtt.py — MQTT publisher with HA auto-discovery

class WhispyMQTT:
    """Publishes Whispy state to MQTT with Home Assistant auto-discovery."""

    def __init__(self, broker: str, port: int = 1883,
                 node_id: str = "office", location: str = "Office",
                 username: str | None = None, password: str | None = None,
                 task: str = "occupancy"):  # occupancy | har | localization
        ...

    def connect(self) -> None:
        """Connect to broker, publish discovery config (retained), set LWT."""
        ...

    def publish_prediction(self, pred: int | str, confidence: float,
                           all_probs: dict[str, float] | None = None) -> None:
        """Publish model prediction to state topics."""
        ...

    def publish_diagnostics(self, csi_rate: float, cpu_temp: float,
                            esp32_alive: bool, uptime_hours: float) -> None:
        """Publish diagnostic entities."""
        ...

    def disconnect(self) -> None:
        """Publish offline availability, disconnect cleanly."""
        ...
```

#### Integration with deploy():
```python
# In collect.py deploy() — add optional MQTT publishing
def deploy(port, model_path, baud=115200, sr=150, var_window=20,
           window_len=500, mqtt_broker=None, mqtt_node="office", ...):
    ...
    if mqtt_broker:
        from whispy.mqtt import WhispyMQTT
        mqtt = WhispyMQTT(broker=mqtt_broker, node_id=mqtt_node)
        mqtt.connect()

    while not _stop.is_set():
        ...
        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0]

        if mqtt_broker:
            mqtt.publish_prediction(pred, prob.max(), ...)
            mqtt.publish_diagnostics(csi_rate, cpu_temp, ...)
    ...
```

#### CLI addition:
```
whispy deploy --port /dev/ttyUSB0 --model model.pkl \
    --mqtt-broker 192.168.1.100 --mqtt-node office --mqtt-location "Office"
```

### 1.7 Home Assistant Automations (examples users can build)

```yaml
# Turn off lights when room unoccupied for 5 minutes
automation:
  - alias: "Auto lights off"
    trigger:
      - platform: state
        entity_id: binary_sensor.whispy_office_occupancy
        to: "off"
        for: "00:05:00"
    action:
      - service: light.turn_off
        target:
          area_id: office

# Alert when ESP32 goes offline
  - alias: "ESP32 offline alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.whispy_office_esp32_alive
        to: "off"
        for: "00:01:00"
    action:
      - service: notify.mobile_app
        data:
          message: "Whispy ESP32 in office is offline!"

# HVAC eco mode when no one home
  - alias: "HVAC eco when empty"
    trigger:
      - platform: state
        entity_id: binary_sensor.whispy_office_occupancy
        to: "off"
        for: "00:15:00"
    action:
      - service: climate.set_preset_mode
        target:
          entity_id: climate.thermostat
        data:
          preset_mode: eco
```

### 1.8 Dependencies

```toml
# pyproject.toml addition
[project.optional-dependencies]
mqtt = ["paho-mqtt>=2.0", "ha-mqtt-discoverable>=0.13"]
```

Alternatively, for minimal footprint (no ha-mqtt-discoverable):
```toml
mqtt = ["paho-mqtt>=2.0"]  # ~200 KB, manual discovery payloads
```

---

## Part 2 — Watchdog & System Health

### 2.1 Failure Modes on Raspberry Pi

| Failure | Frequency | Impact | Detection |
|---------|-----------|--------|-----------|
| **USB-serial disconnect** | ~1/week | Collection stops silently | Serial read returns 0 bytes for >5s |
| **ESP32 firmware crash** | ~1/2 weeks | CSI_DATA lines stop but serial still open | CSI rate drops to 0 |
| **Pi SD card full** | Once (fatal) | Writes fail, filesystem corrupts | `shutil.disk_usage()` |
| **Pi CPU throttle** | Hot days | Processing slows, windows missed | CPU temp > 80°C |
| **Python process crash** | Rare | Everything stops | systemd detects exit |
| **Pi kernel panic / hang** | Very rare | Everything stops | Hardware watchdog (bcm2835_wdt) |
| **Network loss** | Variable | MQTT disconnects, no alerts | MQTT on_disconnect callback |
| **Power outage** | Variable | Everything stops | systemd auto-start on boot |

### 2.2 Three-Layer Watchdog Architecture

```
Layer 3: Hardware watchdog (bcm2835_wdt)
  └── Reboots Pi if kernel hangs (WatchdogSec in systemd)
  
Layer 2: systemd service supervision
  └── Restarts whispy process if it crashes (Restart=always)
  
Layer 1: Application-level watchdog (whispy.watchdog)
  └── Monitors serial health, CSI rate, disk, CPU temp
  └── Recovers ESP32 via GPIO power cycle
  └── Rotates/compresses old data files
  └── Sends alerts (MQTT, email, webhook)
  └── Sends sd_notify WATCHDOG=1 heartbeat to systemd
```

### 2.3 Layer 2 — systemd Service

```ini
# /etc/systemd/system/whispy.service
[Unit]
Description=Whispy WiFi CSI Sensing Service
After=network-online.target mosquitto.service
Wants=network-online.target

[Service]
Type=notify
User=pi
WorkingDirectory=/home/pi
ExecStart=/home/pi/.venv/bin/whispy deploy \
    --port /dev/ttyUSB0 \
    --model /home/pi/models/occupancy.pkl \
    --mqtt-broker localhost \
    --mqtt-node office \
    --watchdog

# Process supervision
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Watchdog — if no WATCHDOG=1 in 30s, systemd kills & restarts
WatchdogSec=30

# Resource limits
MemoryMax=512M
CPUQuota=80%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=whispy

[Install]
WantedBy=multi-user.target
```

#### Key systemd features:
- **`Type=notify`** — Whispy calls `sd_notify.ready()` after model loads, so systemd knows startup succeeded
- **`WatchdogSec=30`** — Whispy must send `WATCHDOG=1` every <30s or systemd force-restarts it
- **`Restart=always` + `RestartSec=10`** — Auto-restart on any exit (crash, watchdog kill, OOM)
- **`StartLimitBurst=5` / `StartLimitIntervalSec=300`** — Max 5 restarts in 5 minutes, then give up (prevents restart storm)
- **`MemoryMax=512M`** — OOM-kill before Pi becomes unresponsive

### 2.4 Layer 1 — Application Watchdog (`whispy/watchdog.py`)

```python
# whispy/watchdog.py

class HealthMonitor:
    """Monitors system and peripheral health on Raspberry Pi."""

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self._sd_notify = None  # sd-notify for systemd heartbeat
        self._alert_backends: list[AlertBackend] = []

    # ── Checks ──────────────────────────────────────────────
    def check_csi_rate(self, current_rate: float) -> HealthStatus:
        """CRITICAL if rate=0 for >10s, WARNING if <50 Hz."""
        ...

    def check_disk_usage(self) -> HealthStatus:
        """CRITICAL if >95%, WARNING if >85%. Auto-rotate if critical."""
        ...

    def check_cpu_temperature(self) -> HealthStatus:
        """CRITICAL if >80°C, WARNING if >70°C."""
        ...

    def check_memory(self) -> HealthStatus:
        """WARNING if >80% RAM used."""
        ...

    def check_serial_health(self, serial_errors: int) -> HealthStatus:
        """CRITICAL if serial read errors > threshold."""
        ...

    # ── Recovery Actions ────────────────────────────────────
    def power_cycle_esp32(self, gpio_pin: int = 17) -> bool:
        """Toggle GPIO pin to cut/restore ESP32 USB power via relay/MOSFET.

        Wiring: Pi GPIO17 → NPN base → relay coil → ESP32 5V
        Sequence: HIGH (cut power) → wait 3s → LOW (restore) → wait 5s
        """
        ...

    def rotate_data_files(self, data_dir: str, max_gb: float = 4.0) -> int:
        """Delete oldest .npz files when total size exceeds max_gb.
        Returns number of files deleted."""
        ...

    def compress_old_files(self, data_dir: str, older_than_hours: int = 24) -> int:
        """gzip .npz files older than threshold. Returns count."""
        ...

    # ── Heartbeat ───────────────────────────────────────────
    def heartbeat(self) -> None:
        """Send WATCHDOG=1 to systemd. Call this every ~15s."""
        if self._sd_notify and self._sd_notify.enabled():
            self._sd_notify.notify("WATCHDOG=1")

    def report_ready(self) -> None:
        """Tell systemd that initialization is complete."""
        if self._sd_notify and self._sd_notify.enabled():
            self._sd_notify.ready()

    def report_status(self, msg: str) -> None:
        """Update systemd status line (visible in systemctl status)."""
        if self._sd_notify and self._sd_notify.enabled():
            self._sd_notify.status(msg)

    # ── Main Loop ───────────────────────────────────────────
    def run_checks(self, csi_rate: float, serial_errors: int) -> list[HealthStatus]:
        """Run all checks, take recovery actions, send alerts, heartbeat."""
        results = [
            self.check_csi_rate(csi_rate),
            self.check_disk_usage(),
            self.check_cpu_temperature(),
            self.check_memory(),
            self.check_serial_health(serial_errors),
        ]

        # Auto-recovery
        for r in results:
            if r.level == "CRITICAL" and r.check == "csi_rate":
                if self.config.gpio_reset_pin:
                    self.power_cycle_esp32(self.config.gpio_reset_pin)

            if r.level == "CRITICAL" and r.check == "disk_usage":
                self.rotate_data_files(self.config.data_dir)

        # Alerts
        critical = [r for r in results if r.level == "CRITICAL"]
        if critical:
            for backend in self._alert_backends:
                backend.send(critical)

        # systemd heartbeat
        self.heartbeat()

        return results
```

### 2.5 Configuration

```python
@dataclass
class WatchdogConfig:
    # Thresholds
    csi_rate_warning: float = 50.0       # Hz
    csi_rate_critical_timeout: float = 10.0  # seconds of 0 rate
    disk_warning_pct: float = 85.0
    disk_critical_pct: float = 95.0
    cpu_temp_warning: float = 70.0       # °C
    cpu_temp_critical: float = 80.0      # °C
    memory_warning_pct: float = 80.0

    # Recovery
    gpio_reset_pin: int | None = 17      # None = disabled
    gpio_reset_duration: float = 3.0     # seconds power off
    data_dir: str = "./whispy_data"
    max_data_gb: float = 4.0             # auto-rotate threshold

    # Alerts
    mqtt_alerts: bool = True             # publish to whispy/{node}/alert
    email_to: str | None = None
    webhook_url: str | None = None       # Slack/Discord/Telegram

    # systemd
    heartbeat_interval: float = 15.0     # seconds (< WatchdogSec/2)
```

### 2.6 GPIO Power-Cycle Circuit

```
Raspberry Pi                    ESP32 USB power
─────────────                   ─────────────────
GPIO 17 ──[1kΩ]──┐
                  │
              [NPN 2N2222]
                  │  C ──── [Relay/MOSFET] ──── ESP32 5V VIN
                  │  E ──── GND
                  B
```

- **Normally closed relay**: ESP32 powered by default. GPIO HIGH → relay opens → power cut
- **N-channel MOSFET** (e.g., IRLZ44N): simpler, no relay click, same logic
- Cost: ~$2 for relay module, or $1 for MOSFET
- `gpiozero.OutputDevice(17)` — simple Python API, no root needed with `/dev/gpiomem`

### 2.7 Alert Backends

```python
class AlertBackend(ABC):
    @abstractmethod
    def send(self, alerts: list[HealthStatus]) -> None: ...

class MQTTAlert(AlertBackend):
    """Publish to whispy/{node_id}/alert topic."""
    def send(self, alerts):
        payload = json.dumps([{"check": a.check, "level": a.level,
                               "message": a.message} for a in alerts])
        self.client.publish(f"whispy/{self.node_id}/alert", payload)

class EmailAlert(AlertBackend):
    """Send via smtplib (Gmail app password or local SMTP)."""
    ...

class WebhookAlert(AlertBackend):
    """POST to Slack/Discord/Telegram webhook URL."""
    def send(self, alerts):
        msg = "\n".join(f"⚠️ {a.check}: {a.message}" for a in alerts)
        requests.post(self.url, json={"text": msg}, timeout=5)
```

### 2.8 CLI Integration

```
# Install as systemd service
whispy watchdog install \
    --port /dev/ttyUSB0 \
    --model /home/pi/models/occupancy.pkl \
    --mqtt-broker localhost \
    --mqtt-node office \
    --gpio-pin 17 \
    --email alerts@example.com

# This generates /etc/systemd/system/whispy.service and enables it

# Check health manually
whispy watchdog status
# Output:
#   CSI Rate:     148 Hz     ✓
#   CPU Temp:     52.3°C     ✓
#   Disk Usage:   34%        ✓
#   Memory:       45%        ✓
#   ESP32:        alive      ✓
#   Uptime:       14.3 hours
#   Last alert:   none

# View logs
whispy watchdog logs --lines 50
# (wraps: journalctl -u whispy --lines 50)

# Uninstall service
whispy watchdog uninstall
```

### 2.9 Integration with Deploy Loop

```python
def deploy(port, model_path, ..., watchdog=False, gpio_pin=None, ...):
    ...
    health = None
    if watchdog:
        from whispy.watchdog import HealthMonitor, WatchdogConfig
        config = WatchdogConfig(gpio_reset_pin=gpio_pin)
        health = HealthMonitor(config)
        health.report_ready()   # tell systemd we're up
        health.report_status("Model loaded, listening for CSI")

    last_health_check = time.time()

    while not _stop.is_set():
        ...
        pred = model.predict(x)[0]
        ...

        # Health check every 15s
        if health and time.time() - last_health_check > 15:
            csi_rate = (buf.count - last_count) / (time.time() - last_time)
            health.run_checks(csi_rate, len(buf.errors))
            health.report_status(f"pred={pred} rate={csi_rate:.0f}Hz")
            last_health_check = time.time()
```

### 2.10 Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
watchdog = ["sd-notify>=0.1", "psutil>=5.9"]
pi = ["gpiozero>=2.0", "RPi.GPIO>=0.7"]  # Pi-specific GPIO
```

---

## Part 3 — Combined Architecture

When both extensions are active, the full system looks like:

```
┌──────────────────────────────────────────────────────────────────┐
│  systemd (Layer 2)                                                │
│  ├── Restart=always, WatchdogSec=30                              │
│  └── manages whispy.service                                       │
│       │                                                           │
│       ▼                                                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  whispy deploy --watchdog --mqtt-broker localhost           │  │
│  │                                                             │  │
│  │  ┌─────────┐    ┌──────────┐    ┌──────────────────────┐  │  │
│  │  │ Serial  │───▶│CSIBuffer │───▶│ model.predict()      │  │  │
│  │  │ Reader  │    │          │    │                       │  │  │
│  │  │ Thread  │    └──────────┘    └───────┬──────────────┘  │  │
│  │  └─────────┘                            │                  │  │
│  │                                         ▼                  │  │
│  │                               ┌──────────────────┐        │  │
│  │                               │ WhispyMQTT       │        │  │
│  │                               │ .publish_pred()   │──MQTT──┼──┼──▶ Mosquitto ──▶ HA
│  │                               │ .publish_diag()   │        │  │
│  │                               └──────────────────┘        │  │
│  │                                         ▲                  │  │
│  │                               ┌──────────────────┐        │  │
│  │                               │ HealthMonitor     │        │  │
│  │                               │ .run_checks()     │        │  │
│  │                               │ .heartbeat()  ────┼──sd_notify──▶ systemd
│  │                               │ .power_cycle() ───┼──GPIO──┼──▶ ESP32 relay
│  │                               └──────────────────┘        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  bcm2835_wdt (Layer 3) — reboots Pi if kernel hangs              │
└──────────────────────────────────────────────────────────────────┘
```

## Part 4 — Implementation Priority

| Step | Effort | Impact | Order |
|------|--------|--------|-------|
| `whispy/mqtt.py` — basic publisher (paho-mqtt, manual discovery) | 2h | High | 1 |
| Integrate MQTT into `deploy()` + CLI flags | 1h | High | 2 |
| `whispy/watchdog.py` — health checks, sd_notify | 3h | High | 3 |
| systemd service template + `whispy watchdog install` CLI | 2h | High | 4 |
| GPIO power-cycle recovery | 1h | Medium | 5 |
| `ha-mqtt-discoverable` upgrade (cleaner code, more entities) | 2h | Medium | 6 |
| Alert backends (email, webhook) | 1h | Low | 7 |
| HA automation example configs | 0.5h | Low | 8 |
| Data rotation + compression | 1h | Medium | 9 |
| Recalibrate button + model select (bidirectional MQTT) | 2h | Medium | 10 |

**Total estimated effort: ~15 hours**

---

## Part 5 — Key Design Decisions to Make

1. **`ha-mqtt-discoverable` vs raw `paho-mqtt`?**
   - `ha-mqtt-discoverable` is cleaner but adds a dependency (~5 MB). For a Pi with pip, this is fine.
   - Raw `paho-mqtt` is lighter (~200 KB) but requires ~100 lines of manual discovery JSON.
   - **Recommendation:** Start with raw `paho-mqtt` for minimal deps, upgrade later if needed.

2. **MQTT QoS level?**
   - QoS 0 (fire and forget) for high-frequency state updates (predictions, CSI rate)
   - QoS 1 (at least once) for discovery configs and alerts
   - QoS 2 unnecessary — CSI sensing is tolerant of occasional missed updates

3. **Publish frequency?**
   - Predictions: every window (~3.3s at 500-sample/150Hz window)
   - Diagnostics: every 15s
   - Availability: on connect + LWT on disconnect

4. **Node ID scheme?**
   - Default: hostname-based (`pi-office`, `pi-bedroom`)
   - User-configurable via `--mqtt-node`
   - Must be `[a-zA-Z0-9_-]` for MQTT topic compatibility

5. **GPIO pin for ESP32 reset — configurable or fixed?**
   - Configurable via `--gpio-pin`. Default 17 (common choice, no conflict with SPI/I2C).
   - `None` = disable GPIO reset (for non-Pi hosts or no relay wired)
