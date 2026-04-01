"""Device registry and receiver discovery for global Whispy deployments.

Each Whispy node (a Raspberry Pi + ESP32 combo) is represented as a
``DeviceInfo`` with a globally unique ``node_id``, geographic location,
hardware attributes, and a list of attached ESP32 receivers.

The registry persists to a JSON file (server side) or is published over
MQTT so the central backend always knows every active device.

Usage — device side
-------------------
    from whispy.device import DeviceInfo, ReceiverInfo

    dev = DeviceInfo.from_system(
        node_id="lab-toronto-01",
        location="Toronto",
        latitude=43.6532, longitude=-79.3832,
        tags=["research", "office"],
    )
    dev.add_receiver(ReceiverInfo(port="/dev/ttyUSB0", chip="ESP32-C6"))

Usage — backend side
--------------------
    from whispy.device import DeviceRegistry

    registry = DeviceRegistry("./devices.json")
    registry.upsert(dev)
    all_devices = registry.list()
    toronto = registry.find(location="Toronto")
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =========================================================================
# Receiver (ESP32) info
# =========================================================================

@dataclass
class ReceiverInfo:
    """A single ESP32 receiver attached to the host via USB-serial."""
    port: str                           # e.g. "/dev/ttyUSB0", "COM5"
    chip: str = "ESP32-C6"              # ESP32 variant
    mac: str = ""                       # ESP32 MAC address (if known)
    firmware: str = ""                  # firmware version
    baud: int = 115200
    status: str = "unknown"             # "active", "idle", "error", "unknown"
    csi_rate: float = 0.0               # current packets/sec
    last_seen: str = ""                 # ISO timestamp

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ReceiverInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def discover_receivers() -> list[ReceiverInfo]:
    """Auto-discover ESP32 receivers on USB-serial ports.

    Scans available serial ports and returns those whose description or
    hardware ID suggests an ESP32 (USB VID/PID 0x303A for Espressif).
    """
    receivers: list[ReceiverInfo] = []
    try:
        from serial.tools.list_ports import comports
        for p in comports():
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            # Espressif USB VID = 0x303A
            is_esp = ("303a" in hwid or "esp32" in desc or "cp210" in desc
                      or "ch340" in desc or "ch9102" in desc)
            if is_esp:
                chip = "ESP32-C6"  # default; could refine by PID
                if "c3" in desc:
                    chip = "ESP32-C3"
                elif "s3" in desc:
                    chip = "ESP32-S3"
                receivers.append(ReceiverInfo(
                    port=p.device,
                    chip=chip,
                    mac="",
                    firmware="",
                    status="unknown",
                ))
    except ImportError:
        pass  # pyserial not installed
    return receivers


# =========================================================================
# Device info
# =========================================================================

@dataclass
class DeviceInfo:
    """Full description of a Whispy node deployed somewhere in the world."""

    # ── identity ────────────────────────────────────────────
    node_id: str                        # globally unique, e.g. "lab-toronto-01"
    name: str = ""                      # human-friendly, e.g. "Toronto Lab Node 1"

    # ── location ────────────────────────────────────────────
    location: str = ""                  # city / building / room
    latitude: float = 0.0
    longitude: float = 0.0
    timezone: str = ""                  # e.g. "America/Toronto"
    floor: str = ""                     # building floor
    room: str = ""                      # room name/number

    # ── hardware ────────────────────────────────────────────
    hostname: str = ""
    platform: str = ""                  # e.g. "Linux-6.1.0-rpi-aarch64"
    python_version: str = ""
    whispy_version: str = ""
    cpu_model: str = ""
    ram_mb: int = 0
    disk_gb: float = 0.0

    # ── receivers ───────────────────────────────────────────
    receivers: list[ReceiverInfo] = field(default_factory=list)

    # ── deployment ──────────────────────────────────────────
    task: str = "occupancy"             # occupancy / har / localization
    model_name: str = ""                # name of deployed model
    labels: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # ── status ──────────────────────────────────────────────
    status: str = "offline"             # "online", "offline", "error"
    uptime_hours: float = 0.0
    last_seen: str = ""                 # ISO timestamp
    registered_at: str = ""             # ISO timestamp
    ip_address: str = ""

    # ── methods ─────────────────────────────────────────────
    def add_receiver(self, r: ReceiverInfo) -> None:
        self.receivers.append(r)

    @classmethod
    def from_system(
        cls,
        node_id: str,
        location: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
        tags: list[str] | None = None,
        task: str = "occupancy",
        auto_discover_receivers: bool = True,
    ) -> DeviceInfo:
        """Create a DeviceInfo populated from the current system."""
        import shutil
        try:
            import whispy
            ver = whispy.__version__
        except Exception:
            ver = "unknown"

        disk = shutil.disk_usage("/")
        ram = 0
        cpu = ""
        try:
            import psutil
            ram = int(psutil.virtual_memory().total / 1024 / 1024)
            # try to read Pi model
        except ImportError:
            pass
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("Model"):
                        cpu = line.split(":")[-1].strip()
                        break
        except Exception:
            cpu = platform.processor() or platform.machine()

        dev = cls(
            node_id=node_id,
            name=f"Whispy {node_id}",
            location=location,
            latitude=latitude,
            longitude=longitude,
            hostname=socket.gethostname(),
            platform=platform.platform(),
            python_version=platform.python_version(),
            whispy_version=ver,
            cpu_model=cpu,
            ram_mb=ram,
            disk_gb=round(disk.total / 1024**3, 1),
            task=task,
            tags=tags or [],
            status="online",
            registered_at=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat(),
        )

        if auto_discover_receivers:
            dev.receivers = discover_receivers()

        # try to get local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            dev.ip_address = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        return dev

    def to_dict(self) -> dict:
        d = asdict(self)
        d["receivers"] = [r.to_dict() if isinstance(r, ReceiverInfo) else r
                          for r in self.receivers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DeviceInfo:
        receivers = [ReceiverInfo.from_dict(r) if isinstance(r, dict) else r
                     for r in d.pop("receivers", [])]
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        dev = cls(**fields)
        dev.receivers = receivers
        return dev

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        lines = [
            f"  Node:       {self.node_id}",
            f"  Location:   {self.location} ({self.latitude:.4f}, {self.longitude:.4f})",
            f"  Host:       {self.hostname} ({self.platform})",
            f"  Task:       {self.task}  Model: {self.model_name or 'none'}",
            f"  Receivers:  {len(self.receivers)}",
        ]
        for r in self.receivers:
            lines.append(f"    - {r.port}  {r.chip}  {r.status}  {r.csi_rate:.0f} Hz")
        lines.append(f"  Status:     {self.status}  Uptime: {self.uptime_hours:.1f}h")
        lines.append(f"  Tags:       {', '.join(self.tags) if self.tags else 'none'}")
        return "\n".join(lines)


# =========================================================================
# Device registry (server side — persists to JSON)
# =========================================================================

class DeviceRegistry:
    """Server-side registry of all known Whispy devices.

    Persists to a JSON file.  Thread-safe for concurrent access from
    FastAPI and the MQTT subscriber.
    """

    def __init__(self, path: str = "./whispy_devices.json"):
        self._path = Path(path)
        self._devices: dict[str, DeviceInfo] = {}
        self._lock = __import__("threading").Lock()
        self._load()

    def _load(self) -> None:
        if self._path.is_file():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                for d in data:
                    dev = DeviceInfo.from_dict(d)
                    self._devices[dev.node_id] = dev
            except Exception as e:
                print(f"[registry] WARNING: Could not load {self._path}: {e}")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([d.to_dict() for d in self._devices.values()], f, indent=2)

    def upsert(self, device: DeviceInfo) -> None:
        """Insert or update a device.  Merges receiver lists."""
        with self._lock:
            existing = self._devices.get(device.node_id)
            if existing:
                # preserve registration time
                device.registered_at = existing.registered_at
            device.last_seen = datetime.now(timezone.utc).isoformat()
            self._devices[device.node_id] = device
            self._save()

    def get(self, node_id: str) -> DeviceInfo | None:
        with self._lock:
            return self._devices.get(node_id)

    def remove(self, node_id: str) -> bool:
        with self._lock:
            if node_id in self._devices:
                del self._devices[node_id]
                self._save()
                return True
            return False

    def list(self) -> list[DeviceInfo]:
        with self._lock:
            return list(self._devices.values())

    def find(self, **kwargs) -> list[DeviceInfo]:
        """Filter devices by attribute values.

        Example: ``registry.find(location="Toronto", task="occupancy")``
        """
        with self._lock:
            results = list(self._devices.values())
            for key, val in kwargs.items():
                results = [d for d in results if getattr(d, key, None) == val]
            return results

    def online(self) -> list[DeviceInfo]:
        return self.find(status="online")

    @property
    def count(self) -> int:
        return len(self._devices)

    def summary(self) -> str:
        lines = [f"Whispy Device Registry — {self.count} devices"]
        online = len(self.online())
        lines.append(f"  Online: {online}  Offline: {self.count - online}")
        for d in self.list():
            icon = "●" if d.status == "online" else "○"
            lines.append(f"  {icon} {d.node_id:25s}  {d.location:20s}  "
                         f"{len(d.receivers)} rx  {d.task}")
        return "\n".join(lines)
