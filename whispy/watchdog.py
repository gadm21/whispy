"""Watchdog — rolling CSI cache, health monitoring, systemd integration.

The watchdog maintains a **resizable ring buffer** (default 1 GB) of raw CSI
packets.  When the buffer is full it silently overwrites the oldest data.
An ``export()`` helper lets the user (or an MQTT command) dump the last
N minutes / files from the cache to disk at any time.

Health checks cover CSI rate, CPU temperature, disk usage, memory, and
serial errors.  On a Raspberry Pi the watchdog can power-cycle an ESP32
via a GPIO-controlled relay when the CSI stream stalls.

When running as a systemd ``Type=notify`` service the watchdog sends
periodic ``WATCHDOG=1`` heartbeats and reports status strings visible in
``systemctl status whispy``.

Usage
-----
    from whispy.watchdog import RollingCache, HealthMonitor, WatchdogConfig

    cache = RollingCache(max_bytes=1 * 1024**3)   # 1 GB
    monitor = HealthMonitor(WatchdogConfig())

    # inside deploy loop:
    cache.push(timestamp, rssi, real64, imag64)
    monitor.tick(csi_rate, serial_errors)
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from whispy.core import CSI_SUBCARRIER_MASK, N_SUBCARRIERS_RAW


# =========================================================================
# Rolling CSI cache
# =========================================================================

# Each packet is stored as a flat byte row:
#   timestamp (int64, 8 B) + rssi (float64, 8 B)
#   + real (64 × float64, 512 B) + imag (64 × float64, 512 B)
# Total per packet: 1040 bytes
_PACKET_BYTES = 8 + 8 + N_SUBCARRIERS_RAW * 8 + N_SUBCARRIERS_RAW * 8  # 1040


class RollingCache:
    """Thread-safe, fixed-size ring buffer for raw CSI packets.

    Parameters
    ----------
    max_bytes : Maximum cache size in bytes.  Default 1 GiB.
        The actual capacity in packets is ``max_bytes // 1040``.
    """

    def __init__(self, max_bytes: int = 1 * 1024**3):
        self._capacity = max(1, max_bytes // _PACKET_BYTES)
        # pre-allocate numpy arrays
        self._ts = np.zeros(self._capacity, dtype=np.int64)
        self._rssi = np.zeros(self._capacity, dtype=np.float64)
        self._real = np.zeros((self._capacity, N_SUBCARRIERS_RAW), dtype=np.float64)
        self._imag = np.zeros((self._capacity, N_SUBCARRIERS_RAW), dtype=np.float64)
        self._head = 0          # next write position
        self._count = 0         # total packets ever written (for ordering)
        self._lock = threading.Lock()

    # ── properties ──────────────────────────────────────────
    @property
    def capacity(self) -> int:
        """Max number of packets the cache can hold."""
        return self._capacity

    @property
    def size(self) -> int:
        """Current number of valid packets."""
        with self._lock:
            return min(self._count, self._capacity)

    @property
    def total_pushed(self) -> int:
        """Total packets pushed since creation (including overwritten)."""
        with self._lock:
            return self._count

    @property
    def used_bytes(self) -> int:
        return self.size * _PACKET_BYTES

    @property
    def max_bytes(self) -> int:
        return self._capacity * _PACKET_BYTES

    # ── push ────────────────────────────────────────────────
    def push(self, timestamp: int, rssi: float,
             real: np.ndarray, imag: np.ndarray) -> None:
        """Append one CSI packet.  Overwrites oldest if full."""
        with self._lock:
            idx = self._head % self._capacity
            self._ts[idx] = timestamp
            self._rssi[idx] = rssi
            self._real[idx, :] = real
            self._imag[idx, :] = imag
            self._head = (self._head + 1) % self._capacity
            self._count += 1

    def push_parsed(self, parsed: dict[str, Any]) -> None:
        """Push from a ``parse_csi_line`` result dict."""
        if parsed is None:
            return
        self.push(parsed["timestamp"], parsed["rssi"],
                  parsed["real"], parsed["imag"])

    # ── read ────────────────────────────────────────────────
    def latest(self, n: int | None = None) -> dict[str, np.ndarray]:
        """Return the last *n* packets (or all if *n* is None).

        Returns dict with keys: timestamp, rssi, real, imag — all copies.
        """
        with self._lock:
            total = min(self._count, self._capacity)
            if n is None or n > total:
                n = total
            if n == 0:
                return {
                    "timestamp": np.empty(0, dtype=np.int64),
                    "rssi": np.empty(0, dtype=np.float64),
                    "real": np.empty((0, N_SUBCARRIERS_RAW), dtype=np.float64),
                    "imag": np.empty((0, N_SUBCARRIERS_RAW), dtype=np.float64),
                }
            # ring buffer index arithmetic
            if self._count <= self._capacity:
                start = total - n
                idx = np.arange(start, total)
            else:
                end = self._head  # one past the last written
                start = (end - n) % self._capacity
                if start < end:
                    idx = np.arange(start, end)
                else:
                    idx = np.concatenate([
                        np.arange(start, self._capacity),
                        np.arange(0, end),
                    ])
            return {
                "timestamp": self._ts[idx].copy(),
                "rssi": self._rssi[idx].copy(),
                "real": self._real[idx].copy(),
                "imag": self._imag[idx].copy(),
            }

    def latest_seconds(self, seconds: float, sr: int = 150) -> dict[str, np.ndarray]:
        """Return approximately the last *seconds* of data."""
        n = int(seconds * sr)
        return self.latest(n)

    def clear(self) -> None:
        with self._lock:
            self._head = 0
            self._count = 0

    def resize(self, max_bytes: int) -> None:
        """Resize the cache.  Data beyond the new capacity is lost."""
        new_cap = max(1, max_bytes // _PACKET_BYTES)
        with self._lock:
            old = self.latest()  # copy everything out
            n = min(len(old["timestamp"]), new_cap)
            self._capacity = new_cap
            self._ts = np.zeros(new_cap, dtype=np.int64)
            self._rssi = np.zeros(new_cap, dtype=np.float64)
            self._real = np.zeros((new_cap, N_SUBCARRIERS_RAW), dtype=np.float64)
            self._imag = np.zeros((new_cap, N_SUBCARRIERS_RAW), dtype=np.float64)
            if n > 0:
                self._ts[:n] = old["timestamp"][-n:]
                self._rssi[:n] = old["rssi"][-n:]
                self._real[:n] = old["real"][-n:]
                self._imag[:n] = old["imag"][-n:]
            self._head = n % new_cap
            self._count = n

    def __repr__(self) -> str:
        mb = self.max_bytes / 1024**2
        used = self.used_bytes / 1024**2
        return (f"RollingCache({used:.1f}/{mb:.1f} MB, "
                f"{self.size}/{self.capacity} packets)")


# =========================================================================
# Export from cache
# =========================================================================

def export_cache(
    cache: RollingCache,
    out_dir: str = "./whispy_export",
    minutes: float | None = None,
    n_files: int = 1,
    sr: int = 150,
    label: str = "export",
) -> list[str]:
    """Export data from the rolling cache to ``.npz`` files.

    Parameters
    ----------
    cache : RollingCache to read from.
    out_dir : output directory.
    minutes : how many minutes of data to export.  If None, exports
              everything currently in the cache.
    n_files : split the export into this many files.
    sr : sampling rate (used to compute packet count from minutes).
    label : filename label prefix.

    Returns
    -------
    List of saved .npz file paths.
    """
    if minutes is not None:
        data = cache.latest_seconds(minutes * 60, sr=sr)
    else:
        data = cache.latest()

    n_total = len(data["timestamp"])
    if n_total == 0:
        print("[export] Cache is empty — nothing to export")
        return []

    os.makedirs(out_dir, exist_ok=True)

    # compute mag / phase, apply subcarrier mask
    mag_full = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)
    phase_full = np.arctan2(data["imag"], data["real"])
    mag = mag_full[:, CSI_SUBCARRIER_MASK]
    phase = phase_full[:, CSI_SUBCARRIER_MASK]

    # split into n_files chunks
    chunk_size = max(1, n_total // n_files)
    saved: list[str] = []
    wall = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    for fi in range(n_files):
        lo = fi * chunk_size
        hi = min(lo + chunk_size, n_total) if fi < n_files - 1 else n_total
        if lo >= n_total:
            break

        base = f"{label}_{wall}_{fi + 1:03d}"
        npz_path = os.path.join(out_dir, base + ".npz")
        meta_path = os.path.join(out_dir, base + ".meta.json")

        np.savez_compressed(
            npz_path,
            mag=mag[lo:hi],
            phase=phase[lo:hi],
            rssi=data["rssi"][lo:hi],
            timestamp=data["timestamp"][lo:hi],
        )

        dur = 0.0
        ts_chunk = data["timestamp"][lo:hi]
        if len(ts_chunk) > 1:
            dur = (ts_chunk[-1] - ts_chunk[0]) / 1_000_000

        meta = {
            "label": label,
            "file_index": fi + 1,
            "samples": int(hi - lo),
            "duration_sec": round(dur, 3),
            "sr": sr,
            "subcarriers": int(CSI_SUBCARRIER_MASK.sum()),
            "wall_time": wall,
            "source": "rolling_cache",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [export] {npz_path}  samples={hi - lo}  dur={dur:.1f}s")
        saved.append(npz_path)

    print(f"[export] Done — {len(saved)} files saved to {out_dir}")
    return saved


# =========================================================================
# Health status
# =========================================================================

@dataclass
class HealthStatus:
    check: str          # e.g. "csi_rate", "disk", "cpu_temp"
    level: str          # "OK", "WARNING", "CRITICAL"
    value: float = 0.0
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.level == "OK"


# =========================================================================
# Watchdog config
# =========================================================================

@dataclass
class WatchdogConfig:
    # CSI rate
    csi_rate_warning: float = 50.0
    csi_rate_critical_timeout: float = 10.0

    # Disk
    disk_warning_pct: float = 85.0
    disk_critical_pct: float = 95.0

    # CPU temperature
    cpu_temp_warning: float = 70.0
    cpu_temp_critical: float = 80.0

    # Memory
    memory_warning_pct: float = 80.0

    # GPIO reset
    gpio_reset_pin: int | None = None
    gpio_reset_off_seconds: float = 3.0

    # Data rotation
    data_dir: str = "./whispy_data"
    max_data_gb: float = 4.0

    # Heartbeat
    heartbeat_interval: float = 15.0


# =========================================================================
# Health monitor
# =========================================================================

class HealthMonitor:
    """Application-level watchdog with health checks and recovery actions."""

    def __init__(self, config: WatchdogConfig | None = None):
        self.config = config or WatchdogConfig()
        self._last_nonzero_rate_time = time.time()
        self._sd_notifier: Any = None
        self._gpio_device: Any = None
        self._init_sd_notify()

    # ── systemd integration ─────────────────────────────────
    def _init_sd_notify(self) -> None:
        try:
            import sd_notify
            self._sd_notifier = sd_notify.Notifier()
            if not self._sd_notifier.enabled():
                self._sd_notifier = None
        except ImportError:
            self._sd_notifier = None

    def report_ready(self) -> None:
        """Tell systemd that startup is complete."""
        if self._sd_notifier:
            self._sd_notifier.ready()

    def report_status(self, msg: str) -> None:
        """Update the status line in ``systemctl status``."""
        if self._sd_notifier:
            self._sd_notifier.status(msg)

    def heartbeat(self) -> None:
        """Send WATCHDOG=1 to systemd."""
        if self._sd_notifier:
            self._sd_notifier.notify("WATCHDOG=1")

    # ── individual checks ───────────────────────────────────
    def check_csi_rate(self, rate: float) -> HealthStatus:
        now = time.time()
        if rate > 0:
            self._last_nonzero_rate_time = now

        gap = now - self._last_nonzero_rate_time
        if gap > self.config.csi_rate_critical_timeout:
            return HealthStatus("csi_rate", "CRITICAL", rate,
                                f"No CSI for {gap:.0f}s")
        if rate < self.config.csi_rate_warning and rate > 0:
            return HealthStatus("csi_rate", "WARNING", rate,
                                f"Low rate {rate:.0f} Hz")
        return HealthStatus("csi_rate", "OK", rate, f"{rate:.0f} Hz")

    def check_disk(self) -> HealthStatus:
        usage = shutil.disk_usage("/")
        pct = usage.used / usage.total * 100
        if pct > self.config.disk_critical_pct:
            return HealthStatus("disk", "CRITICAL", pct,
                                f"Disk {pct:.1f}% full")
        if pct > self.config.disk_warning_pct:
            return HealthStatus("disk", "WARNING", pct,
                                f"Disk {pct:.1f}% full")
        return HealthStatus("disk", "OK", pct, f"{pct:.1f}%")

    def check_cpu_temp(self) -> HealthStatus:
        temp = self._read_cpu_temp()
        if temp is None:
            return HealthStatus("cpu_temp", "OK", 0, "N/A")
        if temp > self.config.cpu_temp_critical:
            return HealthStatus("cpu_temp", "CRITICAL", temp,
                                f"CPU {temp:.1f}°C — throttling!")
        if temp > self.config.cpu_temp_warning:
            return HealthStatus("cpu_temp", "WARNING", temp,
                                f"CPU {temp:.1f}°C")
        return HealthStatus("cpu_temp", "OK", temp, f"{temp:.1f}°C")

    def check_memory(self) -> HealthStatus:
        try:
            import psutil
            mem = psutil.virtual_memory()
            pct = mem.percent
        except ImportError:
            return HealthStatus("memory", "OK", 0, "psutil not installed")
        if pct > self.config.memory_warning_pct:
            return HealthStatus("memory", "WARNING", pct,
                                f"RAM {pct:.1f}%")
        return HealthStatus("memory", "OK", pct, f"{pct:.1f}%")

    # ── recovery actions ────────────────────────────────────
    def power_cycle_esp32(self) -> bool:
        """Toggle GPIO pin to cut and restore ESP32 power."""
        pin = self.config.gpio_reset_pin
        if pin is None:
            return False
        try:
            from gpiozero import OutputDevice
            dev = OutputDevice(pin)
            dev.on()   # cut power (relay normally-closed)
            time.sleep(self.config.gpio_reset_off_seconds)
            dev.off()  # restore
            time.sleep(2.0)
            print(f"  [watchdog] ESP32 power-cycled via GPIO {pin}")
            return True
        except Exception as e:
            print(f"  [watchdog] GPIO reset failed: {e}")
            return False

    def rotate_data(self) -> int:
        """Delete oldest .npz files if data_dir exceeds max_data_gb."""
        data_dir = Path(self.config.data_dir)
        if not data_dir.is_dir():
            return 0
        files = sorted(data_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        limit = self.config.max_data_gb * 1024**3
        deleted = 0
        while total > limit and files:
            f = files.pop(0)
            sz = f.stat().st_size
            f.unlink()
            # also remove sidecar
            meta = f.with_suffix(".meta.json")
            if meta.exists():
                meta.unlink()
            total -= sz
            deleted += 1
        if deleted:
            print(f"  [watchdog] Rotated {deleted} files from {data_dir}")
        return deleted

    # ── main tick ───────────────────────────────────────────
    def tick(self, csi_rate: float = 0, serial_errors: int = 0) -> list[HealthStatus]:
        """Run all health checks, take recovery actions, send heartbeat.

        Call this every ~15 seconds from the deploy loop.
        """
        results = [
            self.check_csi_rate(csi_rate),
            self.check_disk(),
            self.check_cpu_temp(),
            self.check_memory(),
        ]

        for r in results:
            if r.level == "CRITICAL" and r.check == "csi_rate":
                if self.config.gpio_reset_pin is not None:
                    self.power_cycle_esp32()
            if r.level == "CRITICAL" and r.check == "disk":
                self.rotate_data()

        self.heartbeat()

        status_parts = [f"{r.check}={r.message}" for r in results]
        self.report_status(" | ".join(status_parts))

        return results

    def summary(self) -> str:
        """Human-readable health summary."""
        results = [
            self.check_csi_rate(0),
            self.check_disk(),
            self.check_cpu_temp(),
            self.check_memory(),
        ]
        icons = {"OK": "✓", "WARNING": "⚠", "CRITICAL": "✗"}
        lines = []
        for r in results:
            icon = icons.get(r.level, "?")
            lines.append(f"  {icon} {r.check:12s}  {r.message}")
        return "\n".join(lines)

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _read_cpu_temp() -> float | None:
        """Read CPU temperature on Linux (Raspberry Pi)."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read().strip()) / 1000
        except Exception:
            pass
        try:
            from gpiozero import CPUTemperature
            return CPUTemperature().temperature
        except Exception:
            return None


# =========================================================================
# systemd service template
# =========================================================================

SYSTEMD_TEMPLATE = """\
[Unit]
Description=Whispy WiFi CSI Sensing Service
After=network-online.target mosquitto.service
Wants=network-online.target

[Service]
Type=notify
User={user}
WorkingDirectory={workdir}
ExecStart={exec_start}

Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5
WatchdogSec=30

MemoryMax=512M
CPUQuota=80%

StandardOutput=journal
StandardError=journal
SyslogIdentifier=whispy

[Install]
WantedBy=multi-user.target
"""


def generate_service_file(
    port: str,
    model_path: str,
    mqtt_broker: str | None = None,
    mqtt_node: str = "office",
    gpio_pin: int | None = None,
    venv_python: str | None = None,
) -> str:
    """Generate the contents of a systemd .service file."""
    import getpass
    python = venv_python or "python3"
    cmd = f"{python} -m whispy deploy --port {port} --model {model_path} --watchdog"
    if mqtt_broker:
        cmd += f" --mqtt-broker {mqtt_broker} --mqtt-node {mqtt_node}"
    if gpio_pin is not None:
        cmd += f" --gpio-pin {gpio_pin}"
    return SYSTEMD_TEMPLATE.format(
        user=getpass.getuser(),
        workdir=os.path.expanduser("~"),
        exec_start=cmd,
    )
