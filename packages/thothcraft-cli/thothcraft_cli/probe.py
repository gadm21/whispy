"""Hardware capability discovery for `thothcraft device init`.

A Thoth device is any authenticated node implementing the device
protocol — this probe reports which sensing modalities the local
machine can offer.
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
from typing import Dict, List


def _has_camera() -> tuple:
    """Detect a usable camera."""
    if platform.system() == "Linux":
        if glob.glob("/dev/video*"):
            return True, "Video device (/dev/video*)"
        return False, "no /dev/video* device"
    if platform.system() == "Darwin":
        return True, "Built-in camera (assumed on macOS)"
    if platform.system() == "Windows":
        # Heuristic: most laptops have one; refine via WMI later.
        return True, "Camera (assumed — verify in app)"
    return False, "unsupported platform"


def _has_microphone() -> tuple:
    if platform.system() == "Linux":
        if os.path.isdir("/proc/asound") and glob.glob("/proc/asound/card*"):
            return True, "ALSA audio device"
        return False, "no ALSA capture device"
    return True, "Built-in microphone (assumed)"


def _wifi_info() -> tuple:
    if platform.system() == "Linux":
        try:
            out = subprocess.run(
                ["iw", "dev"], capture_output=True, text=True, timeout=5)
            if "Interface" in out.stdout:
                return True, "Wi-Fi interface present"
        except (OSError, subprocess.SubprocessError):
            pass
        if glob.glob("/sys/class/net/wl*"):
            return True, "Wi-Fi interface present"
        return False, "no Wi-Fi interface"
    return True, "Wi-Fi (assumed)"


def _has_bluetooth() -> tuple:
    if platform.system() == "Linux":
        if shutil.which("bluetoothctl") or os.path.isdir("/sys/class/bluetooth"):
            return True, "Bluetooth available"
        return False, "unavailable"
    return True, "Bluetooth available"


def _esp32_receivers() -> List[str]:
    """USB-serial ports that may carry ESP32 CSI receivers."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if platform.system() == "Darwin":
        ports = glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*")
    if platform.system() == "Windows":
        try:
            import serial.tools.list_ports  # type: ignore
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            ports = []
    return ports


def _has_radar() -> tuple:
    """Dedicated radar modules attach over serial/USB — same probe."""
    ports = _esp32_receivers()
    return (bool(ports), f"possible on {', '.join(ports)}" if ports else "unavailable")


def _has_csi() -> tuple:
    ports = _esp32_receivers()
    if ports:
        return True, f"ESP32 receiver possible on {', '.join(ports)}"
    return False, "unsupported — needs ESP32 receiver or compatible NIC"


def _has_accel() -> tuple:
    # Laptops rarely expose accelerometers; check common Linux paths.
    if glob.glob("/sys/bus/iio/devices/iio:device*"):
        return True, "IIO sensor found"
    return False, "unavailable"


def scan() -> Dict[str, tuple]:
    """Return {capability: (available, detail)}."""
    return {
        "Camera": _has_camera(),
        "Microphone": _has_microphone(),
        "Wi-Fi": _wifi_info(),
        "Bluetooth": _has_bluetooth(),
        "Accelerometer": _has_accel(),
        "Radar": _has_radar(),
        "CSI": _has_csi(),
    }
