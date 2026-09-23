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


# USB vendor IDs of the UART bridges used on ESP32 dev boards.
_ESP32_VIDS = {"10C4", "1A86", "0403", "303A", "067B"}
# Strong identity hints — an explicit Espressif/ESP32 string alone is enough.
_ESP32_STRONG = ("esp32", "espressif", "usb jtag", "usb-serial converter")
# Bridge-chip hints — only trusted together with a matching vendor ID, so a
# generic "USB Serial Port" adapter is never reported as a CSI receiver.
_ESP32_BRIDGE = ("cp210", "ch340", "ch341", "ch343", "wch", "silicon labs",
                 "ftdi", "ft232", "pl2303", "uart bridge", "usb to uart")


def _esp32_receivers() -> List[str]:
    """USB-serial ports that carry genuine ESP32 CSI receivers (excludes Bluetooth)."""
    valid_ports = []
    if platform.system() == "Windows":
        try:
            import serial.tools.list_ports  # type: ignore
            for p in serial.tools.list_ports.comports():
                hwid = (p.hwid or "").upper()
                desc = (p.description or "").lower()
                # Exclude virtual Bluetooth serial ports
                if "BTHENUM" in hwid or "bluetooth" in desc:
                    continue
                strong = any(k in desc or k in hwid.lower() for k in _ESP32_STRONG)
                bridged = (any(v in hwid for v in _ESP32_VIDS)
                           and any(k in desc for k in _ESP32_BRIDGE))
                if strong or bridged:
                    valid_ports.append(p.device)
        except ImportError:
            pass
    elif platform.system() == "Darwin":
        ports = glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*")
        valid_ports.extend(ports)
    else:
        ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        valid_ports.extend(ports)
    return valid_ports


def _has_radar() -> tuple:
    """Detect mmWave radar (Infineon BGT60TR13C via SPI or USB)."""
    # Check Linux SPI bus for MMW-HAT
    if glob.glob("/dev/spidev*"):
        return True, "SPI radar interface (/dev/spidev*)"
    # Check USB-attached radar devices — require an explicit radar/chip
    # identifier so generic serial adapters never report as radar.
    radar_keywords = ("bgt60", "radar", "infineon", "xensiv", "mmwave",
                      "iwr", "awr", "xethru")
    if platform.system() == "Windows":
        try:
            import serial.tools.list_ports  # type: ignore
            for p in serial.tools.list_ports.comports():
                hwid = (p.hwid or "").upper()
                desc = (p.description or "").lower()
                if "BTHENUM" in hwid or "bluetooth" in desc:
                    continue
                if any(k in desc or k in hwid.lower() for k in radar_keywords):
                    return True, f"Radar detected on {p.device} ({p.description})"
        except ImportError:
            pass
    return False, "unavailable — no mmWave radar hardware detected"


def _has_csi() -> tuple:
    ports = _esp32_receivers()
    if ports:
        return True, f"ESP32 receiver on {', '.join(ports)}"
    return False, "unavailable — no ESP32 receiver connected"


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
