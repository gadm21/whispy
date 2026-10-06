"""Host radio-environment scanner — Wi-Fi + BLE as seen by the node itself.

Pairs with the ESP32-C6 firmware scan extension: while the ESP32 reports
``WIFI_DATA``/``BLE_DATA`` lines over serial, this adapter lets the *host*
(the Pi or laptop it is attached to) contribute its own radios as another
RSSI vantage point, so the deployment has multilateration-grade coverage.

Payload schema matches the firmware lines exactly:

- ``wifi_scan``  ``{mac, kind:"ap", rssi, channel, ssid, freq_mhz}``
- ``ble_scan``   ``{addr, addr_type, rssi, tx_power, name, mfg}``
- ``self``       ``{role:"host", mac, name:"<hostname>", owner}``

Discovery:
- Wi-Fi — Windows: ``netsh wlan show networks mode=bssid``.
  Linux: ``iw dev <if> scan`` (needs CAP_NET_ADMIN/root; grant the
  thoth-node unit ``AmbientCapabilities=CAP_NET_ADMIN``), fallback to
  ``wpa_cli -i <if> scan_results`` and ``nmcli dev wifi list``.
- BLE — ``bleak`` (BlueZ on Linux, WinRT on Windows). If bleak is not
  installed the adapter still serves Wi-Fi; BLE samples just never appear.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import platform
import re
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional

from whispy import (SensorAdapter, SensorDescriptor, SensorHandle,
                    SensorSample)

log = logging.getLogger(__name__)

_IS_WIN = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Wi-Fi scan backends
# ---------------------------------------------------------------------------


def _run(cmd: List[str], timeout: float = 15.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout,
                             check=False)
        return out.stdout.decode("utf-8", errors="replace")
    except Exception as exc:                       # noqa: BLE001
        log.debug("%s failed: %s", cmd, exc)
        return None


def _wifi_scan_windows() -> List[Dict[str, Any]]:
    """netsh wlan show networks mode=bssid -> AP list."""
    text = _run(["netsh", "wlan", "show", "networks", "mode=bssid"])
    if not text:
        return []
    aps: List[Dict[str, Any]] = []
    ssid: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"\s*SSID\s+\d+\s*:\s*(.*)", line)
        if m:
            ssid = m.group(1).strip() or None
            continue
        m = re.match(r"\s*BSSID\s+\d+\s*:\s*([0-9a-fA-F:]+)", line)
        if m:
            aps.append({"mac": m.group(1).lower(), "ssid": ssid})
            continue
        if not aps:
            continue
        m = re.match(r"\s*Signal\s*:\s*(\d+)%", line)
        if m:
            # Windows only exposes percent; map to a plausible dBm.
            aps[-1]["rssi"] = int(int(m.group(1)) / 2) - 100
            continue
        m = re.match(r"\s*Channel\s*:\s*(\d+)", line)
        if m:
            aps[-1]["channel"] = int(m.group(1))
    return aps


def _wifi_ifaces_linux() -> List[str]:
    text = _run(["iw", "dev"]) or ""
    return re.findall(r"Interface\s+(\S+)", text)


def _chan_from_freq(freq_mhz: int) -> Optional[int]:
    if freq_mhz == 2484:
        return 14
    if 2400 < freq_mhz < 2500:
        return (freq_mhz - 2407) // 5
    if 5000 < freq_mhz < 5900:
        return (freq_mhz - 5000) // 5
    return None


def _parse_iw_dump(text: str) -> List[Dict[str, Any]]:
    """Parse 'iw ... scan dump' output."""
    aps: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        m = re.match(r"BSS ([0-9a-f:]{17})\(", line.strip())
        if m:
            cur = {"mac": m.group(1), "rssi": None, "ssid": None,
                   "channel": None, "freq_mhz": None}
            aps.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s*freq:\s*([\d.]+)", line)
        if m:
            cur["freq_mhz"] = int(float(m.group(1)))
            cur["channel"] = _chan_from_freq(cur["freq_mhz"])
            continue
        m = re.match(r"\s*signal:\s*(-?[\d.]+)", line)
        if m:
            cur["rssi"] = int(float(m.group(1)))
            continue
        m = re.match(r"\s*SSID:\s*(.*)", line)
        if m:
            cur["ssid"] = m.group(1) or None
    return [a for a in aps if a.get("rssi") is not None]


def _wifi_scan_linux(iface: str) -> List[Dict[str, Any]]:
    text = _run(["iw", "dev", iface, "scan"])
    if text and "BSS " in text:
        return _parse_iw_dump(text)
    # cached results sometimes readable without privileges
    text = _run(["iw", "dev", iface, "scan", "dump"])
    if text and "BSS " in text:
        return _parse_iw_dump(text)
    # wpa_supplicant route (netdev group works without root)
    _run(["wpa_cli", "-i", iface, "scan"])
    text = _run(["wpa_cli", "-i", iface, "scan_results"])
    if text and "\n" in text:
        aps = []
        for row in text.splitlines()[1:]:
            f = row.split("\t")
            if len(f) < 5:
                continue
            try:
                aps.append({"mac": f[0], "freq_mhz": int(f[1]),
                            "rssi": int(f[2]),
                            "channel": _chan_from_freq(int(f[1])),
                            "ssid": f[4] or None})
            except ValueError:
                continue
        if aps:
            return aps
    # NetworkManager route
    text = _run(["nmcli", "-t", "-f", "BSSID,SIGNAL,CHAN,SSID,FREQ",
                 "dev", "wifi", "list"])
    if not text:
        return []
    aps = []
    for row in text.splitlines():
        f = row.split(":")
        if len(f) < 7 or not re.fullmatch(r"[0-9A-Fa-f]{2}", f[0]):
            continue
        try:
            mac = ":".join(f[:6]).lower()
            sig_dbm = int(int(f[6]) / 2) - 100 if f[6].isdigit() else None
            aps.append({"mac": mac, "rssi": sig_dbm,
                        "channel": int(f[7]) if f[7].isdigit() else None,
                        "ssid": f[8] if len(f) > 8 and f[8] else None})
        except (ValueError, IndexError):
            continue
    return aps


def wifi_scan() -> List[Dict[str, Any]]:
    """Scan host Wi-Fi for APs. Returns [] when no radio/no permission."""
    if _IS_WIN:
        return _wifi_scan_windows()
    for iface in _wifi_ifaces_linux():
        aps = _wifi_scan_linux(iface)
        if aps:
            return aps
    return []


def host_mac() -> Optional[str]:
    """WLAN adapter MAC (station identity for self/localization keys)."""
    if _IS_WIN:
        text = _run(["netsh", "wlan", "show", "interfaces"]) or ""
        m = re.search(r"Physical address\s*:\s*([0-9A-Fa-f:]{17})", text)
        return m.group(1).lower() if m else None
    for iface in _wifi_ifaces_linux():
        try:
            with open(f"/sys/class/net/{iface}/address") as fh:
                return fh.read().strip()
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# BLE scan via bleak (optional)
# ---------------------------------------------------------------------------


def _bleak():
    try:
        import bleak  # type: ignore
        return bleak
    except Exception:
        return None


def ble_scan(seconds: float) -> List[Dict[str, Any]]:
    """Run one BLE discovery window; returns advertiser list.

    Empty list when bleak is missing, no adapter exists, or nothing was
    heard in the window.
    """
    bleak = _bleak()
    if bleak is None:
        return []
    seen: Dict[str, Dict[str, Any]] = {}

    async def _go() -> None:
        def _cb(device, adv):                     # noqa: ANN001
            mfg = None
            if adv.manufacturer_data:
                for cid, blob in adv.manufacturer_data.items():
                    mfg = f"{cid:04x}" + blob.hex()
                    break
            seen[device.address.lower()] = {
                "addr": device.address.lower(),
                "addr_type": 0,
                "rssi": adv.rssi,
                "tx_power": adv.tx_power,
                "name": adv.local_name or device.name,
                "mfg": mfg,
            }
        scanner = bleak.BleakScanner(detection_callback=_cb)
        try:
            await scanner.start()
            await asyncio.sleep(max(0.5, seconds))
        finally:
            try:
                await scanner.stop()
            except Exception:
                pass

    try:
        asyncio.run(_go())
    except Exception as exc:                       # noqa: BLE001
        log.debug("ble scan failed: %s", exc)
        return []
    return list(seen.values())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class _RadioHandle(SensorHandle):
    """Polls host Wi-Fi periodically and BLE in duty-cycled windows."""

    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        cfg = dict(config or {})
        meta = descriptor.metadata or {}
        self._wifi_period = float(
            cfg.get("wifi_period_s", meta.get("wifi_period_s", 10.0)))
        self._ble_period = float(
            cfg.get("ble_period_s", meta.get("ble_period_s", 20.0)))
        self._ble_window = float(
            cfg.get("ble_window_s", meta.get("ble_window_s", 6.0)))
        self._owner = str(cfg.get("owner", meta.get("owner", "")))
        self._seq = itertools.count()

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
            sensor_type="radio_env",
            timestamp=time.time(),
            sequence=next(self._seq),
            payload_type=ptype,
            payload={"source": "host", **data},
            metadata={"adapter": "radio",
                      "hardware_id": self._desc.hardware_id},
        )

    def stream(self, max_samples: Optional[int] = None):
        count = 0

        def _emit(s: SensorSample):
            nonlocal count
            count += 1
            return s

        yield _emit(self._sample("self", {
            "role": "host",
            "mac": host_mac(),
            "name": socket.gethostname(),
            "owner": self._owner or None,
        }))
        if max_samples is not None and count >= max_samples:
            return

        next_wifi = 0.0
        next_ble = time.monotonic() + 2.0     # brief settle before BLE
        while True:
            now = time.monotonic()
            if now >= next_wifi:
                for ap in wifi_scan():
                    yield _emit(self._sample("wifi_scan", {
                        "mac": ap.get("mac"),
                        "kind": "ap",
                        "rssi": ap.get("rssi"),
                        "channel": ap.get("channel"),
                        "ssid": ap.get("ssid"),
                        "freq_mhz": ap.get("freq_mhz"),
                    }))
                    if max_samples is not None and count >= max_samples:
                        return
                next_wifi = now + self._wifi_period
            if now >= next_ble:
                for dev in ble_scan(self._ble_window):
                    yield _emit(self._sample("ble_scan", dev))
                    if max_samples is not None and count >= max_samples:
                        return
                next_ble = time.monotonic() + self._ble_period
            time.sleep(0.2)

    def latest(self) -> Optional[SensorSample]:
        for s in self.stream(max_samples=2):
            if s.payload_type != "self":
                return s
        return None

    def close(self) -> None:
        pass


class RadioSensorAdapter(SensorAdapter):
    """Discovers one radio_env sensor when the host has a Wi-Fi or BLE radio."""

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Host Wi-Fi/BLE scanner",
            "version": "0.1.0",
            "description": ("Wi-Fi AP + BLE advertiser RSSI from the host "
                            "radios (netsh/iw + bleak)"),
            "config_schema": {
                "type": "object",
                "properties": {
                    "wifi_period_s": {"type": "number", "default": 10},
                    "ble_period_s": {"type": "number", "default": 20},
                    "ble_window_s": {"type": "number", "default": 6},
                    "owner": {"type": "string"},
                },
            },
        }

    def discover(self) -> List[SensorDescriptor]:
        has_wifi = bool(_wifi_ifaces_linux()) or (
            _IS_WIN and bool(_run(["netsh", "wlan", "show", "interfaces"])))
        has_ble = _bleak() is not None
        if not (has_wifi or has_ble):
            return []
        hw = host_mac() or socket.gethostname()
        caps = []
        if has_wifi:
            caps.append("wifi_scan")
        if has_ble:
            caps.append("ble_scan")
        return [SensorDescriptor(
            id=SensorDescriptor.make_id("radio", hw),
            modality="radio_env",
            adapter="radio_env",
            name=f"{socket.gethostname()} radios",
            hardware_id=hw,
            capabilities=caps,
            config_schema=self.metadata().get("config_schema", {}),
            stable=True,
            metadata={
                "platform": platform.system(),
                "ble_backend": "bleak" if has_ble else None,
            },
        )]

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> SensorHandle:
        return _RadioHandle(descriptor, config)
