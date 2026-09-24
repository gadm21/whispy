"""Device handles — local and remote parity (§7.4)."""

from .base import DeviceHandle, SensorHandle
from .local import LocalDevice, LanDevice, lan, local
from .remote import RemoteDevice

__all__ = [
    "DeviceHandle",
    "SensorHandle",
    "LocalDevice",
    "LanDevice",
    "RemoteDevice",
    "local",
    "lan",
]
