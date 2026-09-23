"""Device abstraction — same interface for local and remote nodes (§7.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from ..contracts import Device, Sensor, SensorSample


class SensorHandle(ABC):
    """A streamable sensor on a device."""

    @property
    @abstractmethod
    def info(self) -> Sensor:
        """Static sensor contract."""

    @abstractmethod
    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        """Yield real samples; raises on disconnect/authorization errors."""

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None


class DeviceHandle(ABC):
    """A Thoth node — local (same machine/LAN) or remote (via Brain)."""

    @property
    @abstractmethod
    def info(self) -> Device:
        """Device contract."""

    @abstractmethod
    def sensors(self) -> List[Sensor]:
        """Sensor inventory."""

    @abstractmethod
    def sensor(self, sensor_id_or_type: str) -> SensorHandle:
        """Resolve a sensor by id or type."""

    def status(self) -> Dict[str, Any]:
        return self.info.to_dict()


__all__ = ["DeviceHandle", "SensorHandle"]
