"""Unified plugin registry — one discovery mechanism for all plugin kinds.

Whispy discovers third-party capabilities through three entry-point
groups::

    [project.entry-points."whispy.sensors"]
    opencv-camera = "whispy_sensor_opencv_camera:OpenCvCameraAdapter"

    [project.entry-points."whispy.models"]
    opencv-haar-person = "whispy_model_opencv_person:HaarPersonModel"

    [project.entry-points."whispy.actuators"]
    windows-speaker = "whispy_actuator_speaker:SpeakerAdapter"

Each group may contain either adapter-style classes (``discover()`` +
``connect()``) or legacy single-device classes (``SensorDriver`` /
action-executor ``Actuator``); the registry tags each entry with the
interface it satisfies so callers never have to guess.

Every plugin reports a uniform :class:`PluginInfo` — package name, entry
point name, version, availability, and any load error — so ``thoth
status`` and conformance tooling can render one consistent inventory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

SENSOR_GROUP = "whispy.sensors"
MODEL_GROUP = "whispy.models"
ACTUATOR_GROUP = "whispy.actuators"


@dataclass
class PluginInfo:
    """Registry record for one discovered plugin."""

    name: str                                  # entry-point name
    group: str                                 # whispy.sensors | …
    package: str = ""                          # providing distribution
    version: str = ""
    kind: str = ""                             # adapter | driver | executor | model
    cls: Optional[Type] = None                 # loaded class (None on error)
    available: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "package": self.package,
            "version": self.version,
            "kind": self.kind,
            "available": self.available,
            "error": self.error,
        }


def _classify(group: str, cls: Type) -> str:
    """Tag a loaded plugin class with the interface it satisfies."""
    try:
        from .sensors.base import SensorAdapter, SensorDriver
        if issubclass(cls, SensorAdapter):
            return "adapter"
        if issubclass(cls, SensorDriver):
            return "driver"
    except Exception:
        pass
    try:
        from .actuators.base import Actuator, ActuatorAdapter
        if issubclass(cls, ActuatorAdapter):
            return "adapter"
        if issubclass(cls, Actuator):
            return "executor"
    except Exception:
        pass
    try:
        from .processors.base import Processor
        if issubclass(cls, Processor):
            return "model"
    except Exception:
        pass
    return "unknown"


class PluginRegistry:
    """Discovers and loads whispy plugins from entry points.

    Results are cached per-process; call :meth:`refresh` after installing
    new packages in the same interpreter.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, PluginInfo]] = {}

    # -- discovery ------------------------------------------------------------
    def _scan(self, group: str) -> Dict[str, PluginInfo]:
        if group in self._cache:
            return self._cache[group]
        from importlib.metadata import entry_points, version

        found: Dict[str, PluginInfo] = {}
        for ep in entry_points(group=group):
            info = PluginInfo(name=ep.name, group=group)
            try:
                dist = getattr(ep, "dist", None)
                if dist is not None:
                    info.package = dist.metadata.get("Name", "")
                    info.version = dist.version
            except Exception:
                pass
            try:
                cls = ep.load()
                if not isinstance(cls, type):
                    raise TypeError(f"{ep.value} is not a class")
                info.cls = cls
                info.kind = _classify(group, cls)
                info.available = True
                if not info.version:
                    try:
                        info.version = version(info.package) if info.package \
                            else getattr(cls, "__version__", "")
                    except Exception:
                        info.version = ""
            except Exception as exc:
                info.error = f"{type(exc).__name__}: {exc}"
                logger.debug("plugin %s (%s) failed to load: %s",
                             ep.name, group, exc)
            found[ep.name] = info
        self._cache[group] = found
        return found

    def refresh(self) -> None:
        self._cache.clear()

    # -- per-kind views ----------------------------------------------------------
    def discover_sensors(self) -> Dict[str, PluginInfo]:
        """Sensor plugins: ``adapter`` (new) or ``driver`` (legacy)."""
        return self._scan(SENSOR_GROUP)

    def discover_models(self) -> Dict[str, PluginInfo]:
        """Model plugins: ``Processor`` implementations."""
        return self._scan(MODEL_GROUP)

    def discover_actuators(self) -> Dict[str, PluginInfo]:
        """Actuator plugins: ``adapter`` (device handles) or ``executor``
        (action executors like home_assistant/webhook)."""
        return self._scan(ACTUATOR_GROUP)

    def all(self) -> Dict[str, PluginInfo]:
        out: Dict[str, PluginInfo] = {}
        for group in (SENSOR_GROUP, MODEL_GROUP, ACTUATOR_GROUP):
            for name, info in self._scan(group).items():
                out[f"{group}:{name}"] = info
        return out

    # -- loading ------------------------------------------------------------------
    def load(self, group: str, name: str) -> Type:
        """Load one plugin class or raise a descriptive error."""
        info = self._scan(group).get(name)
        if info is None:
            raise KeyError(
                f"no {group} plugin named {name!r}; installed: "
                f"{sorted(self._scan(group))}")
        if not info.available or info.cls is None:
            raise ImportError(
                f"plugin {name!r} failed to load: {info.error}")
        return info.cls


_REGISTRY: Optional[PluginRegistry] = None


def registry() -> PluginRegistry:
    """Shared process-wide registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PluginRegistry()
    return _REGISTRY


__all__ = [
    "ACTUATOR_GROUP",
    "MODEL_GROUP",
    "SENSOR_GROUP",
    "PluginInfo",
    "PluginRegistry",
    "registry",
]
