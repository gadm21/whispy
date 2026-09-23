"""System telemetry sensor (CPU/RAM via psutil when available)."""

from .telemetry import SystemTelemetryDriver

__all__ = ["SystemTelemetryDriver"]
