# ThothCraft Sensor Drivers

Any hardware becomes a Thoth node by shipping a `SensorDriver` — the
"Works with ThothCraft" contract. Drivers are ordinary pip packages
discovered through the `thothcraft.sensors` entry-point group.

## Interface

```python
from thothcraft.sensors.base import (
    SensorDriver, SensorMeta, SensorFrame, HealthReport,
)

class MyRadarDriver(SensorDriver):
    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="my-radar", version="1.0.0",
            modalities=("radar",),
            config_schema={"type": "object", "properties": {
                "port": {"type": "string", "default": "/dev/ttyUSB0"}}},
        )

    def discover(self):
        # return [{}] per attached device, [] if none
        return [{"port": "/dev/ttyUSB0"}]

    def open(self, config=None):
        self._dev = open_device(config.get("port", "/dev/ttyUSB0"))

    def stream(self):
        while self._running:
            yield SensorFrame.now("radar", self._dev.read_frame())

    def close(self):
        self._running = False
        self._dev.close()

    def health(self) -> HealthReport:
        return HealthReport(status="ok", metrics={"fps": self._fps})
```

## SensorFrame contract

- `sensor_type`: `radar | csi | camera | env | mic | ...`
- `timestamp_ns`: monotonically increasing nanoseconds
- `data`: `numpy.ndarray` in standard units (meters, dB, °C, hPa, %RH)
- `meta`: free-form extras (frame index, gain, ...)

## Packaging

```toml
# pyproject.toml
[project]
name = "thothcraft-sensor-myradar"
dependencies = ["thothcraft-sdk>=0.1.0"]

[project.entry-points."thothcraft.sensors"]
myradar = "thothcraft_sensor_myradar:MyRadarDriver"
```

`pip install thothcraft-sensor-myradar` → `thothcraftd` loads it at
startup; `thothcraft sensors drivers` lists it.

## Conformance — "Works with ThothCraft"

```bash
thothcraft sensors test myradar
```

Runs `check_driver()`: metadata → discover → open → stream (valid frames,
monotonic timestamps) → health → close. Passing + registry listing earns
the verified badge.

## Scaffold a new driver

```bash
thothcraft sensors new myradar   # generates package skeleton + test
```
