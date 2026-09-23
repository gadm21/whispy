# Whispy Sensor Drivers

Any hardware becomes a Thoth sensor by shipping a `SensorDriver` — the
"Works with Thoth" contract. Drivers are ordinary pip packages discovered
through the `whispy.sensors` entry-point group.

## Interface

```python
from whispy.sensors import SensorDriver, SensorMeta, HealthReport
from whispy.contracts import SensorSample

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
            yield SensorSample(
                device_id=self._device_id, sensor_id="my-radar-0",
                sensor_type="radar", timestamp=time.time(),
                sequence=self._seq, payload=self._dev.read_frame())

    def close(self):
        self._running = False
        self._dev.close()

    def health(self) -> HealthReport:
        return HealthReport(status="ok", metrics={"fps": self._fps})
```

## SensorSample contract

A driver yields real `SensorSample` measurements — never availability
booleans. Health is reported separately via `health()` and `Sensor.online`.

- `sensor_type`: `radar | csi | camera | env | mic | system | ...`
- `timestamp`: seconds (float), monotonically increasing
- `sequence`: monotonically increasing int
- `payload`: the measurement (array/dict) in standard units

## Packaging

```toml
# pyproject.toml
[project]
name = "whispy-sensor-myradar"
dependencies = ["whispy>=0.1.0"]

[project.entry-points."whispy.sensors"]
myradar = "whispy_sensor_myradar:MyRadarDriver"
```

`pip install whispy-sensor-myradar` makes the driver discoverable:
`whispy.local()` auto-detects it, and the `thoth` node daemon loads it at
startup.

## Conformance — "Works with Thoth"

```python
from whispy.sensors import check_driver
report = check_driver(MyRadarDriver())
assert report["passed"]
```

Runs metadata → discover → open → stream (valid samples, monotonic
timestamps) → health → close. Passing earns the verified badge.

## Scaffold a new driver

```bash
thoth sensors new myradar   # generates package skeleton + conformance test
```
