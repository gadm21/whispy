# Whispy SDK

Whispy is the standalone programmable-sensing SDK at the heart of
Thothcraft. It runs anywhere Python 3.10+ runs and needs no cloud account
for local sensing.

```bash
pip install whispy
```

Optional extras pull in sensor drivers and deep-learning support:

```bash
pip install "whispy[sensors]"   # bundled sensor drivers
pip install "whispy[dl]"        # TorchScript processors
pip install "whispy[all]"       # everything
```

## Two ways to sense

**Local** — this machine's own sensors:

```python
import whispy

node = whispy.local()                  # LocalDevice
radar = node.sensor("radar")           # SensorHandle
for sample in radar.stream(max_samples=10):
    print(sample.timestamp, sample.payload)
```

**Remote** — a node through Brain:

```python
import whispy

client = whispy.Client()               # talks to api.thothcraft.com
pi = client.device("thoth-pi-a")       # RemoteDevice
radar = pi.sensor("radar")
for sample in radar.stream(max_samples=10):
    ...
```

Both return a `DeviceHandle` with the same interface — `sensors()`,
`sensor(id)`, captures, and predictions — so code is portable between local
and remote.

## The pipeline

```python
import whispy
from whispy import SampleStream, WindowSynchronizer, create_processor

node = whispy.local()
stream = SampleStream(node.sensor("radar").stream())
windows = WindowSynchronizer(window_s=2.0).sync(stream)

proc = create_processor({
    "processor": "rule",
    "name": "snr-occupancy",
    "rules": [{"when": "snr_mean > 12", "label": "occupied", "confidence": 0.9}],
    "else": "empty",
    "sensor": "radar", "task": "occupancy",
})

for window in windows:
    print(proc.predict(window))
```

## API surface

| Area | Symbols |
|---|---|
| Entry points | `local`, `Client` |
| Contracts | `Sensor`, `SensorSample`, `SensorWindow`, `Prediction`, `Action`, `ActionResult`, `Capture`, `Deployment`, `ModelManifest`, `ModelInput`, `Device`, `ModalityState`, `DeploymentState`, `ActionStatus`, `RetryPolicy` |
| Devices | `DeviceHandle`, `SensorHandle`, `LocalDevice`, `LanDevice`, `RemoteDevice` |
| Sensors | `SensorDriver`, `SensorMeta`, `HealthReport`, `FixtureDriver` |
| Streams | `SampleStream`, `WindowSynchronizer`, `WindowFeatures` |
| Processors | `Processor`, `ProcessorMeta`, `RuleProcessor`, `TorchScriptProcessor`, `FusionProcessor`, `create_processor` |
| Actuators | `Actuator`, `WebhookActuator`, `HomeAssistantActuator`, `DeviceActuator`, `create_actuator` |
| Errors | `WhispyError`, `AuthError`, `EntitlementError`, `NotFoundError`, `APIError` |

## Dive deeper

- [Sensors](/whispy/sensors) — write a `SensorDriver`.
- [Streams & windows](/whispy/streams) — `SampleStream`, `WindowSynchronizer`.
- [Processors](/whispy/processors) — rules, TorchScript, fusion.
- [Actuators](/whispy/actuators) — turn predictions into actions.
- [Remote devices](/whispy/remote) — `Client`, `LanDevice`, `RemoteDevice`.
