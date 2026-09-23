# whispy

The ThothCraft programmable sensing SDK — local and remote sensors,
windows, processors, models, and actuators through one Python API.

```bash
pip install whispy
```

## Local sensing

```python
import whispy

node = whispy.local()              # this machine's sensors
print(node.sensors())
system = node.sensor("system")
for sample in system.stream(max_samples=5):
    print(sample.timestamp, sample.payload)
```

## Remote sensing through Brain

```python
import whispy

client = whispy.Client()           # ~/.whispy/credentials.json
for device in client.devices():
    print(device.info.name, device.info.online)

pi = client.device("thoth-pi-a")
radar = pi.sensor("radar")
for sample in radar.stream(max_samples=10):
    print(sample.timestamp, sample.payload)
```

## Processors

```python
from whispy import RuleProcessor

proc = RuleProcessor({
    "name": "radar-occupancy",
    "rules": [{"when": "snr_mean > 12 and radar_energy > 0.4",
               "label": "occupied", "confidence": 0.9}],
    "else": "empty",
})
prediction = proc.predict(window)
```

## Actuators

Actuators return explicit results — `queued | executing | succeeded |
failed | unsupported`. Nothing reports success without a confirmed
downstream effect.

```python
from whispy import Action, create_actuator

action = Action(type="webhook", config={"url": "https://hooks.slack.com/..."},
                min_confidence=0.8)
result = create_actuator(action).trigger(action, prediction)
assert result.status.value == "succeeded"
```

## Contracts

`whispy.contracts` is the single source of truth for Device, Sensor,
SensorSample, SensorWindow, Prediction, Action, ModelManifest
(`whispy-model/v1`; legacy `thoth-model/v1` accepted), and the Deployment
state machine — shared by Thoth, Brain, thothHUB, and the mobile app.
