# Whispy — the ThothCraft programmable sensing SDK

Whispy is the Python SDK for the ThothCraft sensing platform. It defines the
versioned contracts every component converges on — Device, Sensor, Sample,
Window, Prediction, Action, ModelManifest, Deployment, Capture — and provides
the building blocks to sense, window, infer and act:

- **Sensors** — pluggable drivers (radar, camera, microphone, system telemetry,
  fixtures) discovered via entry points.
- **Streams** — bounded, non-blocking `SampleStream` ingestion; sensor reads
  never stall the loop.
- **Windows** — `WindowSynchronizer` cuts rolling windows across sensors with
  explicit missing/stale markers; `WindowFeatures` computes features.
- **Processors** — `rule`, `torchscript` and `fusion` processors turn a
  `SensorWindow` into a `Prediction`.
- **Actuators** — webhook, device and Home Assistant actuators fire explicit
  `ActionResult`s (never fake success).
- **Client** — talk to Brain for remote devices, models and deployments.

```python
import whispy

node = whispy.local()                    # this machine's sensors
for sensor in node.sensors():
    print(sensor.id, sensor.type)

radar = node.sensor("radar")
for sample in radar.stream(max_samples=10):
    print(sample.timestamp, sample.payload)
```

Remote sensing through Brain:

```python
client = whispy.Client()
pi = client.device("thoth-pi-a")
for sample in pi.sensor("radar").stream(max_samples=10):
    ...
```

## Install

```sh
python -m pip install -e packages/whispy          # core SDK
python -m pip install -e "packages/whispy[sensors]"  # + hardware drivers
python -m pip install -e "packages/whispy[dl]"       # + torch inference
```

## The platform

Whispy is one component of the ThothCraft architecture:

- **whispy** (this repo) — the programmable sensing SDK and shared contracts.
- **thoth** — the installable node application (`thoth` CLI, `thoth daemon`)
  that turns a machine into a persistent sensing/inference node.
- **Brain** — the cloud control + data plane (accounts, devices, models,
  deployments, captures) exposed over the versioned `/v1` API.
- **thothHUB** — the web portal for managing devices, models and captures.

## Tests

```sh
python -m pytest packages/whispy/tests
```
