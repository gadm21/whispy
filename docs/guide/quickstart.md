# Quickstart

Install the Thoth node (which bundles the Whispy SDK) and run your first
sensor in a few minutes.

## Install

**Linux / Raspberry Pi**

```bash
curl -fsSL https://get.thothcraft.com/install.sh | sudo bash
```

**macOS**

```bash
curl -fsSL https://get.thothcraft.com/install.sh | bash
```

**Windows (PowerShell)**

```powershell
irm https://get.thothcraft.com/install.ps1 | iex
```

The installer installs the `whispy` SDK and the `thoth` node app, then
registers the `thoth` daemon to run in the background.

> **Requirements:** Python 3.10 or newer.

## Verify

```bash
thoth status      # node + daemon health
thoth sensors     # discovered sensors on this machine
thoth doctor      # diagnose install, drivers, connectivity
```

## Pair with Brain

Pairing links the node to your account so it appears in thothHUB and can
receive model deployments.

```bash
thoth pair
```

Then open [thothHUB](https://hub.thothcraft.com) to see the device.

## Run the daemon

```bash
thoth daemon      # starts the sense → predict → act loop
```

## Use the SDK directly

```python
import whispy

node = whispy.local()                 # this machine's sensors
for sensor in node.sensors():
    print(sensor.id, sensor.type, sensor.online)

radar = node.sensor("radar")
for sample in radar.stream(max_samples=10):
    print(sample.timestamp, sample.payload)
```

## Next steps

- [Whispy SDK](/whispy/) — streams, windows, processors, actuators.
- [Thoth CLI](/thoth/cli) — every command.
- [Brain v1 API](/brain/) — devices, captures, models, deployments.
