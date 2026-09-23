# Remote devices

Whispy reaches sensors three ways, all behind the same `DeviceHandle`
interface — so the same code works on your laptop, a node on your LAN, or a
node halfway around the world.

| Class | Reach | How |
|---|---|---|
| `LocalDevice` | this machine | `whispy.local()` |
| `LanDevice` | a node on the network | direct local API (port `5000`) |
| `RemoteDevice` | a node via Brain | `whispy.Client().device(id)` |

## Local

```python
import whispy
node = whispy.local()          # LocalDevice — this machine's sensors
```

## LAN

A `thoth` node exposes a local API on port `5000`. Connect to it directly:

```python
from whispy.devices import LanDevice
node = LanDevice("192.168.1.42")   # or the node's hostname
```

## Remote via Brain

`Client` authenticates to Brain (`api.thothcraft.com`) and returns
`RemoteDevice` handles for the nodes you own.

```python
import whispy

client = whispy.Client()
client.login()                       # or set credentials via env
pi = client.device("thoth-pi-a")     # RemoteDevice
radar = pi.sensor("radar")
for sample in radar.stream(max_samples=10):
    ...
```

The default Brain base URL is `https://api.thothcraft.com`; override it with
the `THOTHCRAFT_API_URL` environment variable for self-hosted or staging
deployments.

## A uniform handle

`LocalDevice`, `LanDevice`, and `RemoteDevice` all implement `DeviceHandle`:

- `sensors()` → list the device's `Sensor`s
- `sensor(id)` → a `SensorHandle` you can `stream()` from
- captures and predictions

Write your pipeline once against `DeviceHandle`; point it at a local, LAN,
or remote node by changing the constructor.

## Errors

Remote calls raise the SDK's error hierarchy:

- `AuthError` — bad or missing credentials
- `EntitlementError` — plan limit reached
- `NotFoundError` — device/capture/model doesn't exist
- `APIError` — any other Brain error

All derive from `WhispyError`, so `except WhispyError` catches them all.
