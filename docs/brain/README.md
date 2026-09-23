# Brain v1 API

Brain is the cloud control and data plane, served at
`https://api.thothcraft.com`. The public surface is a versioned `/v1` REST
API with **ownership-based authorization** — every resource is scoped to the
account that owns it.

Base URL: `https://api.thothcraft.com/v1`

## Authentication

All `/v1` endpoints require an authenticated user. thothHUB and the Whispy
`Client` authenticate against Brain; requests without valid credentials are
rejected.

## Resources

### Account

```
GET /v1/account
```

Returns the authenticated account profile and entitlements.

### Devices

```
GET /v1/devices                      # list your devices (includes offline)
GET /v1/devices/{device_id}          # one device
GET /v1/devices/{device_id}/sensors  # the device's sensors
GET /v1/devices/{device_id}/predictions
```

Devices are the Thoth nodes paired to your account. Each reports its sensors
and online state.

### Captures

```
GET  /v1/devices/{device_id}/captures   # captures for a device
POST /v1/devices/{device_id}/captures   # start a capture (201)
GET  /v1/captures                       # all your captures
POST /v1/captures/{capture_id}/stop     # stop a capture
```

Captures record synchronized sensor windows for datasets and review.

### Streams

```
GET /v1/devices/{device_id}/streams/{sensor_id}
```

Stream a device's sensor live through Brain — the same data a Whispy
`RemoteDevice` consumes.

### Models

```
GET  /v1/models        # list registered models
POST /v1/models        # register a model (201)
```

Models are `whispy-model/v1` packages — a rule config or a TorchScript
artifact plus a `ModelManifest` describing inputs, outputs, and hardware
requirements.

### Deployments

```
GET  /v1/deployments   # list deployments
POST /v1/deployments   # deploy a model to a device (201)
```

A deployment pushes a model to a node. The node validates, installs, and
activates the processor, then acknowledges with a stable
`runtime_model_id`.

## Consuming the API

Use the Whispy `Client` rather than hand-rolling requests:

```python
import whispy
client = whispy.Client()
for device in client.devices():
    print(device.id, device.online)
```

Or call the REST endpoints directly from any HTTP client with your
credentials.
