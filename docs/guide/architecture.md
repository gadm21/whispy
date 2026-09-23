# Architecture

Thothcraft separates the **sensing runtime** (Whispy), the **edge node**
(Thoth), and the **cloud plane** (Brain + thothHUB). Each layer is usable on
its own.

## Layers

```
┌─────────────────────────────────────────────────────────────┐
│ thothHUB (hub.thothcraft.com) — web portal                  │
│   pair devices · manage models · browse captures · fleet    │
└──────────────────────────────▲──────────────────────────────┘
                               │ HTTPS /v1
┌──────────────────────────────┴──────────────────────────────┐
│ Brain (api.thothcraft.com) — cloud control + data plane     │
│   /v1 devices · sensors · captures · models · deployments   │
└──────────────────────────────▲──────────────────────────────┘
                               │ pairing + telemetry + commands
┌──────────────────────────────┴──────────────────────────────┐
│ Thoth node — edge application (CLI + daemon)                │
│   capture · deployment · diagnostics · IPC · local API      │
│   SMA loop: sense → predict → act                           │
└──────────────────────────────▲──────────────────────────────┘
                               │ depends on
┌──────────────────────────────┴──────────────────────────────┐
│ Whispy — programmable sensing SDK                           │
│   sensors · streams · synchronization · windows ·           │
│   processors · datasets · actuators · local & remote        │
└─────────────────────────────────────────────────────────────┘
```

## Whispy — the SDK

Whispy is the foundation. It defines the contracts every other layer speaks:

- **Contracts** — `Sensor`, `SensorSample`, `SensorWindow`, `Prediction`,
  `Action`, `Capture`, `Deployment`, `ModelManifest`, and friends.
- **Devices** — `LocalDevice` (this machine), `LanDevice` (a node on the
  network), `RemoteDevice` (a node through Brain), all behind `DeviceHandle`.
- **Sensors** — the `SensorDriver` plugin contract plus built-in drivers.
- **Streams & sync** — `SampleStream` and `WindowSynchronizer` turn raw
  samples into aligned `SensorWindow`s.
- **Processors** — `RuleProcessor`, `TorchScriptProcessor`, `FusionProcessor`
  behind `create_processor`.
- **Actuators** — `WebhookActuator`, `HomeAssistantActuator`,
  `DeviceActuator` behind `create_actuator`.

## Thoth — the node

Thoth is the application you install on a machine. It embeds Whispy and adds:

- a **CLI** (`thoth status`, `thoth sensors`, `thoth capture`, `thoth models`,
  `thoth pair`, `thoth doctor`, `thoth daemon`),
- a **daemon** that runs the sense → measure → act (SMA) loop continuously,
- **deployment** handling for `whispy-model/v1` packages pushed from Brain,
- **diagnostics**, **IPC**, and a **local API** on port `5000`.

The daemon's SMA loop:

```
local sensor streams
  → SampleStream
  → WindowSynchronizer
  → active processor
  → Prediction
  → action dispatch → Whispy actuator
```

## Brain — the cloud

Brain is the control and data plane. It exposes a versioned `/v1` REST API
with ownership-based authorization: every device, capture, model, and
deployment is scoped to the account that owns it. Nodes pair with Brain,
report telemetry, and receive deployment commands; thothHUB and the Whispy
`Client` both consume the same API.

## thothHUB — the portal

thothHUB is the web front end for Brain: pair devices, deploy models, start
and stop captures, and watch fleet state.

## Data flow

1. A **Thoth node** pairs with Brain (`thoth pair`) and reports its sensors.
2. You **deploy a model** (`POST /v1/deployments`) — Brain ships a
   `whispy-model/v1` manifest to the node.
3. The node **activates the processor** and runs the SMA loop locally.
4. **Predictions** actuate locally and can be reported to Brain.
5. **Captures** record synchronized sensor windows and upload to Brain for
   datasets and review in thothHUB.
