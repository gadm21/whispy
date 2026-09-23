# Introduction

**Thothcraft** is a programmable sensing platform. It turns any machine — a
Raspberry Pi, a laptop, a server — into a local-first sensor that can see,
reason, and act on its environment.

The platform is four cooperating pieces:

| Piece | What it is | Install |
|---|---|---|
| **Whispy** | The standalone sensing SDK. Sensors, streams, synchronization, windows, processors, datasets, actuators. | `pip install whispy` |
| **Thoth** | The edge node. A CLI + daemon that runs the sense → predict → act loop on a machine. | `pip install thoth-node` |
| **Brain** | The cloud control + data plane. A versioned `/v1` API for devices, captures, models, and deployments. | `api.thothcraft.com` |
| **thothHUB** | The web portal for pairing, models, captures, and fleet state. | `hub.thothcraft.com` |

## The mental model

```
sensors → SampleStream → WindowSynchronizer → Processor → Prediction → Actuator
```

- **Sensors** produce `SensorSample` measurements — radar, CSI, IMU,
  environmental, camera, and anything a `SensorDriver` plugs in.
- **Streams** carry samples; the **WindowSynchronizer** aligns them into
  fixed `SensorWindow`s.
- A **processor** maps a window to a `Prediction` — a one-line rule or a
  TorchScript network are the same kind of asset.
- An **actuator** turns a prediction into an action — a webhook, a Home
  Assistant call, or a command back to a device.

Everything runs **local-first**: sensing, inference, and actuation happen on
the node. Brain adds fleet management, model distribution, and capture
storage — it is not required for the local loop to work.

## Where to go next

- [Quickstart](/guide/quickstart) — install Thoth + Whispy and run your first
  sensor.
- [Architecture](/guide/architecture) — how the pieces fit together.
- [Whispy SDK](/whispy/) — the programmable sensing API.
- [Brain v1 API](/brain/) — the cloud control plane.
