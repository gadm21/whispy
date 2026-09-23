# Thoth daemon

The `thoth` daemon is the always-on process that runs the node's
**sense → measure → act (SMA)** loop. The installer registers it to start
automatically; you can also run it in the foreground.

## The SMA loop

```
local sensor streams
  → SampleStream
  → WindowSynchronizer
  → active processor
  → Prediction
  → action dispatch → Whispy actuator
```

Each tick the daemon:

1. Pulls samples from every enabled sensor into `SampleStream`s.
2. Aligns them into `SensorWindow`s via `WindowSynchronizer`.
3. Runs the **active processor** to produce a `Prediction`.
4. Dispatches the configured **action** through a Whispy actuator, honoring
   confidence gating, label filters, and the action's `RetryPolicy`.

## Running it

```bash
thoth daemon                  # foreground
thoth daemon --window 2.0     # window duration (seconds)
thoth daemon --tick 0.5       # loop cadence (seconds)
```

## As a service

The installer registers the daemon for your platform:

- **Linux / Raspberry Pi** — a `systemd` unit (`thoth.service`).
- **macOS** — a LaunchAgent.
- **Windows** — a logon task.

Manage it with your platform's service tools, or check it with
`thoth status` and `thoth doctor`.

## Deployments

When Brain pushes a `thoth-model/v1` deployment, the daemon validates the
manifest, installs the processor artifact, activates it, and acknowledges
with a stable `runtime_model_id`. The new processor takes over the SMA loop
without a restart.

## Local API

The daemon serves a local API on port `5000`. The `thoth` CLI and Whispy
`LanDevice` clients use it to query status, list sensors, and control
captures.
