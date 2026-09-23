# Thoth node

**Thoth** is the edge application — the thing you install on a machine to
make it a sensing node. It embeds the Whispy SDK and adds a CLI, a
background daemon, deployment handling, diagnostics, IPC, and a local API.

```bash
pip install thoth-node      # provides the `thoth` command
```

> **Requirements:** Python 3.10 or newer. `thoth-node` depends on
> `whispy >= 0.1.0`.

## What it does

- **Runs the SMA loop** — sense → measure → act, continuously, on-device.
- **Exposes a local API** on port `5000` so Whispy `LanDevice` clients and
  the CLI can talk to it.
- **Receives deployments** — `thoth-model/v1` packages pushed from Brain are
  validated, installed, and activated locally.
- **Pairs with Brain** — `thoth pair` links the node to your account for
  fleet management and captures.
- **Captures datasets** — `thoth capture` records synchronized sensor
  windows for training and review.

## Components

| Subpackage | Responsibility |
|---|---|
| `thoth.cli` | the `thoth` command-line interface |
| `thoth.daemon` | the always-on SMA loop service |
| `thoth.capture` | synchronized capture recording |
| `thoth.deployment` | model install/activate/rollback |
| `thoth.diagnostics` | `thoth doctor` health checks |
| `thoth.ipc` | CLI ↔ daemon communication |
| `thoth.local_api` | the port-`5000` node API |
| `thoth.models` | runtime model registry/store |
| `thoth.settings` | node configuration |

## Local-first

Thoth is designed to keep working without the cloud. Sensing, inference, and
actuation all happen on the node; Brain connectivity adds pairing, model
distribution, and capture upload but is not required for the local loop.

## Next steps

- [CLI reference](/thoth/cli) — every `thoth` command.
- [Daemon](/thoth/daemon) — the SMA loop and service setup.
