# Thoth CLI

The `thoth` command controls a node. Most commands talk to the running
daemon over IPC; `daemon` starts it.

## Commands

### `thoth status`

Show node and daemon health — whether the daemon is running, which sensors
are online, and the active deployment.

```bash
thoth status
```

### `thoth sensors`

List the sensors discovered on this machine and their state.

```bash
thoth sensors
```

### `thoth pair`

Pair the node with your Brain account so it appears in thothHUB and can
receive deployments.

```bash
thoth pair
```

### `thoth daemon`

Start the sense → predict → act loop in the foreground. The installer
registers this to run automatically in the background.

```bash
thoth daemon [--window <seconds>] [--tick <seconds>]
```

### `thoth capture`

Record synchronized sensor windows into a capture.

```bash
thoth capture start --sensors radar,csi,imu
thoth capture list
thoth capture stop <capture_id>
```

### `thoth models`

List the runtime models installed on the node.

```bash
thoth models
```

### `thoth predict`

Inject a prediction into the dispatch path (useful for testing actuators).

```bash
thoth predict --label occupied --confidence 0.9
```

### `thoth doctor`

Run diagnostics — install health, driver discovery, daemon connectivity,
and Brain reachability.

```bash
thoth doctor
```

## Global options

```bash
thoth --port 5000 <command>     # target a node API on a custom port
```
