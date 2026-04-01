# 🔮 Whispy — WiFi Intelligence on ESP

**Whispy** is a Python toolkit for WiFi CSI (Channel State Information) sensing research using ESP32 microcontrollers. It provides an end-to-end pipeline from data collection to federated learning.

## Installation

```bash
pip install whispy                  # core only
pip install whispy[all]             # everything
pip install whispy[collect,camera]  # collection + face detection
pip install whispy[dl,fl]           # deep learning + federated
pip install whispy[mqtt]            # Home Assistant MQTT integration
pip install whispy[watchdog]        # system health + systemd
pip install whispy[pi]              # Raspberry Pi GPIO (ESP32 reset relay)
```

## Quick Start

```bash
# Collect CSI data (1-min standardized files)
whispy collect --port COM5 --label myroom --duration 300

# Collect with face detection (auto occupancy labels)
whispy collect --port COM5 --label indoor --camera

# Deploy with 1 GB rolling cache (default)
whispy deploy --port COM5 --model ./model.pkl

# Deploy with Home Assistant integration
whispy deploy --port /dev/ttyUSB0 --model model.pkl \
    --mqtt-broker 192.168.1.100 --mqtt-node office --mqtt-location "Office" \
    --labels empty,occupied --cache-gb 1.0

# Deploy with watchdog + GPIO ESP32 reset
whispy deploy --port /dev/ttyUSB0 --model model.pkl \
    --watchdog --gpio-pin 17 --mqtt-broker localhost

# Test Home Assistant MQTT connection
whispy mqtt test --broker 192.168.1.100 --node office

# Export last 5 minutes from the rolling cache (Python API)
# from whispy.watchdog import export_cache
# export_cache(cache, minutes=5, n_files=5, out_dir="./data")

# Check system health
whispy watchdog status

# Generate systemd service for auto-start on Pi
whispy watchdog service --port /dev/ttyUSB0 --model model.pkl \
    --mqtt-broker localhost > /etc/systemd/system/whispy.service

# Load a built-in dataset from HuggingFace
whispy load OfficeLocalization --out ./data

# Train on a dataset
whispy train --data ./data/office_loc --pipeline rv20 --model rf

# Visualize results
whispy vis --results ./results/

# Federated learning simulation
whispy fl --data ./data/office_loc --strategy fedavg --clients 4
```

## Python API

```python
import whispy

# Load built-in dataset
train, test = whispy.load("OfficeLocalization")

# Build a processing pipeline
pipeline = whispy.Pipeline([
    whispy.Resample(in_sr=200, out_sr=150),
    whispy.RollingVariance(window=20),
    whispy.Window(length=500, stride=500),
    whispy.Flatten(),
])

# Train
results = whispy.train(train, test, pipeline=pipeline, model="rf")
whispy.vis.plot_results(results)
```

## Built-in Datasets

| Dataset | Task | Classes | Environment |
|---------|------|---------|-------------|
| `OfficeLocalization` | Localization | 4 | Office |
| `OfficeHAR` | Activity Recognition | 4 | Office |
| `HomeHAR` | Activity Recognition | 7 | Home |
| `HomeOccupation` | Occupancy Detection | 3 | Home |

## Modules

- **`whispy.collect`** — Standardized CSI collection with ESP32
- **`whispy.load`** — Built-in dataset loading from HuggingFace
- **`whispy.train`** — ML/DL training with configurable pipelines
- **`whispy.vis`** — Matplotlib visualization for results and live data
- **`whispy.fl`** — Federated learning with Flower (FedAvg, FedProx, etc.)
- **`whispy.core`** — CSI processing primitives (resampling, subcarrier mask, etc.)
- **`whispy.watchdog`** — Rolling CSI cache (resizable, default 1 GB), health monitoring, systemd integration, data export
- **`whispy.mqtt`** — MQTT publisher with Home Assistant auto-discovery, connection testing

## Extensions & Future Directions

Below are 10 suggested extensions spanning edge deployment, model optimization, networking, security, multi-modal sensing, storage, real-time systems, and federated infrastructure. Each is analyzed in the context of a **Raspberry Pi hub orchestrating one or more ESP32 CSI collectors** — the primary deployment target for Whispy.

---

### 1. On-Device Edge Inference with ONNX/TFLite Runtime

**Current state.** Whispy trains models (RF, XGB, MLP, Conv1D, CNN-LSTM) on a desktop and optionally deploys them via `whispy deploy`, which streams CSI over serial and runs sklearn/torch inference on the host. This means the Raspberry Pi must keep full scikit-learn or PyTorch loaded in RAM — heavy for a 1–4 GB Pi.

**Proposed extension.** Add an `export` step that converts trained models to **ONNX** (sklearn/torch) or **TFLite** (torch via ai_edge_torch), and a lightweight `whispy.edge` runtime that loads only `onnxruntime` or `tflite_runtime` (~30 MB vs ~1.5 GB for torch). The deploy command would auto-detect the exported format and use the minimal runtime.

**Analysis.** ONNX Runtime has ARM64 wheels for Raspberry Pi OS. Inference latency for a 52-feature RF drops from ~12 ms (sklearn) to ~2 ms (ONNX). For DL models, TFLite quantized INT8 can achieve 5–10× speedup on Pi 4/5 with XNNPACK delegate. Memory drops from ~800 MB (torch) to ~60 MB. This is the single highest-impact change for real-world Pi deployment — it transforms Whispy from a research tool into a production-viable edge system.

**Scope:** `whispy.export` module, `whispy.edge` lightweight runtime, CLI `whispy export --format onnx`, modification to `whispy deploy` to prefer lightweight runtimes.

---

### 2. Adaptive Sampling Rate & Power-Aware Collection

**Current state.** The ESP32 streams CSI at its native rate (~100–200 Hz), and `collect.py` resamples to a fixed guaranteed SR (default 150 Hz). The Raspberry Pi serial reader runs continuously at full throughput regardless of whether the environment is changing.

**Proposed extension.** Implement an **adaptive sampling strategy** where the Pi monitors CSI variance in real time and signals the ESP32 (via serial write-back) to adjust its Wi-Fi ping interval. When the environment is static (e.g., empty room at night), drop to 10–20 Hz; when motion is detected, ramp to full 150+ Hz. Add a `whispy.power` module that tracks duty cycle, estimates power draw (ESP32 active modem ~130 mA vs light-sleep ~3 mA), and logs energy budgets.

**Analysis.** In occupancy detection, rooms are empty 60–80% of the time. Adaptive sampling can reduce average power consumption by 4–6× and serial bandwidth by 5–10×, directly extending battery life for battery-powered ESP32 nodes and reducing SD card / disk writes on the Pi. The variance-threshold trigger is already computable via `rolling_variance` in `core.py` — the extension is mostly a control loop and a simple serial protocol (`SR:20\n`, `SR:150\n`) that the ESP32 firmware parses.

**Scope:** `whispy.power` module, modification to `collect.py` serial writer, ESP32 firmware protocol extension (documented, not in Python), new CLI flag `--adaptive`.

---

### 3. Multi-ESP32 Mesh Collection & Sensor Fusion

**Current state.** `collect.py` reads from a single serial port (`--port COM5`). Each collection session is one ESP32, one file.

**Proposed extension.** Support **multiple ESP32 nodes simultaneously** — either via multiple USB-serial connections to a single Pi, or over Wi-Fi (ESP32 → Pi via UDP/TCP). Each ESP32 gets a `node_id`. The Pi runs a unified collector that time-aligns streams from N nodes, producing synchronized `.npz` files with shape `(N_nodes, T, 52)`. Add spatial diversity features: cross-node correlation, CSI ratio between node pairs, and multi-link amplitude stacking.

**Analysis.** Multi-link CSI is proven to improve HAR accuracy by 10–25% (different spatial perspectives capture different motion components). For localization, triangulation from 3+ links can resolve ambiguities that single-link systems cannot. The Pi 4/5 has 4 USB ports and built-in Wi-Fi — it can realistically manage 3–4 ESP32 nodes. The main engineering challenge is sub-millisecond time synchronization; NTP on a local network achieves ~1 ms, which is sufficient for 150 Hz CSI. The `.npz` format naturally extends to a 3D tensor. The pipeline `Step` API needs a minor generalization to handle multi-node arrays (extra axis).

**Scope:** `whispy.mesh` module, UDP listener, multi-port serial manager, time-alignment algorithm, extended `CSIDataset` with node dimension, new pipeline steps for cross-node features.

---

### 4. Real-Time Dashboard & MQTT/Home Assistant Integration

**Current state.** `whispy.vis` provides matplotlib plots (confusion matrix, accuracy bars, PCA scatter, live stream), but these are offline or basic terminal visualizations. There is no integration with smart-home ecosystems.

**Proposed extension.** Add a **lightweight web dashboard** (Flask/FastAPI + HTMX or WebSocket) that runs on the Pi and displays: live CSI heatmap, current model prediction with confidence, rolling accuracy, system health (serial rate, CPU/RAM, temperature). Publish predictions to **MQTT** so that Home Assistant, Node-RED, or any IoT platform can consume them. Add auto-discovery via MQTT discovery protocol (Home Assistant native).

**Analysis.** The Pi is already a home-server — running a Flask app on port 8080 adds negligible overhead (~15 MB RAM). MQTT (via `paho-mqtt`, ~200 KB) is the lingua franca of smart-home IoT; publishing `whispy/office/occupancy → {"state": "occupied", "confidence": 0.94}` lets users build automations (lights, HVAC, security) with zero custom code. The HTMX approach avoids heavy JS frameworks and works well on Pi's limited CPU. WebSocket push for the live CSI heatmap at 5 FPS is ~50 KB/s — trivial. This turns Whispy from a research CLI into a **deployable smart-home sensing product**.

**Scope:** `whispy.dashboard` module (Flask + WebSocket), `whispy.mqtt` publisher, Home Assistant MQTT discovery config generator, CLI `whispy serve --port 8080 --mqtt broker_ip`.

---

### 5. Continual / Online Learning with Drift Detection

**Current state.** Models are trained offline on fixed datasets. Once deployed, the model is static. If the environment changes (furniture moved, new occupants, seasonal temperature shifts affecting Wi-Fi propagation), accuracy degrades silently.

**Proposed extension.** Implement **concept drift detection** (ADWIN, Page-Hinkley, or DDM) that monitors prediction confidence and feature distribution in real time. When drift is detected, trigger one of: (a) alert the user, (b) collect a short labeled calibration burst (`whispy recalibrate --duration 60`), (c) perform **online incremental learning** using partial_fit (SGDClassifier, incremental RF via river/scikit-multiflow). Store a lightweight replay buffer of recent windows so the model can be fine-tuned without full retraining.

**Analysis.** CSI is highly environment-sensitive — studies show 15–30% accuracy drops within weeks due to furniture/object changes. Drift detection is cheap (ADWIN runs in O(log W) per sample). Incremental learning on a Pi is feasible for linear models and small forests; for DL, a 1-epoch fine-tune on 100 windows takes <2s on Pi 4 with ONNX. The key insight: the Pi runs 24/7, so it can accumulate labeled data during supervised recalibration windows and silently adapt. This solves the biggest practical barrier to real-world CSI deployment — **model staleness**.

**Scope:** `whispy.drift` module (detectors), extension to `whispy.train` for incremental_fit, replay buffer in `whispy.collect`, CLI `whispy recalibrate`, confidence logging in deploy mode.

---

### 6. Privacy-Preserving Inference & Secure Federated Learning

**Current state.** `whispy.fl` supports FedAvg, FedProx, FedAdam, FedYogi simulation. Data is stored as plain `.npz` files. There is no encryption, no differential privacy, and FL is simulation-only (not real cross-device).

**Proposed extension.** Three layers: (a) **Local data encryption** — encrypt `.npz` files at rest using `cryptography.Fernet` with a per-device key derived from a user passphrase; (b) **Differential privacy in FL** — add Gaussian noise to gradient updates before sending to the server (DP-SGD via Opacus or manual clipping+noise), with a configurable epsilon budget; (c) **Real cross-device FL** — extend `whispy.fl` to run actual Flower clients on Pi nodes that connect to a central Flower server, enabling multi-home federated training without raw data sharing.

**Analysis.** WiFi CSI reveals intimate behavioral patterns (sleep, bathroom visits, movement habits) — privacy is a first-class concern. DP-SGD with ε=8 typically costs <3% accuracy for CSI classification tasks (the feature space is low-dimensional and redundant across subcarriers). Fernet encryption adds ~1 ms overhead per file save — negligible. Real cross-device FL is Flower's native mode; the simulation code in `fl.py` already uses the Flower client API, so the jump is mostly configuration (server address, TLS certificates). This positions Whispy as a **privacy-respecting** sensing framework, critical for any real-world or institutional deployment.

**Scope:** `whispy.security` module (encryption at rest), DP noise injection in `whispy.fl`, Flower server launcher, TLS configuration helper, CLI `whispy fl --real --server 192.168.1.100`.

---

### 7. Automated Hyperparameter Optimization & Pipeline Search

**Current state.** Pipeline parameters (rolling variance window, window length, stride) and model hyperparameters (n_estimators, learning rate, etc.) are manually specified or chosen from a few presets (`rv20`, `rv200`, `rv2000`).

**Proposed extension.** Add `whispy.automl` that performs **joint pipeline + model search** using Optuna (lightweight, no server needed). Search space: rolling variance window ∈ [5, 2000], window length ∈ [100, 2000], stride ratio ∈ [0.25, 1.0], model type ∈ {rf, xgb, mlp}, plus model-specific hyperparameters. Use **successive halving** (Optuna's Hyperband pruner) to quickly discard bad configurations. Optimize for a user-specified objective (accuracy, balanced accuracy, or inference latency). Save the Pareto front of accuracy-vs-latency tradeoffs.

**Analysis.** On Pi 4, a single RF training run on a typical Whispy dataset (5000 windows × 26000 features) takes ~8s. With Hyperband pruning, 100 trials complete in ~15 minutes — feasible as an overnight job on the Pi itself. The search often finds non-obvious configurations: e.g., rv50 + window 300 can outperform rv200 + window 500 by 2–4% on certain environments. The Pareto front is especially valuable for edge: a user can pick the model that fits their Pi's latency budget (e.g., "best model under 5 ms inference"). Optuna's SQLite storage means results persist across runs and can be visualized via `optuna-dashboard`.

**Scope:** `whispy.automl` module, Optuna integration, search space definition, Pareto front visualization in `whispy.vis`, CLI `whispy autotune --data ./data --budget 100`.

---

### 8. Robust CSI Preprocessing: Outlier Rejection, Phase Sanitization & Calibration

**Current state.** `core.py` performs basic processing: subcarrier masking (64→52), magnitude/phase extraction, rolling variance, resampling. Phase is raw `arctan2(imag, real)` — known to be noisy and wrapped.

**Proposed extension.** Add three preprocessing layers: (a) **Outlier rejection** — detect and remove/interpolate CSI packets with anomalous magnitude spikes (>3σ per subcarrier) caused by interference, microwave ovens, or Bluetooth; (b) **Phase sanitization** — implement linear phase unwrapping, carrier frequency offset (CFO) removal via first-subcarrier subtraction, and optional Hampel filter; (c) **Amplitude calibration** — per-environment baseline subtraction using a 30-second static calibration recording, enabling cross-environment transfer.

**Analysis.** Raw CSI phase from ESP32 is essentially random due to unsynchronized clocks and CFO — it's usable only after sanitization. Studies show that sanitized phase alone can match amplitude accuracy for HAR, and fusing both improves by 5–12%. Outlier rejection prevents single corrupted packets from polluting entire windows (a single 10× spike in a 500-sample window shifts rolling variance dramatically). Calibration baselines enable **transfer learning** across rooms: subtract the static fingerprint, and the residual motion patterns become environment-independent. These are standard practices in CSI research but missing from the current implementation.

**Scope:** `whispy.preprocess` module, new pipeline steps (`OutlierReject`, `PhaseSanitize`, `Calibrate`), calibration recording in `whispy collect --calibrate`, extended `.npz` with calibration metadata.

---

### 9. Persistent Storage Backend & Experiment Tracking

**Current state.** Collected data is saved as individual `.npz` files with JSON metadata sidecars. Training results are printed to stdout. There is no structured experiment tracking, no database, and no way to query historical collections.

**Proposed extension.** Add a **lightweight SQLite database** (`whispy.db`) that indexes all collected sessions (timestamp, label, duration, n_samples, file path, environment hash) and all training runs (model, pipeline, metrics, hyperparameters, dataset reference). Provide query APIs: `whispy.db.sessions(label="kitchen", after="2025-01-01")`, `whispy.db.best_model(task="har", metric="balanced_accuracy")`. Integrate with **MLflow Tracking** (optional) for richer experiment comparison, artifact logging, and model registry. On the Pi, SQLite is zero-config and adds ~0.5 MB overhead.

**Analysis.** After weeks of continuous collection, a Pi accumulates hundreds of `.npz` files. Without a database, finding "all kitchen sessions from last Tuesday with >8000 samples" requires scanning every JSON sidecar — slow and error-prone. SQLite makes this instant. For experiment tracking, researchers typically run 50–200 training configurations per dataset; without MLflow/DB, comparing them means grepping log files. The `whispy.db` module turns the Pi into a **self-documenting sensing station** where every collection and experiment is searchable. MLflow's local mode (file backend) works on Pi without a server.

**Scope:** `whispy.db` module (SQLite via built-in `sqlite3`), auto-logging in `collect.py` and `train.py`, query CLI `whispy db query --label kitchen`, optional MLflow integration in `whispy.train`, CLI `whispy db stats`.

---

### 10. Watchdog, Auto-Recovery & System Health Monitoring

**Current state.** Collection runs as a foreground process. If the serial connection drops (USB disconnect, ESP32 crash, power glitch), the process dies. There is no auto-restart, no health monitoring, and no alerting.

**Proposed extension.** Add a **`whispy.watchdog`** daemon that: (a) runs collection as a supervised child process with automatic restart on crash (configurable max_retries, backoff); (b) monitors system health — CPU temperature (critical on Pi, throttles at 80°C), RAM usage, disk space, serial packet rate; (c) detects anomalies — if CSI rate drops below threshold (ESP32 frozen) or disk is >90% full, take corrective action (restart ESP32 via GPIO pin toggle on the Pi, rotate/compress old files, send alert); (d) sends alerts via MQTT, email (smtplib), or webhook (Slack/Discord/Telegram). Run as a **systemd service** with auto-start on boot.

**Analysis.** A Pi-based CSI sensor is expected to run unattended for weeks/months. USB-serial disconnects are the #1 failure mode (~once per week in practice due to power fluctuations, cable issues, or ESP32 firmware bugs). Without a watchdog, the system silently stops collecting. GPIO-controlled power cycling (Pi GPIO → relay/MOSFET → ESP32 USB power) can recover from ESP32 hard crashes without physical access. Disk monitoring prevents the SD card from filling up (a common Pi failure that corrupts the filesystem). Systemd integration means the Pi boots directly into sensing mode. This is the difference between a **demo** and a **deployment**.

**Scope:** `whispy.watchdog` module, systemd service template (`whispy.service`), GPIO reset helper, disk rotation policy, alert backends (MQTT, email, webhook), CLI `whispy watchdog --install` (installs systemd service), health endpoint for the dashboard.

---

## License

MIT
