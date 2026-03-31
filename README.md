# 🔮 Whispy — WiFi Intelligence on ESP

**Whispy** is a Python toolkit for WiFi CSI (Channel State Information) sensing research using ESP32 microcontrollers. It provides an end-to-end pipeline from data collection to federated learning.

## Installation

```bash
pip install whispy                  # core only
pip install whispy[all]             # everything
pip install whispy[collect,camera]  # collection + face detection
pip install whispy[dl,fl]           # deep learning + federated
```

## Quick Start

```bash
# Collect CSI data (1-min standardized files)
whispy collect --port COM5 --label myroom --duration 300

# Collect with face detection (auto occupancy labels)
whispy collect --port COM5 --label indoor --camera

# Deploy a trained model (inference only, no file saved)
whispy deploy --port COM5 --model ./model.pkl

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

## License

MIT
