# ThothCraft Processors

A **processor** is anything that maps a sensor window to a prediction —
the deployable unit of the ThothCraft model ecosystem. A one-line SNR
threshold and a TorchScript network are the same kind of asset.

## Processor types

| Type | Artifact | Runs where | Example |
|---|---|---|---|
| `rule` | JSON config only | anywhere | `snr_mean > 12 → occupied` |
| `classical` | Python module / pip package | device or Brain | DBSCAN people-count |
| `torchscript` | `.pt` file | device runtime | radar occupancy CNN |
| `fusion` | config + optional code | Brain | radar+CSI vote |

## Interface

```python
from thothcraft.processors import Processor, ProcessorMeta, Prediction, SensorWindow

class MyProcessor(Processor):
    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="my-processor", version="1.0.0",
            processor_type="classical",
            sensor="radar", task="occupancy",
            inputs=("radar",), outputs=("label", "confidence"),
            config_schema={"type": "object", "properties": {
                "snr_threshold": {"type": "number", "default": 12.0}}},
        )

    def predict(self, window: SensorWindow) -> Prediction:
        arr = window.to_numpy("radar")
        ...
        return Prediction(label="occupied", confidence=0.9)

    def configure(self, config):  # optional per-device tunables
        self.threshold = config.get("snr_threshold", 12.0)
```

`SensorWindow` exposes `window["radar"]` / `window.to_numpy(name)` and
derived features: `<sensor>_mean|std|max|min|energy`, plus `snr_mean`.

`Prediction` fields: `label`, `confidence`, `people_count`, `xy`
(N×2 meters), `extras` — the same shape the spatial-state engine,
Home Assistant bridge, and ROS2 bridge consume.

## Rule processors

No code, no artifact — just config evaluated by `RuleProcessor`:

```python
client.create_rule_model(
    "snr-occupancy",
    rules=[{"when": "snr_mean > snr_threshold", "label": "occupied", "confidence": 0.9}],
    else_label="empty",
    params={"snr_threshold": 12.0},       # defaults; per-device overrides via deploy config
    sensor="radar", task="occupancy",
)
```

Expression grammar: `<feature_or_param> <op> <number_or_param>` where
`op ∈ {>, >=, <, <=, ==, !=}`. Rules evaluate in order; first match wins,
otherwise `else_label`.

## Registry

```bash
thothcraft models registry --sensor radar --task occupancy
thothcraft models install thothcraft/radar-occupancy-v2 <device>
thothcraft models publish <model_id> --registry-name you/your-model
```

- `visibility`: `private` (default) → `community` (anyone can deploy) →
  `official` (admin-reviewed)
- `registry_name`: unique `namespace/name-version` handle
- Deploy by name: `device.deploy("thothcraft/radar-occupancy-v2")`

## API

- `GET /api/datasets/models/registry?sensor=&task=` — catalog
- `GET /api/datasets/models/registry/{name}` — resolve handle
- `POST /api/datasets/models/rule` — create rule processor
- `POST /api/datasets/models/{id}/publish?visibility=&registry_name=` — publish
