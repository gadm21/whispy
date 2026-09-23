# Whispy Processors

A **processor** maps a sensor window to a `Prediction` — the deployable
unit of the ThothCraft model ecosystem. A one-line SNR threshold and a
TorchScript network are the same kind of asset.

## Processor types

| Type | Artifact | Runs where | Example |
|---|---|---|---|
| `rule` | JSON config only | anywhere | `snr_mean > 12 -> occupied` |
| `torchscript` | `.pt` file | device runtime | radar occupancy CNN |
| `fusion` | config + optional code | device or Brain | radar+CSI vote |

## Interface

```python
from whispy.processors import Processor, ProcessorMeta
from whispy.contracts import Prediction, SensorWindow

class MyProcessor(Processor):
    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="my-processor", version="1.0.0",
            processor_type="torchscript",
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

`SensorWindow` exposes `window["radar"]` / `window.to_numpy(name)` and,
via `WindowFeatures`, derived features: `<sensor>_mean|std|max|min|energy`,
plus `snr_mean`.

`Prediction` fields: `label`, `confidence`, `people_count`, `scores`,
`metadata` — the same shape actuators and the Brain consume.

## Rule processors

No code, no artifact — just config evaluated by `RuleProcessor`:

```python
from whispy.processors import create_processor

proc = create_processor({
    "processor": "rule",
    "name": "snr-occupancy",
    "rules": [{"when": "snr_mean > snr_threshold", "label": "occupied",
               "confidence": 0.9}],
    "else": "empty",
    "params": {"snr_threshold": 12.0},   # per-device overrides via deploy config
    "sensor": "radar", "task": "occupancy",
})
```

Expression grammar supports `and` / `or` / `not`, parentheses, and the
comparison ops `> >= < <= == !=`. Rules evaluate in order; first match
wins, otherwise `else`. Unknown features make a rule non-matching rather
than silently zero, and bad expressions fail at deploy time.

## Instantiation

`create_processor(manifest_or_config, artifact=None)` selects the
implementation from the `processor` field: `rule` → `RuleProcessor`,
`torchscript` → `TorchScriptProcessor`, `fusion` → `FusionProcessor`.

## Deployment

Models are packaged as a `thoth-model/v1` `ModelManifest` and pushed to a
node through Brain's `/v1/deployments` API. The `thoth` node validates,
installs and activates the processor, then acknowledges with a stable
`runtime_model_id`. Manage deployments with the `thoth` CLI:

```bash
thoth models                      # installed runtime models on the node
```
