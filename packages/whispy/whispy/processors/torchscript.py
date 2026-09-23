"""TorchScriptProcessor — executes .pt artifacts on real windows."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

from ..contracts import Prediction, SensorWindow
from .base import Processor, ProcessorMeta

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


class TorchScriptProcessor(Processor):
    """Runs a TorchScript model against window tensors.

    Config/manifest::

        {
            "processor": "torchscript",
            "name": "radar-occupancy-v2",
            "inputs": [{"sensor": "radar", "window_seconds": 2.0}],
            "outputs": ["empty", "occupied"]
        }

    ``artifact`` is the raw ``.pt`` bytes. The model receives one tensor
    per declared input (samples flattened to float32) and must return a
    score vector aligned with ``outputs``.
    """

    def __init__(self, config: Dict[str, Any],
                 artifact: Optional[bytes] = None,
                 meta: Optional[ProcessorMeta] = None):
        self._config = config
        self._inputs = list(config.get("inputs") or [])
        self._outputs: List[str] = list(config.get("outputs") or [])
        self._meta = meta or ProcessorMeta(
            name=config.get("name", "torchscript-processor"),
            processor_type="torchscript",
            sensor=(self._inputs[0].get("sensor") if self._inputs else "any"),
            task=config.get("task", "occupancy"),
            inputs=tuple(i.get("sensor") for i in self._inputs
                         if isinstance(i, dict)),
        )
        self._model = None
        if artifact is not None:
            self.load(artifact)

    def metadata(self) -> ProcessorMeta:
        return self._meta

    def load(self, artifact: bytes) -> None:
        """Load a TorchScript artifact; raises when torch is unavailable."""
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TorchScriptProcessor requires torch; install whispy[dl]") from exc
        self._model = torch.jit.load(io.BytesIO(artifact), map_location="cpu")
        self._model.eval()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _input_tensor(self, window: SensorWindow, sensor: str):
        import torch  # type: ignore
        samples = window.samples.get(sensor) or []
        vals: List[float] = []
        for s in samples:
            payload = s.payload
            if np is not None:
                vals.extend(np.asarray(payload, dtype=float).ravel().tolist())
            elif isinstance(payload, (list, tuple)):
                vals.extend(float(v) for v in payload)
            else:
                vals.append(float(payload))
        if not vals:
            raise KeyError(f"no samples for input sensor {sensor!r}")
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

    def predict(self, window: SensorWindow) -> Prediction:
        if self._model is None:
            return Prediction(label="error", confidence=0.0,
                              metadata={"error": "model artifact not loaded"})
        missing = window.missing()
        required = [i.get("sensor") for i in self._inputs if isinstance(i, dict)]
        absent = [s for s in required if s in missing or not window.samples.get(s)]
        if absent:
            return Prediction(
                label="unknown", confidence=0.0,
                metadata={"error": f"missing input sensors: {absent}",
                          "missing": absent})
        try:
            import torch  # type: ignore
            tensors = [self._input_tensor(window, i["sensor"])
                       for i in self._inputs]
            with torch.no_grad():
                out = self._model(*tensors) if len(tensors) > 1 else self._model(tensors[0])
            scores = out.squeeze().tolist()
            if not isinstance(scores, list):
                scores = [scores]
        except Exception as exc:
            logger.warning("TorchScript inference failed: %s", exc)
            return Prediction(label="error", confidence=0.0,
                              metadata={"error": str(exc)})

        best = max(range(len(scores)), key=lambda i: scores[i])
        label = (self._outputs[best] if best < len(self._outputs)
                 else str(best))
        score_map = {
            (self._outputs[i] if i < len(self._outputs) else str(i)): float(s)
            for i, s in enumerate(scores)
        }
        return Prediction(label=label, confidence=float(scores[best]),
                          scores=score_map)


__all__ = ["TorchScriptProcessor"]
