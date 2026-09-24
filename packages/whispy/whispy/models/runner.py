"""ModelRunner — binds a model's named inputs to concrete SensorHandles.

Each active model gets its own runner with its own stream subscriptions
and window builder — no global synchronizer decides all model inputs::

    RuntimeModel / ModelHandle
        └── ModelRunner
               ├── input bindings   {"audio": SensorHandle(rpi1/mic)}
               ├── stream subscriptions (local, LAN, fixture — all equal)
               ├── window builder   (WindowSynchronizer per runner)
               ├── processor/model
               └── predictions

The runner does not care where a handle comes from: a LAN handle and a
local handle both produce ``SensorSample`` streams.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..contracts import (
    InferenceRequest, InferenceResult, InferenceTrace, ModelBinding,
    ModelInput, ModelManifest, Prediction, SensorWindow,
)
from ..devices.base import SensorHandle
from ..processors.base import Processor
from ..streams import SampleStream
from ..synchronization import WindowSynchronizer

#: preprocessing key carrying input_name → sensor_id for the model.
BINDINGS_KEY = "_bindings"


def bound_sensor_id(window: SensorWindow, input_name: str) -> Optional[str]:
    """Resolve a model input name to the bound sensor id in a window."""
    bindings = window.preprocessing.get(BINDINGS_KEY) or {}
    sid = bindings.get(input_name)
    if sid:
        return sid
    # Fall back to modality-style resolution for unbound windows.
    from ..windows import resolve_sensor_id
    return resolve_sensor_id(window, input_name)


def bound_samples(window: SensorWindow, input_name: str):
    """Samples for a named model input (empty list when absent)."""
    sid = bound_sensor_id(window, input_name)
    return list(window.samples.get(sid) or []) if sid else []


class ModelRunner:
    """Runs one model against its own bound sensor streams.

    Parameters
    ----------
    processor:
        The model implementation (``Processor`` interface).
    bindings:
        ``input_name → SensorHandle``. Handles may be local, LAN, or
        fixture — the runner treats them identically.
    window_seconds:
        Default rolling window length for :meth:`predict`.
    device_id:
        Execution node id stamped onto predictions.
    """

    def __init__(self, processor: Processor,
                 bindings: Mapping[str, SensorHandle],
                 inputs: Optional[List[ModelInput]] = None,
                 window_seconds: float = 2.0,
                 device_id: str = "",
                 stale_after_s: float = 5.0,
                 manifest: Optional[ModelManifest] = None,
                 runtime_id: str = "",
                 execution_class: str = "local"):
        self.processor = processor
        self.bindings: Dict[str, SensorHandle] = dict(bindings)
        self.inputs = list(inputs or [])
        self.window_seconds = window_seconds
        self.device_id = device_id
        self.manifest = manifest
        self.runtime_id = runtime_id
        self.execution_class = execution_class
        self._stale_after = stale_after_s
        self._streams: Dict[str, SampleStream] = {}
        self._sync: Optional[WindowSynchronizer] = None
        self._lock = threading.Lock()
        self._started = False

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> "ModelRunner":
        """Open a SampleStream per binding and start ingestion."""
        with self._lock:
            if self._started:
                return self
            for name, handle in self.bindings.items():
                sid = handle.info.id
                stream = SampleStream(handle.stream(), maxlen=8192,
                                      name=f"{name}:{sid}")
                stream.start()
                self._streams[sid] = stream
            self._sync = WindowSynchronizer(
                self._streams, expected=list(self._streams),
                stale_after_s=self._stale_after)
            self._started = True
        return self

    def stop(self) -> None:
        with self._lock:
            for stream in self._streams.values():
                try:
                    stream.close()
                except Exception:
                    pass
            self._streams.clear()
            self._sync = None
            self._started = False

    close = stop

    def __enter__(self) -> "ModelRunner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- inference ---------------------------------------------------------------
    def _window(self, window_seconds: Optional[float] = None) -> SensorWindow:
        assert self._sync is not None
        window = self._sync.rolling(window_seconds or self.window_seconds)
        window.preprocessing[BINDINGS_KEY] = {
            name: handle.info.id for name, handle in self.bindings.items()}
        return window

    def predict(self, window_seconds: Optional[float] = None,
                warmup_s: Optional[float] = None) -> Prediction:
        """Cut one window over the bound streams and run the model.

        ``warmup_s`` defaults to the window length so first-time callers
        get a populated window; pass 0 to predict on whatever is buffered.
        """
        self.start()
        wait = self.window_seconds if warmup_s is None else warmup_s
        if wait > 0:
            time.sleep(wait)
        window = self._window(window_seconds)
        pred = self.processor.predict(window)
        if self.device_id and not pred.device_id:
            pred.device_id = self.device_id
        pred.source_window = pred.source_window or {
            "bindings": dict(window.preprocessing.get(BINDINGS_KEY) or {}),
            "start": window.start_timestamp, "end": window.end_timestamp,
        }
        return pred

    def predict_window(self, window: SensorWindow) -> Prediction:
        """Run the model on an externally-built window (e.g. a capture)."""
        window.preprocessing.setdefault(BINDINGS_KEY, {
            name: handle.info.id for name, handle in self.bindings.items()})
        pred = self.processor.predict(window)
        if self.device_id and not pred.device_id:
            pred.device_id = self.device_id
        return pred

    def stream_predictions(self, interval_s: Optional[float] = None
                           ) -> Iterator[Prediction]:
        """Yield predictions continuously at ``interval_s`` cadence."""
        self.start()
        interval = interval_s or self.window_seconds
        while True:
            yield self.predict(warmup_s=0.0)
            time.sleep(interval)

    # -- canonical inference (§15) -------------------------------------------
    def infer(self, request: Optional[InferenceRequest] = None,
              window_seconds: Optional[float] = None,
              warmup_s: Optional[float] = None) -> InferenceResult:
        """Run one inference and return a canonical InferenceResult.

        Wraps :meth:`predict` with a complete :class:`InferenceTrace`:
        model identity, artifact hash, runtime id, execution device/class,
        input bindings, input interval, latency and confidence.
        """
        request = request or InferenceRequest(
            model_id=self.manifest.model_id if self.manifest else "")
        started = time.time()
        try:
            pred = self.predict(window_seconds=window_seconds,
                                warmup_s=warmup_s)
        except Exception as exc:
            return InferenceResult(
                request_id=request.request_id, status="failed",
                error=str(exc),
                trace=self._trace(request, started, None, None))
        latency_ms = (time.time() - started) * 1000.0
        interval = None
        if pred.source_window:
            interval = {
                "start": float(pred.source_window.get("start") or 0.0),
                "end": float(pred.source_window.get("end") or 0.0),
            }
        return InferenceResult(
            request_id=request.request_id, status="succeeded",
            prediction=pred,
            trace=self._trace(request, started, latency_ms, pred.confidence,
                              interval=interval),
        )

    def _trace(self, request: InferenceRequest, started: float,
               latency_ms: Optional[float], confidence: Optional[float],
               interval: Optional[Dict[str, float]] = None
               ) -> InferenceTrace:
        manifest = self.manifest
        return InferenceTrace(
            model_id=request.model_id or (manifest.model_id if manifest else ""),
            model_version=manifest.version if manifest else "",
            artifact_hash=manifest.artifact_sha256 if manifest else "",
            runtime_id=self.runtime_id,
            execution_device=self.device_id,
            execution_class=self.execution_class,
            input_bindings={name: h.info.id
                            for name, h in self.bindings.items()},
            input_interval=interval,
            inference_timestamp=started,
            latency_ms=latency_ms,
            confidence=confidence,
        )


def bindings_from_config(
        inputs: List[ModelInput],
        binding_cfgs: List[Dict[str, Any]],
        local_device=None,
        lan_resolver=None) -> Dict[str, "SensorHandle"]:
    """Resolve ``ModelBinding`` configs to SensorHandles.

    ``local_device`` is a DeviceHandle for ``source="local"`` bindings;
    ``lan_resolver`` is ``callable(host, config) → DeviceHandle`` for
    ``source="lan"`` (defaults to ``whispy.lan``).
    """
    from ..devices.local import lan as _lan

    out: Dict[str, SensorHandle] = {}
    by_name = {i.name: i for i in inputs}
    for raw in binding_cfgs:
        binding = ModelBinding.from_dict(raw)
        name = binding.input_name or (
            by_name.get(binding.sensor_id).name
            if binding.sensor_id in by_name else binding.sensor_id)
        if binding.source in ("lan", "remote"):
            resolver = lan_resolver or _lan
            device = resolver(binding.source_device,
                              **{k: v for k, v in binding.config.items()
                                 if k in ("port", "token", "timeout")})
            out[name] = device.sensor(binding.sensor_id)
        elif binding.source in ("local", ""):
            if local_device is None:
                raise ValueError(
                    f"binding {name!r} needs a local device")
            out[name] = local_device.sensor(binding.sensor_id)
        else:
            raise ValueError(
                f"unknown binding source {binding.source!r} for {name!r}")
    return out


__all__ = [
    "BINDINGS_KEY",
    "ModelRunner",
    "bound_samples",
    "bound_sensor_id",
    "bindings_from_config",
]
