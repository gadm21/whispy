"""Model plugin registry — ``whispy.model("name")`` → ModelHandle.

Model plugins are ``Processor`` implementations registered under the
``whispy.models`` entry-point group::

    [project.entry-points."whispy.models"]
    opencv-haar-person = "whispy_model_opencv_person:HaarPersonModel"

The built-in processors (``rule``, ``torchscript``, ``fusion``) are
registered through the same mechanism so ``whispy.models()`` lists
everything uniformly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from ..contracts import ModelInput, SensorWindow
from ..devices.base import SensorHandle
from ..plugins import PluginInfo, registry
from ..processors.base import PROCESSOR_TYPES, Processor
from .runner import ModelRunner


def _builtin_models() -> Dict[str, Type[Processor]]:
    from ..processors.fusion import FusionProcessor
    from ..processors.rules import RuleProcessor
    from ..processors.torchscript import TorchScriptProcessor

    return {
        "rule": RuleProcessor,
        "torchscript": TorchScriptProcessor,
        "fusion": FusionProcessor,
    }


def installed_models() -> Dict[str, Type[Processor]]:
    """All loadable model classes: built-ins + ``whispy.models`` plugins."""
    out = _builtin_models()
    for name, info in registry().discover_models().items():
        if info.available and info.cls is not None:
            out[name] = info.cls
    return out


def models() -> List[Dict[str, Any]]:
    """Inventory of installed model plugins (built-ins + entry points)."""
    out: List[Dict[str, Any]] = []
    for name, cls in _builtin_models().items():
        out.append({"name": name, "kind": "model", "builtin": True,
                    "available": True, "package": "whispy"})
    for name, info in registry().discover_models().items():
        out.append({**info.to_dict(), "builtin": False})
    return out


class ModelHandle:
    """An installed model: metadata + ``bind()``/``predict()``.

    ``bind`` maps the model's named inputs to SensorHandles and returns
    a :class:`ModelRunner` that owns its own streams and window builder.
    """

    def __init__(self, name: str, processor: Processor,
                 config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.processor = processor
        self.config = dict(config or {})

    # -- introspection ----------------------------------------------------------
    def metadata(self):
        return self.processor.metadata()

    @property
    def inputs(self) -> List[ModelInput]:
        """Declared model inputs (capability form when available)."""
        raw = self.config.get("inputs")
        if raw:
            return [i if isinstance(i, ModelInput)
                    else ModelInput.from_dict(i) for i in raw]
        meta = self.processor.metadata()
        return [ModelInput(name=m, modality=m) for m in (meta.inputs or ())]

    def health(self) -> Dict[str, Any]:
        return self.processor.health()

    # -- binding / inference ------------------------------------------------------
    def bind(self, *args: SensorHandle,
             window_seconds: Optional[float] = None,
             device_id: str = "",
             **inputs: SensorHandle) -> ModelRunner:
        """Bind named inputs to sensor handles.

        ``model.bind(video=camera)`` binds input ``video`` to ``camera``.
        With a single declared input, ``model.bind(camera)`` works too.
        """
        bindings: Dict[str, SensorHandle] = dict(inputs)
        declared = self.inputs
        for handle in args:
            if len(declared) == 1 and declared[0].name not in bindings:
                bindings[declared[0].name] = handle
            elif len(args) == 1 and not declared:
                bindings[handle.info.type or handle.info.id] = handle
            else:
                raise ValueError(
                    f"positional binding needs a declared input name; "
                    f"declared inputs: {[i.name for i in declared]}")
        window = window_seconds or self.config.get("window_seconds") \
            or max((i.window_seconds for i in declared), default=2.0)
        return ModelRunner(self.processor, bindings, inputs=declared,
                           window_seconds=window, device_id=device_id)

    def predict(self, window: SensorWindow) -> Prediction:
        """Run the model on an externally-built window/capture."""
        return self.processor.predict(window)


def model(name: str, config: Optional[Dict[str, Any]] = None,
          artifact: Optional[bytes] = None, **config_kw: Any) -> ModelHandle:
    """Instantiate an installed model plugin by name.

    ``whispy.model("opencv-haar-person")`` or, for built-ins,
    ``whispy.model("rule", config={...})``. Keyword args merge into
    ``config``.
    """
    cfg = dict(config or {})
    cfg.update(config_kw)
    cls = installed_models().get(name)
    if cls is None:
        raise KeyError(
            f"no model {name!r}; installed: {sorted(installed_models())}")
    if name in PROCESSOR_TYPES or name in _builtin_models():
        proc = cls(cfg, artifact=artifact) if name == "torchscript" else cls(cfg)
    else:
        try:
            proc = cls(cfg)
        except TypeError:
            proc = cls()
            if cfg and hasattr(proc, "configure"):
                proc.configure(cfg)
    return ModelHandle(name, proc, config=cfg)


__all__ = ["ModelHandle", "installed_models", "model", "models"]
