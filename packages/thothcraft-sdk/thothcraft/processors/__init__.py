"""Processor abstraction: anything that maps sensor windows to predictions.

A ThothCraft "model" is a deployable processor — a declarative threshold
rule, a classical DSP pipeline, a TorchScript network, or a fusion of
other processors' outputs. All share the same interface so the registry,
deployment pipeline, and spatial-state engine treat them uniformly.
"""

from .base import (
    Processor,
    ProcessorMeta,
    Prediction,
    SensorWindow,
    RuleProcessor,
)

__all__ = [
    "Processor",
    "ProcessorMeta",
    "Prediction",
    "SensorWindow",
    "RuleProcessor",
]
