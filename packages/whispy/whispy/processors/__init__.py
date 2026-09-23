"""Whispy processors — rule, torchscript, fusion."""

from .base import PROCESSOR_TYPES, Processor, ProcessorMeta, create_processor
from .rules import RuleProcessor
from .torchscript import TorchScriptProcessor
from .fusion import FusionProcessor

__all__ = [
    "PROCESSOR_TYPES",
    "Processor",
    "ProcessorMeta",
    "create_processor",
    "RuleProcessor",
    "TorchScriptProcessor",
    "FusionProcessor",
]
