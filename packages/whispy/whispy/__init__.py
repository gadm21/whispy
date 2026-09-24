"""Whispy — the ThothCraft programmable sensing SDK.

Local sensing::

    import whispy
    node = whispy.local()                  # this machine's sensors
    radar = node.sensor("radar")
    for sample in radar.stream(max_samples=10):
        print(sample.timestamp, sample.payload)

Remote sensing through Brain::

    client = whispy.Client()
    pi = client.device("thoth-pi-a")
    radar = pi.sensor("radar")
    for sample in radar.stream(max_samples=10):
        ...
"""

from .contracts import (
    Action,
    ActionRequest,
    ActionResult,
    ActionStatus,
    ActuatorCommand,
    ActuatorDescriptor,
    Capture,
    ComputeCapability,
    ContextEvent,
    ContextEvidence,
    ContextState,
    Deployment,
    DeploymentState,
    Device,
    InferencePolicy,
    InferenceRequest,
    InferenceResult,
    InferenceTarget,
    InferenceTrace,
    MinuteManifest,
    MinuteSourceData,
    ModalityState,
    ModelBinding,
    ModelInput,
    ModelManifest,
    Observation,
    ObservationWindow,
    Prediction,
    Relationship,
    RetryPolicy,
    Sensor,
    SensorDescriptor,
    SensorSample,
    SensorWindow,
    SourceDescriptor,
)
from .devices import (
    DeviceHandle, LanDevice, LocalDevice, RemoteDevice, SensorHandle,
    SourceHandle, lan, local,
)
from .cloud.client import Client
from .errors import (
    APIError,
    AmbiguousSourceError,
    AuthError,
    EntitlementError,
    NotFoundError,
    SourceError,
    SourceNotFoundError,
    SourceUnavailableError,
    WhispyError,
)
from .processors import (
    FusionProcessor,
    Processor,
    ProcessorMeta,
    RuleProcessor,
    TorchScriptProcessor,
    create_processor,
)
from .actuators import (
    Actuator,
    ActuatorAdapter,
    ActuatorHandle,
    ActuatorMeta,
    DeviceActuator,
    HomeAssistantActuator,
    Speak,
    WebhookActuator,
    create_actuator,
)
from .sensors import (
    ContextAdapter,
    FixtureDriver,
    HealthReport,
    ObservationAdapter,
    SensorAdapter,
    SensorDriver,
    SensorDriverAdapter,
    SensorMeta,
)
from .models import ModelHandle, ModelRunner, model, models
from .capture import CaptureSession, capture_window
from .plugins import PluginInfo, PluginRegistry
from .streams import SampleStream
from .synchronization import WindowSynchronizer
from .windows import WindowFeatures

__version__ = "0.1.0"

__all__ = [
    # entry points
    "local", "lan", "Client", "model", "models", "capture_window",
    # contracts
    "Action", "ActionRequest", "ActionResult", "ActionStatus",
    "ActuatorCommand", "ActuatorDescriptor", "Capture",
    "ComputeCapability", "ContextEvent", "ContextEvidence",
    "ContextState", "Deployment", "DeploymentState", "Device",
    "InferencePolicy", "InferenceRequest", "InferenceResult",
    "InferenceTarget", "InferenceTrace", "MinuteManifest",
    "MinuteSourceData", "ModalityState", "ModelBinding", "ModelInput",
    "ModelManifest", "Observation", "ObservationWindow", "Prediction",
    "Relationship", "RetryPolicy", "Sensor", "SensorDescriptor",
    "SensorSample", "SensorWindow", "SourceDescriptor",
    # devices
    "DeviceHandle", "SensorHandle", "SourceHandle", "LocalDevice",
    "LanDevice", "RemoteDevice",
    # sensors
    "SensorAdapter", "ObservationAdapter", "ContextAdapter",
    "SensorDriver", "SensorDriverAdapter", "SensorMeta",
    "HealthReport", "FixtureDriver",
    # streams/windows
    "SampleStream", "WindowSynchronizer", "WindowFeatures",
    # processors / models
    "Processor", "ProcessorMeta", "RuleProcessor", "TorchScriptProcessor",
    "FusionProcessor", "create_processor", "ModelHandle", "ModelRunner",
    # actuators
    "Actuator", "ActuatorAdapter", "ActuatorHandle", "ActuatorMeta",
    "DeviceActuator", "HomeAssistantActuator", "WebhookActuator",
    "create_actuator", "Speak",
    # plugins
    "PluginInfo", "PluginRegistry", "CaptureSession",
    # errors
    "WhispyError", "AuthError", "EntitlementError", "NotFoundError",
    "APIError", "SourceError", "SourceNotFoundError",
    "AmbiguousSourceError", "SourceUnavailableError",
    "__version__",
]
