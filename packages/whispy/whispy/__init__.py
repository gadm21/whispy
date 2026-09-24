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
    ActionResult,
    ActionStatus,
    ActuatorCommand,
    ActuatorDescriptor,
    Capture,
    Deployment,
    DeploymentState,
    Device,
    ModalityState,
    ModelBinding,
    ModelInput,
    ModelManifest,
    Prediction,
    RetryPolicy,
    Sensor,
    SensorDescriptor,
    SensorSample,
    SensorWindow,
)
from .devices import (
    DeviceHandle, LanDevice, LocalDevice, RemoteDevice, SensorHandle,
    lan, local,
)
from .cloud.client import Client
from .errors import (
    APIError,
    AuthError,
    EntitlementError,
    NotFoundError,
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
    FixtureDriver,
    HealthReport,
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
    "Action", "ActionResult", "ActionStatus", "ActuatorCommand",
    "ActuatorDescriptor", "Capture", "Deployment", "DeploymentState",
    "Device", "ModalityState", "ModelBinding", "ModelInput",
    "ModelManifest", "Prediction", "RetryPolicy", "Sensor",
    "SensorDescriptor", "SensorSample", "SensorWindow",
    # devices
    "DeviceHandle", "SensorHandle", "LocalDevice", "LanDevice", "RemoteDevice",
    # sensors
    "SensorAdapter", "SensorDriver", "SensorDriverAdapter", "SensorMeta",
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
    "WhispyError", "AuthError", "EntitlementError", "NotFoundError", "APIError",
    "__version__",
]
