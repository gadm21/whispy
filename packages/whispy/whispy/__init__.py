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
    Capture,
    Deployment,
    DeploymentState,
    Device,
    ModalityState,
    ModelInput,
    ModelManifest,
    Prediction,
    RetryPolicy,
    Sensor,
    SensorSample,
    SensorWindow,
)
from .devices import DeviceHandle, LanDevice, LocalDevice, RemoteDevice, SensorHandle, local
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
    DeviceActuator,
    HomeAssistantActuator,
    WebhookActuator,
    create_actuator,
)
from .sensors import SensorDriver, SensorMeta, HealthReport, FixtureDriver
from .streams import SampleStream
from .synchronization import WindowSynchronizer
from .windows import WindowFeatures

__version__ = "0.1.0"

__all__ = [
    # entry points
    "local", "Client",
    # contracts
    "Action", "ActionResult", "ActionStatus", "Capture", "Deployment",
    "DeploymentState", "Device", "ModalityState", "ModelInput",
    "ModelManifest", "Prediction", "RetryPolicy", "Sensor", "SensorSample",
    "SensorWindow",
    # devices
    "DeviceHandle", "SensorHandle", "LocalDevice", "LanDevice", "RemoteDevice",
    # sensors
    "SensorDriver", "SensorMeta", "HealthReport", "FixtureDriver",
    # streams/windows
    "SampleStream", "WindowSynchronizer", "WindowFeatures",
    # processors
    "Processor", "ProcessorMeta", "RuleProcessor", "TorchScriptProcessor",
    "FusionProcessor", "create_processor",
    # actuators
    "Actuator", "DeviceActuator", "HomeAssistantActuator", "WebhookActuator",
    "create_actuator",
    # errors
    "WhispyError", "AuthError", "EntitlementError", "NotFoundError", "APIError",
    "__version__",
]
