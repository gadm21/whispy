"""ThothCraft SDK — devices, minutes, sensors, datasets, research workflows.

    from thothcraft import Client
    thoth = Client.login()
    for device in thoth.devices():
        for minute in device.data():
            print(minute.predictions)
"""

from .client import Client
from .devices import Device, CollectionSession
from .minutes import Minute, SensorData
from .models import Model, Deployment
from .datasets import Dataset
from .spaces import Space, Zone
from .local import LocalDevice, local
from .errors import (
    ThothError,
    AuthError,
    EntitlementError,
    NotFoundError,
    APIError,
)

__version__ = "0.1.0"

__all__ = [
    "Client",
    "Model",
    "Deployment",
    "Dataset",
    "Device",
    "CollectionSession",
    "Minute",
    "SensorData",
    "Space",
    "Zone",
    "LocalDevice",
    "local",
    "ThothError",
    "AuthError",
    "EntitlementError",
    "NotFoundError",
    "APIError",
    "__version__",
]
