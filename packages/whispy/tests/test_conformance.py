"""Conformance checks — check_sensor_adapter / check_model_plugin /
check_actuator_adapter."""
import itertools

from whispy.conformance import (
    check_actuator_adapter, check_model_plugin, check_sensor_adapter,
)
from whispy.contracts import (
    ActionResult, ActionStatus, ActuatorCommand, ActuatorDescriptor,
    Prediction, SensorDescriptor, SensorSample,
)
from whispy.sensors import FixtureDriver, SensorDriverAdapter
from whispy.sensors.base import SensorAdapter, SensorMeta
from whispy.actuators.base import (
    ActuatorAdapter, ActuatorHandle, ActuatorMeta,
)
from whispy.processors.base import Processor, ProcessorMeta


class _GoodSensorAdapter(SensorAdapter):
    def metadata(self):
        return SensorMeta(name="good", modalities=("fixture",))

    def discover(self):
        return [SensorDescriptor(id="fixture-0", modality="fixture",
                                 adapter="good")]

    def connect(self, descriptor, config=None):
        drv = FixtureDriver()
        drv.open({"sensor_type": "fixture", "payloads": [[1.0]]})
        return SensorDriverAdapter(drv).connect(descriptor)


class _EmptySensorAdapter(SensorAdapter):
    def metadata(self):
        return SensorMeta(name="empty", modalities=("camera",))

    def discover(self):
        return []

    def connect(self, descriptor, config=None):
        raise RuntimeError("no hardware")


class _GoodProcessor(Processor):
    def __init__(self, config=None):
        pass

    def metadata(self):
        return ProcessorMeta(name="good-model", processor_type="good")

    def predict(self, window):
        return Prediction(label="ok", confidence=0.9)


class _BadProcessor(Processor):
    def __init__(self, config=None):
        pass

    def metadata(self):
        return ProcessorMeta(name="bad-model", processor_type="bad")

    def predict(self, window):
        raise RuntimeError("inference exploded")


class _GoodActuatorAdapter(ActuatorAdapter):
    class _H(ActuatorHandle):
        @property
        def info(self):
            return ActuatorDescriptor(id="matrix-0", kind="matrix",
                                      operations=["clear"])

        def execute(self, command):
            return ActionResult(status=ActionStatus.SUCCEEDED,
                                action_type="matrix")

    def metadata(self):
        return ActuatorMeta(name="good-act", kinds=("matrix",))

    def discover(self):
        return [ActuatorDescriptor(id="matrix-0", kind="matrix",
                                   adapter="good-act",
                                   operations=["clear"])]

    def connect(self, descriptor, config=None):
        return self._H()


def test_check_sensor_adapter_passes():
    report = check_sensor_adapter(_GoodSensorAdapter(), max_samples=2)
    assert report["passed"], report


def test_check_sensor_adapter_no_hardware():
    """No discovered hardware → discovery passes, stream skipped."""
    report = check_sensor_adapter(_EmptySensorAdapter())
    assert report["passed"]
    assert report["skipped"]


def test_check_sensor_adapter_rejects_non_adapter():
    report = check_sensor_adapter(object())
    assert not report["passed"]


def test_check_model_plugin_passes():
    report = check_model_plugin(_GoodProcessor)
    assert report["passed"], report


def test_check_model_plugin_fails_on_raise():
    report = check_model_plugin(_BadProcessor)
    assert not report["passed"]
    assert any(c["name"] == "predict" and not c["ok"]
               for c in report["checks"])


def test_check_actuator_adapter_passes():
    report = check_actuator_adapter(_GoodActuatorAdapter(), probe=True)
    assert report["passed"], report


def test_check_actuator_adapter_rejects_non_adapter():
    report = check_actuator_adapter(object())
    assert not report["passed"]
