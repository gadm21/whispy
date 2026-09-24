"""SensorAdapter contract + legacy-driver compat + descriptor identity."""
import itertools

import pytest

from whispy.contracts import SensorDescriptor, SensorSample
from whispy.devices.local import LocalDevice
from whispy.sensors import (
    FixtureDriver, SensorAdapter, SensorDriverAdapter, SensorMeta,
)
from whispy.sensors.base import _DriverSensorHandle


class _MultiCamDriver:
    """Legacy-style driver reporting two cameras with hardware ids."""

    def metadata(self):
        return SensorMeta(name="multicam", modalities=("camera",))

    def discover(self):
        return [{"id": "cam-a", "hardware_id": "USB\\VID_1&PID_A\\AAA"},
                {"id": "cam-b", "hardware_id": "USB\\VID_2&PID_B\\BBB"}]

    def open(self, config=None):
        self._open = True

    def stream(self):
        for i in itertools.count():
            yield SensorSample.now("local", "camera-0", "camera", [i],
                                   sequence=i)

    def close(self):
        self._open = False


def test_driver_adapter_stable_ids_from_hardware_id():
    adapter = SensorDriverAdapter(_MultiCamDriver(), name="multicam")
    descs = adapter.discover()
    assert len(descs) == 2
    ids = {d.id for d in descs}
    assert all(d.id.startswith("camera-") for d in descs)
    assert all(d.stable for d in descs)
    assert len(ids) == 2                     # distinct hardware → distinct ids
    # Same hardware id → same descriptor id across discoveries.
    again = {d.id for d in adapter.discover()}
    assert ids == again


def test_driver_adapter_legacy_id_preserved_single_modality():
    """A single-modality driver keeps its historical ``<mod>-0`` id."""
    from whispy.sensors.system import SystemTelemetryDriver
    adapter = SensorDriverAdapter(SystemTelemetryDriver(), name="system")
    descs = adapter.discover()
    assert [d.id for d in descs] == ["system-0"]
    assert descs[0].modality == "system"


def test_driver_adapter_connect_streams_samples():
    adapter = SensorDriverAdapter(_MultiCamDriver(), name="multicam")
    desc = adapter.discover()[0]
    handle = adapter.connect(desc)
    assert isinstance(handle, _DriverSensorHandle)
    sample = next(handle.stream(max_samples=1))
    assert isinstance(sample, SensorSample)
    handle.close()


def test_local_device_inventory_is_physical_instances():
    dev = LocalDevice(device_id="d1",
                      adapters={"multicam": SensorDriverAdapter(
                          _MultiCamDriver(), name="multicam")})
    ids = [s.id for s in dev.sensors()]
    assert len(ids) == 2
    assert all(i.startswith("camera-") for i in ids)


def test_local_device_modality_ambiguity_raises():
    """sensor('camera') must fail when two cameras exist."""
    dev = LocalDevice(device_id="d1",
                      adapters={"multicam": SensorDriverAdapter(
                          _MultiCamDriver(), name="multicam")})
    with pytest.raises(KeyError) as ei:
        dev.sensor("camera")
    assert "ambiguous" in str(ei.value)


def test_local_device_resolves_stable_id_and_name():
    dev = LocalDevice(device_id="d1",
                      adapters={"multicam": SensorDriverAdapter(
                          _MultiCamDriver(), name="multicam")})
    descs = dev.sensor_descriptors()
    handle = dev.sensor(descs[1].id)
    assert handle.info.id == descs[1].id
    sample = next(handle.stream(max_samples=1))
    assert sample.device_id == "d1"


def test_local_device_single_modality_resolves():
    fixture = FixtureDriver()
    dev = LocalDevice(device_id="d1", drivers={"fixture": fixture})
    dev.open({"fixture": {"sensor_type": "radar", "payloads": [[1.0]]}})
    try:
        # FixtureDriver advertises many modalities → ids radar-0, csi-0…
        handle = dev.sensor("radar")
        sample = next(handle.stream(max_samples=1))
        assert sample.device_id == "d1"
    finally:
        dev.close()


def test_sensor_descriptor_roundtrip():
    d = SensorDescriptor(id="camera-f91a", modality="camera",
                         adapter="opencv-camera", name="Integrated Camera",
                         hardware_id="USB\\VID", capabilities=["jpeg"],
                         stable=True)
    d2 = SensorDescriptor.from_dict(d.to_dict())
    assert d2.id == "camera-f91a"
    assert d2.stable
    s = d.to_sensor()
    assert s.type == "camera" and s.metadata["hardware_id"] == "USB\\VID"
