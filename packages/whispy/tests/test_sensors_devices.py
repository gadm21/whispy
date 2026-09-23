"""Sensor driver conformance + local device tests."""
import itertools

import pytest

from whispy import local
from whispy.contracts import SensorSample
from whispy.devices.local import LocalDevice
from whispy.sensors import FixtureDriver, check_driver
from whispy.sensors.system import SystemTelemetryDriver


def test_fixture_driver_conformance():
    driver = FixtureDriver()
    driver.open({"sensor_type": "radar",
                 "payloads": [[1.0, 2.0], [3.0, 4.0]],
                 "sample_rate": 100})
    report = check_driver(driver, max_samples=4)
    assert report["passed"], report


def test_fixture_driver_real_samples():
    driver = FixtureDriver()
    driver.open({"sensor_type": "csi", "payloads": [[42.0]]})
    samples = list(itertools.islice(driver.stream(), 3))
    assert all(isinstance(s, SensorSample) for s in samples)
    assert [s.sequence for s in samples] == [0, 1, 2]
    assert all(s.payload == [42.0] for s in samples)
    driver.close()


def test_system_driver_produces_real_measurements():
    driver = SystemTelemetryDriver()
    driver.open({"sample_rate": 100})
    sample = next(driver.stream())
    assert isinstance(sample.payload, dict)
    assert "cpu_percent" in sample.payload
    driver.close()


def test_local_device_with_fixture():
    fixture = FixtureDriver()
    dev = LocalDevice(device_id="test-dev", drivers={"fixture": fixture})
    dev.open({"fixture": {"sensor_type": "radar", "payloads": [[1.0]]}})
    try:
        sensors = dev.sensors()
        assert sensors
        handle = dev.sensor("fixture")
        sample = next(handle.stream(max_samples=1))
        assert sample.device_id == "test-dev"
    finally:
        dev.close()


def test_local_device_missing_sensor():
    dev = LocalDevice(device_id="test-dev", drivers={})
    with pytest.raises(KeyError):
        dev.sensor("radar")


def test_local_device_info():
    dev = LocalDevice(device_id="test-dev", drivers={})
    info = dev.info
    assert info.id == "test-dev"
    assert info.online
