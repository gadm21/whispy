"""Processor + sensor-driver interface tests."""
import numpy as np
import pytest

from thothcraft.processors import (
    Prediction, ProcessorMeta, RuleProcessor, SensorWindow,
)
from thothcraft.sensors.base import (
    HealthReport, SensorDriver, SensorFrame, SensorMeta, check_driver,
)


# ── SensorWindow ─────────────────────────────────────────────────────────────

def test_window_numpy_and_features():
    w = SensorWindow({'radar': [[1, 2], [3, 4]], 'snr': [10.0, 14.0]})
    assert 'radar' in w
    assert w['radar'].shape == (2, 2)
    assert w.feature('snr_mean') == pytest.approx(12.0)
    assert w.feature('radar_energy') == pytest.approx(7.5)
    with pytest.raises(KeyError):
        w.feature('missing_mean')


def test_window_snr_mean_fallback():
    w = SensorWindow({'radar': [5.0, 15.0]})
    assert w.feature('snr_mean') == pytest.approx(10.0)


# ── RuleProcessor ────────────────────────────────────────────────────────────

def _snr_rule(threshold=12.0):
    return RuleProcessor({
        'name': 'snr-occupancy',
        'rules': [{'when': 'snr_mean > snr_threshold', 'label': 'occupied',
                   'confidence': 0.9}],
        'else': 'empty',
        'params': {'snr_threshold': threshold},
        'sensor': 'radar', 'task': 'occupancy',
    })


def test_rule_fires_above_threshold():
    proc = _snr_rule()
    pred = proc.predict(SensorWindow({'snr': [20.0, 18.0]}))
    assert pred.label == 'occupied'
    assert pred.confidence == 0.9


def test_rule_else_below_threshold():
    proc = _snr_rule()
    pred = proc.predict(SensorWindow({'snr': [1.0, 2.0]}))
    assert pred.label == 'empty'


def test_rule_configure_overrides_params():
    proc = _snr_rule()
    proc.configure({'snr_threshold': 100.0})
    pred = proc.predict(SensorWindow({'snr': [50.0]}))
    assert pred.label == 'empty'


def test_rule_metadata():
    meta = _snr_rule().metadata()
    assert meta.processor_type == 'rule'
    assert meta.sensor == 'radar'
    assert meta.to_dict()['task'] == 'occupancy'


# ── SensorDriver conformance ─────────────────────────────────────────────────

class _FakeDriver(SensorDriver):
    def metadata(self):
        return SensorMeta(name='fake', modalities=('test',))

    def discover(self):
        return [{'id': 'fake-0'}]

    def open(self, config=None):
        self._open = True

    def stream(self):
        for i in range(10):
            yield SensorFrame.now('test', np.array([i]))

    def close(self):
        self._open = False

    def health(self):
        return HealthReport(status='ok')


class _BrokenDriver(_FakeDriver):
    def stream(self):
        yield SensorFrame(sensor_type='test', timestamp_ns=-1, data=np.array([]))


def test_check_driver_passes():
    report = check_driver(_FakeDriver())
    assert report['passed'] is True
    assert all(c['ok'] for c in report['checks'])


def test_check_driver_catches_bad_frames():
    report = check_driver(_BrokenDriver())
    assert report['passed'] is False
    stream = next(c for c in report['checks'] if c['name'] == 'stream')
    assert stream['ok'] is False


# ── ROS2 dry-run bridge ──────────────────────────────────────────────────────

def test_ros2_bridge_dry_run():
    from thothcraft.integrations.ros2 import ROS2Bridge

    class _Client:
        def spaces_state(self):
            return [{'name': 'office', 'occupied': True,
                     'people_count': 2, 'confidence': 0.9}]

    bridge = ROS2Bridge(_Client(), dry_run=True)
    bridge.spin()
    occ = bridge._publishers['/thoth/occupancy'].messages[0]
    import json
    assert json.loads(occ)['occupied'] is True
    assert json.loads(occ)['space'] == 'office'
