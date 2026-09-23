"""Stream buffers, window synchronization, and feature extraction."""
import time

import pytest

from whispy.contracts import SensorSample
from whispy.streams import SampleStream
from whispy.synchronization import WindowSynchronizer
from whispy.windows import WindowFeatures
from whispy.contracts import SensorWindow, ModalityState


def _sample(sensor_id, ts, seq, payload):
    return SensorSample(device_id="d1", sensor_id=sensor_id,
                        sensor_type=sensor_id.split("-")[0],
                        timestamp=ts, sequence=seq,
                        payload_type="list", payload=payload)


def test_stream_bounded_buffer():
    stream = SampleStream(iter([]), maxlen=3)
    for i in range(5):
        stream.put(_sample("radar-0", float(i), i, [i]))
    snap = stream.snapshot()
    assert len(snap) == 3
    assert stream.dropped == 2
    assert snap[-1].sequence == 4


def test_stream_window_slice():
    stream = SampleStream(iter([]))
    for i in range(10):
        stream.put(_sample("radar-0", float(i), i, [i]))
    chunk = stream.window(3.0, 6.0)
    assert [s.sequence for s in chunk] == [3, 4, 5, 6]


def test_stream_subscriptions_are_non_destructive():
    """Two consumers each get every sample; the shared buffer is untouched."""
    stream = SampleStream(iter([]))
    sub_a = stream.subscribe()
    sub_b = stream.subscribe()
    for i in range(3):
        stream.put(_sample("radar-0", float(i), i, [i]))

    # Each subscription sees all samples independently.
    assert [s.sequence for s in sub_a.read()] == [0, 1, 2]
    assert [s.sequence for s in sub_b.read()] == [0, 1, 2]
    # Reading a subscription does not drain the shared buffer.
    assert len(stream.snapshot()) == 3
    # A second read returns only new samples.
    stream.put(_sample("radar-0", 3.0, 3, [3]))
    assert [s.sequence for s in sub_a.read()] == [3]


def test_stream_subscription_starts_empty():
    stream = SampleStream(iter([]))
    stream.put(_sample("radar-0", 0.0, 0, [0]))
    sub = stream.subscribe()          # created after a sample was buffered
    assert sub.read() == []           # no replay of pre-subscription samples
    stream.put(_sample("radar-0", 1.0, 1, [1]))
    assert [s.sequence for s in sub.read()] == [1]


def test_synchronizer_marks_missing_and_stale():
    radar = SampleStream(iter([]))
    now = time.time()
    radar.put(_sample("radar-0", now - 0.1, 0, [1.0]))
    env = SampleStream(iter([]))
    env.put(_sample("env-0", now - 100.0, 0, [22.0]))  # stale

    sync = WindowSynchronizer(
        {"radar-0": radar, "env-0": env},
        expected=["radar-0", "env-0", "csi-0"], stale_after_s=5.0)
    w = sync.cut(now - 1.0, now, now=now)

    assert w.modalities["radar-0"].state == "ok"
    assert w.modalities["env-0"].state == "stale"
    assert w.modalities["csi-0"].state == "missing"
    assert w.missing() == ["csi-0"]
    assert w.stale() == ["env-0"]
    assert not w.is_complete()


def test_window_features_stats():
    samples = [_sample("radar-0", float(i), i, [float(i)]) for i in range(4)]
    w = SensorWindow(start_timestamp=0, end_timestamp=3,
                     samples={"radar": samples})
    f = WindowFeatures(w)
    assert f.feature("radar_mean") == pytest.approx(1.5)
    assert f.feature("radar_max") == pytest.approx(3.0)
    assert f.feature("radar_energy") == pytest.approx((0 + 1 + 4 + 9) / 4)
    assert f.feature("radar_rms") == pytest.approx((14 / 4) ** 0.5)
    with pytest.raises(KeyError):
        f.feature("nonexistent_mean")


def test_window_features_snr_and_rms_accel():
    snr = [_sample("snr-0", 0.0, i, [10.0 + i]) for i in range(4)]
    imu = [_sample("imu-0", 0.0, i, [1.0, 2.0, 2.0]) for i in range(2)]
    w = SensorWindow(start_timestamp=0, end_timestamp=1,
                     samples={"snr": snr, "imu": imu})
    f = WindowFeatures(w)
    assert f.feature("snr_mean") == pytest.approx(11.5)
    assert f.feature("rms_accel") == pytest.approx(3.0)  # sqrt(1+4+4)


def test_feature_from_sample_metadata():
    s = _sample("radar-0", 0.0, 0, [1.0])
    s.metadata["snr_db"] = 14.5
    w = SensorWindow(start_timestamp=0, end_timestamp=1,
                     samples={"radar": [s]})
    assert WindowFeatures(w).feature("snr_db") == pytest.approx(14.5)
