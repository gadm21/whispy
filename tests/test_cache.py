"""Rigorous tests for whispy.watchdog.RollingCache and export_cache."""
import os
import tempfile
import threading
import numpy as np
import pytest

from whispy.watchdog import RollingCache, export_cache, _PACKET_BYTES
from whispy.core import N_SUBCARRIERS_RAW


# ── helpers ─────────────────────────────────────────────────
def _make_packet(ts: int = 0, rssi: float = -50.0):
    """Create a fake CSI packet with known values."""
    real = np.full(N_SUBCARRIERS_RAW, float(ts), dtype=np.float64)
    imag = np.full(N_SUBCARRIERS_RAW, rssi, dtype=np.float64)
    return ts, rssi, real, imag


def _push_n(cache: RollingCache, n: int, start_ts: int = 0):
    """Push n packets with sequential timestamps."""
    for i in range(n):
        cache.push(*_make_packet(ts=start_ts + i, rssi=-50.0 + i * 0.1))


# ── basic construction ──────────────────────────────────────
class TestRollingCacheConstruction:
    def test_default_1gb(self):
        cache = RollingCache()
        assert cache.capacity == 1 * 1024**3 // _PACKET_BYTES
        assert cache.size == 0
        assert cache.total_pushed == 0
        assert cache.used_bytes == 0

    def test_custom_size(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        assert cache.capacity == 10

    def test_minimum_one_packet(self):
        cache = RollingCache(max_bytes=1)
        assert cache.capacity == 1

    def test_zero_bytes_gives_one(self):
        cache = RollingCache(max_bytes=0)
        assert cache.capacity == 1


# ── push and read ───────────────────────────────────────────
class TestPushAndRead:
    def test_push_single(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        cache.push(*_make_packet(ts=42, rssi=-60.0))
        assert cache.size == 1
        assert cache.total_pushed == 1
        data = cache.latest()
        assert data["timestamp"][0] == 42
        assert data["rssi"][0] == -60.0

    def test_push_fill_exactly(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        _push_n(cache, 5)
        assert cache.size == 5
        assert cache.total_pushed == 5
        data = cache.latest()
        np.testing.assert_array_equal(data["timestamp"], [0, 1, 2, 3, 4])

    def test_push_overflow_wraps(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 3)
        _push_n(cache, 7)
        assert cache.size == 3
        assert cache.total_pushed == 7
        data = cache.latest()
        np.testing.assert_array_equal(data["timestamp"], [4, 5, 6])

    def test_push_large_overflow(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 4)
        _push_n(cache, 100)
        assert cache.size == 4
        data = cache.latest()
        np.testing.assert_array_equal(data["timestamp"], [96, 97, 98, 99])

    def test_latest_n_less_than_size(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        _push_n(cache, 10)
        data = cache.latest(3)
        assert len(data["timestamp"]) == 3
        np.testing.assert_array_equal(data["timestamp"], [7, 8, 9])

    def test_latest_n_greater_than_size(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        _push_n(cache, 3)
        data = cache.latest(100)
        assert len(data["timestamp"]) == 3

    def test_latest_empty_cache(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        data = cache.latest()
        assert len(data["timestamp"]) == 0
        assert data["real"].shape == (0, N_SUBCARRIERS_RAW)

    def test_latest_n_after_overflow(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        _push_n(cache, 12)
        data = cache.latest(3)
        np.testing.assert_array_equal(data["timestamp"], [9, 10, 11])

    def test_real_imag_data_integrity(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 3)
        for i in range(5):
            real = np.arange(N_SUBCARRIERS_RAW, dtype=np.float64) + i * 100
            imag = np.arange(N_SUBCARRIERS_RAW, dtype=np.float64) + i * 200
            cache.push(i, -50.0, real, imag)
        data = cache.latest()
        # last 3: ts=2,3,4
        assert data["real"][0, 0] == pytest.approx(200.0)  # ts=2: 2*100
        assert data["real"][2, 0] == pytest.approx(400.0)  # ts=4: 4*100
        assert data["imag"][2, 0] == pytest.approx(800.0)  # ts=4: 4*200

    def test_latest_returns_copies(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        _push_n(cache, 3)
        data1 = cache.latest()
        data2 = cache.latest()
        data1["timestamp"][0] = 9999
        assert data2["timestamp"][0] != 9999


# ── latest_seconds ──────────────────────────────────────────
class TestLatestSeconds:
    def test_latest_seconds(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 1000)
        _push_n(cache, 600)
        data = cache.latest_seconds(2.0, sr=150)
        assert len(data["timestamp"]) == 300

    def test_latest_seconds_more_than_available(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 100)
        _push_n(cache, 50)
        data = cache.latest_seconds(60.0, sr=150)
        assert len(data["timestamp"]) == 50


# ── clear ───────────────────────────────────────────────────
class TestClear:
    def test_clear_resets(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        _push_n(cache, 7)
        assert cache.size == 7
        cache.clear()
        assert cache.size == 0
        assert cache.total_pushed == 0
        data = cache.latest()
        assert len(data["timestamp"]) == 0

    def test_push_after_clear(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        _push_n(cache, 5)
        cache.clear()
        _push_n(cache, 3, start_ts=100)
        assert cache.size == 3
        data = cache.latest()
        np.testing.assert_array_equal(data["timestamp"], [100, 101, 102])


# ── resize ──────────────────────────────────────────────────
class TestResize:
    def test_resize_larger(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        _push_n(cache, 5)
        cache.resize(_PACKET_BYTES * 10)
        assert cache.capacity == 10
        assert cache.size == 5
        data = cache.latest()
        np.testing.assert_array_equal(data["timestamp"], [0, 1, 2, 3, 4])

    def test_resize_smaller_truncates(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        _push_n(cache, 10)
        cache.resize(_PACKET_BYTES * 3)
        assert cache.capacity == 3
        assert cache.size == 3
        data = cache.latest()
        np.testing.assert_array_equal(data["timestamp"], [7, 8, 9])

    def test_resize_after_overflow(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        _push_n(cache, 12)  # overflows, last 5: 7,8,9,10,11
        cache.resize(_PACKET_BYTES * 3)
        data = cache.latest()
        np.testing.assert_array_equal(data["timestamp"], [9, 10, 11])

    def test_push_after_resize(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        _push_n(cache, 5)
        cache.resize(_PACKET_BYTES * 3)
        _push_n(cache, 2, start_ts=100)
        data = cache.latest()
        # 3 slots: had [3,4], now added [100,101] => [4, 100, 101]
        np.testing.assert_array_equal(data["timestamp"], [4, 100, 101])


# ── used_bytes / max_bytes ──────────────────────────────────
class TestBytes:
    def test_used_bytes(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        _push_n(cache, 3)
        assert cache.used_bytes == 3 * _PACKET_BYTES

    def test_max_bytes(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        assert cache.max_bytes == 10 * _PACKET_BYTES


# ── repr ────────────────────────────────────────────────────
class TestRepr:
    def test_repr_format(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 100)
        _push_n(cache, 50)
        s = repr(cache)
        assert "RollingCache" in s
        assert "50/100" in s


# ── thread safety ───────────────────────────────────────────
class TestThreadSafety:
    def test_concurrent_push(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 1000)
        errors = []

        def pusher(start):
            try:
                for i in range(200):
                    cache.push(*_make_packet(ts=start + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher, args=(i * 1000,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.total_pushed == 1000
        assert cache.size == 1000

    def test_concurrent_push_and_read(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 100)
        errors = []

        def pusher():
            try:
                for i in range(500):
                    cache.push(*_make_packet(ts=i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    data = cache.latest(10)
                    assert len(data["timestamp"]) <= 10
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=pusher)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0


# ── export_cache ────────────────────────────────────────────
class TestExportCache:
    def test_export_empty(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 10)
        with tempfile.TemporaryDirectory() as td:
            files = export_cache(cache, out_dir=td)
            assert files == []

    def test_export_single_file(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 100)
        _push_n(cache, 50)
        with tempfile.TemporaryDirectory() as td:
            files = export_cache(cache, out_dir=td, n_files=1)
            assert len(files) == 1
            assert files[0].endswith(".npz")
            assert os.path.isfile(files[0])
            # verify contents
            data = np.load(files[0])
            assert "timestamp" in data
            assert "rssi" in data
            assert "magnitude" in data
            assert len(data["timestamp"]) == 50

    def test_export_multiple_files(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 300)
        _push_n(cache, 300)
        with tempfile.TemporaryDirectory() as td:
            files = export_cache(cache, out_dir=td, n_files=3)
            assert len(files) == 3
            total = sum(len(np.load(f)["timestamp"]) for f in files)
            assert total == 300

    def test_export_minutes(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 1000)
        _push_n(cache, 900)
        with tempfile.TemporaryDirectory() as td:
            files = export_cache(cache, out_dir=td, minutes=1.0, sr=150)
            assert len(files) == 1
            data = np.load(files[0])
            # 1 minute * 150 Hz = 150 packets (capped by what's in the cache)
            assert len(data["timestamp"]) == 150

    def test_export_metadata_file(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 50)
        _push_n(cache, 50)
        with tempfile.TemporaryDirectory() as td:
            files = export_cache(cache, out_dir=td, n_files=1, label="test_label")
            meta_path = files[0].replace(".npz", ".meta.json")
            assert os.path.isfile(meta_path)
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["label"] == "test_label"
            assert meta["samples"] == 50


# ── push_parsed ─────────────────────────────────────────────
class TestPushParsed:
    def test_push_parsed_none(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        cache.push_parsed(None)
        assert cache.size == 0

    def test_push_parsed_dict(self):
        cache = RollingCache(max_bytes=_PACKET_BYTES * 5)
        parsed = {
            "timestamp": 123,
            "rssi": -45.0,
            "real": np.zeros(N_SUBCARRIERS_RAW),
            "imag": np.ones(N_SUBCARRIERS_RAW),
        }
        cache.push_parsed(parsed)
        assert cache.size == 1
        data = cache.latest()
        assert data["timestamp"][0] == 123
        assert data["rssi"][0] == -45.0
        np.testing.assert_array_equal(data["imag"][0], np.ones(N_SUBCARRIERS_RAW))
