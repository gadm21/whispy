"""Rigorous tests for whispy.device — DeviceInfo, ReceiverInfo, DeviceRegistry."""
import json
import tempfile
import threading
import pytest

from whispy.device import DeviceInfo, ReceiverInfo, DeviceRegistry


# ── ReceiverInfo ────────────────────────────────────────────
class TestReceiverInfo:
    def test_defaults(self):
        r = ReceiverInfo(port="/dev/ttyUSB0")
        assert r.port == "/dev/ttyUSB0"
        assert r.chip == "ESP32-C6"
        assert r.baud == 115200
        assert r.status == "unknown"

    def test_to_dict(self):
        r = ReceiverInfo(port="COM5", chip="ESP32-S3", mac="AA:BB:CC:DD:EE:FF")
        d = r.to_dict()
        assert d["port"] == "COM5"
        assert d["chip"] == "ESP32-S3"
        assert d["mac"] == "AA:BB:CC:DD:EE:FF"

    def test_from_dict(self):
        d = {"port": "/dev/ttyUSB1", "chip": "ESP32-C3", "baud": 921600}
        r = ReceiverInfo.from_dict(d)
        assert r.port == "/dev/ttyUSB1"
        assert r.chip == "ESP32-C3"
        assert r.baud == 921600

    def test_from_dict_extra_keys_ignored(self):
        d = {"port": "COM3", "unknown_key": "ignored"}
        r = ReceiverInfo.from_dict(d)
        assert r.port == "COM3"

    def test_roundtrip(self):
        r = ReceiverInfo(port="COM5", chip="ESP32-C6", mac="AA", firmware="1.0",
                         baud=921600, status="active", csi_rate=150.0, last_seen="2025-01-01")
        r2 = ReceiverInfo.from_dict(r.to_dict())
        assert r == r2


# ── DeviceInfo ──────────────────────────────────────────────
class TestDeviceInfo:
    def test_minimal(self):
        d = DeviceInfo(node_id="test-01")
        assert d.node_id == "test-01"
        assert d.status == "offline"
        assert d.latitude == 0.0
        assert d.receivers == []

    def test_full_construction(self):
        d = DeviceInfo(
            node_id="lab-01", name="Lab 1", location="Toronto",
            latitude=43.65, longitude=-79.38, task="occupancy",
            tags=["research"], labels=["empty", "occupied"],
        )
        assert d.location == "Toronto"
        assert d.latitude == pytest.approx(43.65)
        assert d.tags == ["research"]
        assert d.labels == ["empty", "occupied"]

    def test_add_receiver(self):
        d = DeviceInfo(node_id="test-01")
        r = ReceiverInfo(port="COM5")
        d.add_receiver(r)
        assert len(d.receivers) == 1
        assert d.receivers[0].port == "COM5"

    def test_to_dict(self):
        d = DeviceInfo(node_id="test-01", location="Berlin", latitude=52.52)
        d.add_receiver(ReceiverInfo(port="COM3"))
        out = d.to_dict()
        assert out["node_id"] == "test-01"
        assert out["location"] == "Berlin"
        assert len(out["receivers"]) == 1
        assert out["receivers"][0]["port"] == "COM3"

    def test_from_dict(self):
        raw = {
            "node_id": "office-01",
            "location": "NYC",
            "latitude": 40.71,
            "longitude": -74.01,
            "task": "har",
            "receivers": [{"port": "/dev/ttyUSB0", "chip": "ESP32-C6"}],
        }
        d = DeviceInfo.from_dict(raw)
        assert d.node_id == "office-01"
        assert d.location == "NYC"
        assert d.latitude == pytest.approx(40.71)
        assert d.task == "har"
        assert len(d.receivers) == 1

    def test_from_dict_extra_keys(self):
        raw = {"node_id": "x", "unknown_field": 42}
        d = DeviceInfo.from_dict(raw)
        assert d.node_id == "x"

    def test_roundtrip(self):
        d = DeviceInfo(
            node_id="rpi-01", name="Pi1", location="London",
            latitude=51.5, longitude=-0.12, task="localization",
            tags=["test"], labels=["a", "b", "c"],
        )
        d.add_receiver(ReceiverInfo(port="COM1", chip="ESP32-S3"))
        d2 = DeviceInfo.from_dict(d.to_dict())
        assert d2.node_id == d.node_id
        assert d2.location == d.location
        assert d2.latitude == d.latitude
        assert len(d2.receivers) == 1
        assert d2.receivers[0].chip == "ESP32-S3"

    def test_to_json(self):
        d = DeviceInfo(node_id="json-test")
        j = d.to_json()
        parsed = json.loads(j)
        assert parsed["node_id"] == "json-test"

    def test_summary(self):
        d = DeviceInfo(node_id="summary-test", location="Lab", latitude=1.0, longitude=2.0)
        s = d.summary()
        assert "summary-test" in s
        assert "Lab" in s

    def test_from_system(self):
        d = DeviceInfo.from_system(
            node_id="sys-test",
            location="Here",
            latitude=10.0,
            longitude=20.0,
            auto_discover_receivers=False,
        )
        assert d.node_id == "sys-test"
        assert d.location == "Here"
        assert d.status == "online"
        assert d.hostname != ""
        assert d.python_version != ""
        assert d.registered_at != ""


# ── DeviceRegistry ──────────────────────────────────────────
class TestDeviceRegistry:
    def _make_device(self, node_id, location="Test", task="occupancy"):
        return DeviceInfo(node_id=node_id, location=location, task=task, status="online")

    def test_empty_registry(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        assert reg.count == 0
        assert reg.list() == []

    def test_upsert_and_get(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        dev = self._make_device("node-1")
        reg.upsert(dev)
        assert reg.count == 1
        got = reg.get("node-1")
        assert got is not None
        assert got.node_id == "node-1"

    def test_upsert_updates_existing(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        dev1 = self._make_device("node-1", location="A")
        reg.upsert(dev1)
        dev2 = self._make_device("node-1", location="B")
        reg.upsert(dev2)
        assert reg.count == 1
        assert reg.get("node-1").location == "B"

    def test_upsert_preserves_registered_at(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        dev1 = DeviceInfo(node_id="n1", registered_at="2025-01-01T00:00:00")
        reg.upsert(dev1)
        dev2 = DeviceInfo(node_id="n1", registered_at="2026-01-01T00:00:00")
        reg.upsert(dev2)
        assert reg.get("n1").registered_at == "2025-01-01T00:00:00"

    def test_remove(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        reg.upsert(self._make_device("node-1"))
        assert reg.remove("node-1") is True
        assert reg.count == 0
        assert reg.get("node-1") is None

    def test_remove_nonexistent(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        assert reg.remove("ghost") is False

    def test_list(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        for i in range(5):
            reg.upsert(self._make_device(f"node-{i}"))
        assert reg.count == 5
        assert len(reg.list()) == 5

    def test_find_by_location(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        reg.upsert(self._make_device("a", location="Toronto"))
        reg.upsert(self._make_device("b", location="London"))
        reg.upsert(self._make_device("c", location="Toronto"))
        found = reg.find(location="Toronto")
        assert len(found) == 2

    def test_find_by_task(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        reg.upsert(self._make_device("a", task="occupancy"))
        reg.upsert(self._make_device("b", task="har"))
        assert len(reg.find(task="har")) == 1

    def test_find_multi_filter(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        reg.upsert(self._make_device("a", location="X", task="occupancy"))
        reg.upsert(self._make_device("b", location="X", task="har"))
        reg.upsert(self._make_device("c", location="Y", task="occupancy"))
        assert len(reg.find(location="X", task="occupancy")) == 1

    def test_online(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        d1 = self._make_device("a")
        d1.status = "online"
        d2 = self._make_device("b")
        d2.status = "offline"
        reg.upsert(d1)
        reg.upsert(d2)
        assert len(reg.online()) == 1

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg1 = DeviceRegistry(path)
        reg1.upsert(self._make_device("persist-test", location="Mars"))
        # create a new registry from the same file
        reg2 = DeviceRegistry(path)
        assert reg2.count == 1
        assert reg2.get("persist-test").location == "Mars"

    def test_summary(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        reg.upsert(self._make_device("s1"))
        s = reg.summary()
        assert "s1" in s
        assert "1 devices" in s

    def test_thread_safety(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DeviceRegistry(path)
        errors = []

        def writer(start):
            try:
                for i in range(20):
                    reg.upsert(self._make_device(f"thread-{start}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(j,)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert reg.count == 100
