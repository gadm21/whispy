"""Rigorous tests for whispy.cloud_client — REST client against a live test server."""
import io
import json
import tempfile
import os
import time
import threading
import pytest

from whispy.backend import create_app
from whispy.cloud_client import CloudClient


@pytest.fixture
def tmp_dirs():
    with tempfile.TemporaryDirectory() as td:
        yield {
            "data_dir": os.path.join(td, "data"),
            "db_path": os.path.join(td, "test.db"),
            "registry_path": os.path.join(td, "devices.json"),
        }


@pytest.fixture
def server(tmp_dirs):
    """Start a real uvicorn server in a background thread for integration tests."""
    import uvicorn

    app = create_app(api_key="test-key", **tmp_dirs)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)

    # Find a free port
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config.port = port
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to start
    for _ in range(50):
        try:
            from urllib.request import urlopen
            urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    server_thread.join(timeout=5)


@pytest.fixture
def client(server):
    return CloudClient(
        backend_url=server,
        api_key="test-key",
        node_id="test-node-01",
        async_push=False,  # synchronous for testing
    )


# ── construction ────────────────────────────────────────────
class TestConstruction:
    def test_defaults(self):
        c = CloudClient("http://example.com", "key", "node-1")
        assert c.backend_url == "http://example.com"
        assert c.api_key == "key"
        assert c.node_id == "node-1"
        assert c._push_count == 0
        assert c._error_count == 0

    def test_trailing_slash_stripped(self):
        c = CloudClient("http://example.com/", "key", "node-1")
        assert c.backend_url == "http://example.com"

    def test_repr(self):
        c = CloudClient("http://x.com", "k", "n")
        assert "CloudClient" in repr(c)
        assert "n" in repr(c)


# ── health ──────────────────────────────────────────────────
class TestHealth:
    def test_health(self, client):
        result = client.health()
        assert result is not None
        assert result["status"] == "ok"
        assert result["auth_required"] is True


# ── registration ────────────────────────────────────────────
class TestRegistration:
    def test_register(self, client):
        result = client.register(
            location="Toronto",
            latitude=43.65,
            longitude=-79.38,
            task="occupancy",
        )
        assert result is not None
        assert result["status"] == "registered"
        assert result["node_id"] == "test-node-01"

    def test_register_appears_in_devices(self, client):
        client.register(location="Berlin")
        devices = client.get_devices()
        assert devices is not None
        assert len(devices) >= 1
        node_ids = [d["node_id"] for d in devices]
        assert "test-node-01" in node_ids


# ── prediction push ─────────────────────────────────────────
class TestPredictionPush:
    def test_push_prediction(self, client):
        client.push_prediction(
            label="occupied",
            class_idx=1,
            confidence=0.94,
            probabilities={"empty": 0.06, "occupied": 0.94},
        )
        assert client._push_count >= 1
        assert client._error_count == 0

    def test_push_prediction_queryable(self, client):
        client.push_prediction(label="test-label", class_idx=0, confidence=0.8)
        # slight delay for DB write
        time.sleep(0.1)
        preds = client.get_predictions(limit=10)
        assert preds is not None
        assert len(preds) >= 1
        assert any(p["label"] == "test-label" for p in preds)

    def test_push_multiple_predictions(self, client):
        for i in range(10):
            client.push_prediction(label=f"l-{i}", class_idx=i, confidence=0.5)
        assert client._push_count >= 10
        preds = client.get_predictions(limit=100)
        assert len(preds) >= 10


# ── diagnostics push ────────────────────────────────────────
class TestDiagnosticsPush:
    def test_push_diagnostics(self, client):
        client.push_diagnostics(
            csi_rate=148.5,
            cpu_temp=52.3,
            esp32_alive=True,
            cache_mb=256.7,
        )
        assert client._error_count == 0

    def test_push_diagnostics_queryable(self, client):
        client.push_diagnostics(csi_rate=99.9, cpu_temp=45.0)
        time.sleep(0.1)
        diags = client.get_diagnostics(limit=10)
        assert diags is not None
        assert len(diags) >= 1


# ── error handling ──────────────────────────────────────────
class TestErrorHandling:
    def test_bad_url(self):
        c = CloudClient("http://127.0.0.1:1", "key", "node", async_push=False, timeout=2)
        result = c._post("/ingest/prediction", {"node_id": "x"})
        assert result is None
        assert c._error_count == 1
        assert len(c._errors) == 1

    def test_bad_auth(self, server):
        c = CloudClient(server, "wrong-key", "node", async_push=False)
        c.push_prediction(label="x", confidence=0.5)
        assert c._error_count == 1

    def test_stats_property(self, client):
        client.push_prediction(label="x", confidence=0.5)
        s = client.stats
        assert "pushes" in s
        assert "errors" in s
        assert s["pushes"] >= 1


# ── async mode ──────────────────────────────────────────────
class TestAsyncMode:
    def test_async_push_does_not_block(self, server):
        c = CloudClient(server, "test-key", "async-node", async_push=True)
        start = time.time()
        for _ in range(20):
            c.push_prediction(label="x", confidence=0.5)
        elapsed = time.time() - start
        # async should return nearly instantly
        assert elapsed < 2.0
        # wait for background threads
        time.sleep(2.0)
        assert c._push_count >= 10  # at least some should have completed
