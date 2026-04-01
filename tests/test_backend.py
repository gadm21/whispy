"""Rigorous tests for whispy.backend — FastAPI endpoints, auth, ingest, queries."""
import io
import json
import tempfile
import os
import pytest

from whispy.backend import create_app, TimeSeriesDB, _verify_api_key


# ── _verify_api_key ─────────────────────────────────────────
class TestVerifyApiKey:
    def test_no_key_configured_always_passes(self):
        assert _verify_api_key(None, None) is True
        assert _verify_api_key(None, "anything") is True
        assert _verify_api_key("", None) is True

    def test_key_configured_no_provided(self):
        assert _verify_api_key("secret", None) is False

    def test_key_configured_wrong_key(self):
        assert _verify_api_key("secret", "wrong") is False

    def test_key_configured_correct_key(self):
        assert _verify_api_key("secret", "secret") is True

    def test_constant_time(self):
        # just verify it doesn't crash with various inputs
        assert _verify_api_key("abc", "abc") is True
        assert _verify_api_key("abc", "abd") is False


# ── TimeSeriesDB ────────────────────────────────────────────
class TestTimeSeriesDB:
    def test_insert_and_query_predictions(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        db = TimeSeriesDB(path)
        db.insert_prediction("node-1", "occupied", 1, 0.95, {"occupied": 0.95})
        db.insert_prediction("node-1", "empty", 0, 0.85, None)
        db.insert_prediction("node-2", "occupied", 1, 0.90, None)
        rows = db.query("predictions", "node-1")
        assert len(rows) == 2
        assert rows[0]["label"] == "empty"  # DESC order
        assert rows[1]["label"] == "occupied"

    def test_insert_and_query_diagnostics(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        db = TimeSeriesDB(path)
        db.insert_diagnostic("node-1", 148.5, 52.3, True, 256.7)
        rows = db.query("diagnostics", "node-1")
        assert len(rows) == 1
        assert rows[0]["csi_rate"] == pytest.approx(148.5)
        assert rows[0]["esp32_alive"] == 1

    def test_insert_and_query_uploads(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        db = TimeSeriesDB(path)
        db.insert_upload("node-1", "data.npz", 1024, 100, 60.0, "test", "/tmp/data.npz")
        rows = db.query("uploads", "node-1")
        assert len(rows) == 1
        assert rows[0]["filename"] == "data.npz"

    def test_query_all_nodes(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        db = TimeSeriesDB(path)
        db.insert_prediction("a", "x", 0, 0.5)
        db.insert_prediction("b", "y", 1, 0.9)
        rows = db.query("predictions")
        assert len(rows) == 2

    def test_query_limit(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        db = TimeSeriesDB(path)
        for i in range(50):
            db.insert_prediction("n", f"label-{i}", i, 0.5)
        rows = db.query("predictions", limit=10)
        assert len(rows) == 10

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        db = TimeSeriesDB(path)
        db.insert_prediction("n", "a", 0, 0.5)
        db.insert_prediction("n", "b", 1, 0.5)
        db.insert_diagnostic("n", 100, 50, True, 100)
        s = db.stats()
        assert s["predictions"] == 2
        assert s["diagnostics"] == 1
        assert s["uploads"] == 0


# ── FastAPI app fixtures ────────────────────────────────────
@pytest.fixture
def tmp_dirs():
    with tempfile.TemporaryDirectory() as td:
        yield {
            "data_dir": os.path.join(td, "data"),
            "db_path": os.path.join(td, "test.db"),
            "registry_path": os.path.join(td, "devices.json"),
        }


@pytest.fixture
def app_no_auth(tmp_dirs):
    return create_app(api_key=None, **tmp_dirs)


@pytest.fixture
def app_with_auth(tmp_dirs):
    return create_app(api_key="test-secret-key", admin_key="admin-key", **tmp_dirs)


@pytest.fixture
def client_no_auth(app_no_auth):
    from fastapi.testclient import TestClient
    return TestClient(app_no_auth)


@pytest.fixture
def client_auth(app_with_auth):
    from fastapi.testclient import TestClient
    return TestClient(app_with_auth)


# ── health endpoint ─────────────────────────────────────────
class TestHealth:
    def test_health_no_auth(self, client_no_auth):
        r = client_no_auth.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["auth_required"] is False

    def test_health_with_auth_still_public(self, client_auth):
        r = client_auth.get("/health")
        assert r.status_code == 200
        assert r.json()["auth_required"] is True


# ── auth enforcement ────────────────────────────────────────
class TestAuth:
    def test_no_key_rejected(self, client_auth):
        r = client_auth.get("/devices")
        assert r.status_code == 401

    def test_wrong_key_rejected(self, client_auth):
        r = client_auth.get("/devices", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_correct_key_accepted(self, client_auth):
        r = client_auth.get("/devices", headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200

    def test_admin_endpoint_needs_admin_key(self, client_auth):
        # register a device first
        client_auth.post("/devices/register",
                         json={"node_id": "del-test"},
                         headers={"X-API-Key": "test-secret-key"})
        # delete with regular key should fail (admin_key != api_key)
        r = client_auth.delete("/devices/del-test",
                               headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 403

    def test_admin_key_works_for_delete(self, client_auth):
        client_auth.post("/devices/register",
                         json={"node_id": "del-test2"},
                         headers={"X-API-Key": "test-secret-key"})
        r = client_auth.delete("/devices/del-test2",
                               headers={"X-API-Key": "admin-key"})
        assert r.status_code == 200

    def test_no_auth_mode_all_open(self, client_no_auth):
        r = client_no_auth.get("/devices")
        assert r.status_code == 200
        r = client_no_auth.get("/predictions")
        assert r.status_code == 200


# ── REST ingest ─────────────────────────────────────────────
class TestIngest:
    def test_ingest_prediction(self, client_no_auth):
        r = client_no_auth.post("/ingest/prediction", json={
            "node_id": "node-1",
            "label": "occupied",
            "class": 1,
            "confidence": 0.94,
            "probabilities": {"empty": 0.06, "occupied": 0.94},
        })
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ingest_prediction_missing_node_id(self, client_no_auth):
        r = client_no_auth.post("/ingest/prediction", json={
            "label": "occupied", "confidence": 0.9,
        })
        assert r.status_code == 422

    def test_ingest_diagnostics(self, client_no_auth):
        r = client_no_auth.post("/ingest/diagnostics", json={
            "node_id": "node-1",
            "csi_rate": 148.5,
            "cpu_temp": 52.3,
            "esp32_alive": True,
            "cache_mb": 256.7,
        })
        assert r.status_code == 200

    def test_ingest_diagnostics_missing_node_id(self, client_no_auth):
        r = client_no_auth.post("/ingest/diagnostics", json={
            "csi_rate": 100,
        })
        assert r.status_code == 422

    def test_ingest_prediction_with_auth(self, client_auth):
        r = client_auth.post("/ingest/prediction",
                             json={"node_id": "n1", "label": "x", "confidence": 0.5},
                             headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200

    def test_ingest_prediction_auth_rejected(self, client_auth):
        r = client_auth.post("/ingest/prediction",
                             json={"node_id": "n1", "label": "x"})
        assert r.status_code == 401

    def test_ingested_data_queryable(self, client_no_auth):
        for i in range(5):
            client_no_auth.post("/ingest/prediction", json={
                "node_id": "query-test",
                "label": f"label-{i}",
                "class": i,
                "confidence": 0.5 + i * 0.1,
            })
        r = client_no_auth.get("/predictions", params={"node_id": "query-test"})
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5

    def test_ingest_updates_device_last_seen(self, client_no_auth):
        # register device first
        client_no_auth.post("/devices/register", json={"node_id": "seen-test"})
        # ingest prediction
        client_no_auth.post("/ingest/prediction", json={
            "node_id": "seen-test", "label": "x", "confidence": 0.5,
        })
        r = client_no_auth.get("/devices/seen-test")
        assert r.status_code == 200
        assert r.json()["status"] == "online"


# ── device registry endpoints ───────────────────────────────
class TestDeviceEndpoints:
    def test_register_device(self, client_no_auth):
        r = client_no_auth.post("/devices/register", json={
            "node_id": "lab-01",
            "location": "Toronto",
            "latitude": 43.65,
            "longitude": -79.38,
            "task": "occupancy",
        })
        assert r.status_code == 200
        assert r.json()["node_id"] == "lab-01"

    def test_list_devices(self, client_no_auth):
        for i in range(3):
            client_no_auth.post("/devices/register", json={"node_id": f"d-{i}"})
        r = client_no_auth.get("/devices")
        assert len(r.json()) == 3

    def test_get_device(self, client_no_auth):
        client_no_auth.post("/devices/register", json={
            "node_id": "get-test", "location": "Berlin",
        })
        r = client_no_auth.get("/devices/get-test")
        assert r.status_code == 200
        assert r.json()["location"] == "Berlin"

    def test_get_device_not_found(self, client_no_auth):
        r = client_no_auth.get("/devices/nonexistent")
        assert r.status_code == 404

    def test_delete_device(self, client_no_auth):
        client_no_auth.post("/devices/register", json={"node_id": "rm-test"})
        r = client_no_auth.delete("/devices/rm-test")
        assert r.status_code == 200
        r = client_no_auth.get("/devices/rm-test")
        assert r.status_code == 404

    def test_filter_by_location(self, client_no_auth):
        client_no_auth.post("/devices/register", json={"node_id": "a", "location": "X"})
        client_no_auth.post("/devices/register", json={"node_id": "b", "location": "Y"})
        r = client_no_auth.get("/devices", params={"location": "X"})
        assert len(r.json()) == 1
        assert r.json()[0]["node_id"] == "a"

    def test_filter_by_task(self, client_no_auth):
        client_no_auth.post("/devices/register", json={"node_id": "a", "task": "occupancy"})
        client_no_auth.post("/devices/register", json={"node_id": "b", "task": "har"})
        r = client_no_auth.get("/devices", params={"task": "har"})
        assert len(r.json()) == 1


# ── data upload ─────────────────────────────────────────────
class TestUpload:
    def test_upload_file(self, client_no_auth):
        content = b"fake npz data here"
        r = client_no_auth.post(
            "/upload/node-1",
            files={"file": ("test.npz", io.BytesIO(content), "application/octet-stream")},
            params={"label": "test", "samples": 100, "duration_sec": 60.0},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "uploaded"
        assert data["size"] == len(content)

    def test_upload_queryable(self, client_no_auth):
        client_no_auth.post(
            "/upload/upload-q",
            files={"file": ("d.npz", io.BytesIO(b"x"), "application/octet-stream")},
        )
        r = client_no_auth.get("/uploads", params={"node_id": "upload-q"})
        assert len(r.json()) == 1
        assert r.json()[0]["filename"] == "d.npz"

    def test_upload_with_auth(self, client_auth):
        r = client_auth.post(
            "/upload/auth-test",
            files={"file": ("d.npz", io.BytesIO(b"x"), "application/octet-stream")},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert r.status_code == 200


# ── query endpoints ─────────────────────────────────────────
class TestQueryEndpoints:
    def test_predictions_empty(self, client_no_auth):
        r = client_no_auth.get("/predictions")
        assert r.status_code == 200
        assert r.json() == []

    def test_diagnostics_empty(self, client_no_auth):
        r = client_no_auth.get("/diagnostics")
        assert r.status_code == 200

    def test_uploads_empty(self, client_no_auth):
        r = client_no_auth.get("/uploads")
        assert r.status_code == 200

    def test_query_limit(self, client_no_auth):
        for i in range(20):
            client_no_auth.post("/ingest/prediction", json={
                "node_id": "limit-test", "label": str(i), "confidence": 0.5,
            })
        r = client_no_auth.get("/predictions", params={"limit": 5})
        assert len(r.json()) == 5


# ── FL coordination ─────────────────────────────────────────
class TestFL:
    def test_distribute_and_fetch(self, client_no_auth):
        model_bytes = b"fake pickle model data"
        r = client_no_auth.post(
            "/fl/distribute",
            files={"model_file": ("model.pkl", io.BytesIO(model_bytes), "application/octet-stream")},
            params={"task": "occupancy"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "distributed"

        r = client_no_auth.get("/fl/model/occupancy")
        assert r.status_code == 200
        assert r.content == model_bytes

    def test_fetch_nonexistent_model(self, client_no_auth):
        r = client_no_auth.get("/fl/model/nonexistent")
        assert r.status_code == 404

    def test_upload_update(self, client_no_auth):
        r = client_no_auth.post(
            "/fl/update/node-1",
            files={"model_file": ("update.pkl", io.BytesIO(b"update"), "application/octet-stream")},
            params={"task": "occupancy"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "received"

    def test_distribute_requires_admin(self, client_auth):
        r = client_auth.post(
            "/fl/distribute",
            files={"model_file": ("m.pkl", io.BytesIO(b"x"), "application/octet-stream")},
            headers={"X-API-Key": "test-secret-key"},  # not admin key
        )
        assert r.status_code == 403

        r = client_auth.post(
            "/fl/distribute",
            files={"model_file": ("m.pkl", io.BytesIO(b"x"), "application/octet-stream")},
            headers={"X-API-Key": "admin-key"},
        )
        assert r.status_code == 200
