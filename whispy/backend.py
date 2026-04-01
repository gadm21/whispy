"""Central backend server for global Whispy deployments.

Runs a FastAPI server with device registry, data ingest, file upload,
FL coordination, and dashboard API.  Deployable to Railway as a hosted
SaaS — devices authenticate with API keys over REST.

Environment variables (Railway / cloud)::

    WHISPY_API_KEY      — required, shared secret for device auth
    WHISPY_ADMIN_KEY    — optional, admin-only endpoints (default = API_KEY)
    WHISPY_DATA_DIR     — data directory (default ./whispy_backend_data)
    WHISPY_DB_PATH      — SQLite path (default ./whispy_backend.db)
    PORT                — HTTP port (Railway sets this automatically)
    WHISPY_MQTT_BROKER  — optional, MQTT broker for hybrid mode

Usage::

    whispy backend start                          # local dev
    whispy backend start --api-key mysecret       # with auth
    railway up                                    # deploy to Railway
"""
from __future__ import annotations
import hashlib, hmac, json, os, secrets, sqlite3, shutil, threading, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

# =========================================================================
# SQLite store
# =========================================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT, ts TEXT,
    label TEXT, class_idx INTEGER, confidence REAL, probabilities TEXT);
CREATE TABLE IF NOT EXISTS diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT, ts TEXT,
    csi_rate REAL, cpu_temp REAL, esp32_alive INTEGER, cache_mb REAL);
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT, ts TEXT,
    filename TEXT, size_bytes INTEGER, samples INTEGER,
    duration_sec REAL, label TEXT, path TEXT);
CREATE INDEX IF NOT EXISTS idx_pred_node ON predictions(node_id);
CREATE INDEX IF NOT EXISTS idx_diag_node ON diagnostics(node_id);
"""

class TimeSeriesDB:
    def __init__(self, path: str = "./whispy_backend.db"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA); self._conn.commit()

    def insert_prediction(self, node_id, label, class_idx, confidence, probs=None):
        with self._lock:
            self._conn.execute(
                "INSERT INTO predictions(node_id,ts,label,class_idx,confidence,probabilities) VALUES(?,?,?,?,?,?)",
                (node_id, _now(), label, class_idx, confidence, json.dumps(probs) if probs else None))
            self._conn.commit()

    def insert_diagnostic(self, node_id, csi_rate, cpu_temp, esp32_alive, cache_mb):
        with self._lock:
            self._conn.execute(
                "INSERT INTO diagnostics(node_id,ts,csi_rate,cpu_temp,esp32_alive,cache_mb) VALUES(?,?,?,?,?,?)",
                (node_id, _now(), csi_rate, cpu_temp, 1 if esp32_alive else 0, cache_mb))
            self._conn.commit()

    def insert_upload(self, node_id, filename, size_bytes, samples, duration_sec, label, path):
        with self._lock:
            self._conn.execute(
                "INSERT INTO uploads(node_id,ts,filename,size_bytes,samples,duration_sec,label,path) VALUES(?,?,?,?,?,?,?,?)",
                (node_id, _now(), filename, size_bytes, samples, duration_sec, label, path))
            self._conn.commit()

    def query(self, table, node_id=None, limit=100):
        q = f"SELECT * FROM {table}"
        p = []
        if node_id: q += " WHERE node_id=?"; p.append(node_id)
        q += " ORDER BY ts DESC LIMIT ?"; p.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, p).fetchall()]

    def stats(self):
        with self._lock:
            c = self._conn
            return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("predictions","diagnostics","uploads")}

# =========================================================================
# MQTT subscriber
# =========================================================================
class MQTTSubscriber:
    def __init__(self, broker, port=1883, user=None, pw=None, db=None, registry=None):
        self.broker, self.port = broker, port
        self.user, self.pw = user, pw
        self.db, self.registry = db, registry
        self._client = None

    def start(self):
        try: import paho.mqtt.client as mqtt
        except ImportError: print("[backend] paho-mqtt required"); return False
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                   client_id=f"whispy_backend_{int(time.time())}")
        if self.user: self._client.username_pw_set(self.user, self.pw)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
            print(f"[backend] MQTT subscribing to {self.broker}:{self.port}")
            return True
        except Exception as e: print(f"[backend] MQTT error: {e}"); return False

    def stop(self):
        if self._client: self._client.loop_stop(); self._client.disconnect()

    def _on_connect(self, client, ud, flags, rc, props=None):
        for t in ["whispy/+/prediction/state","whispy/+/diagnostics/#",
                   "whispy/+/availability","whispy/+/device/info"]:
            client.subscribe(t, qos=1)
        print("[backend] MQTT subscribed")

    def _on_message(self, client, ud, msg):
        parts = msg.topic.split("/")
        if len(parts) < 3: return
        node_id, payload = parts[1], msg.payload.decode("utf-8", errors="replace")
        try:
            if parts[2] == "prediction" and len(parts) > 3:
                d = json.loads(payload)
                if self.db:
                    self.db.insert_prediction(node_id, d.get("label",""),
                        d.get("class",-1), d.get("confidence",0), d.get("probabilities"))
            elif parts[2] == "diagnostics" and len(parts) > 3 and parts[3] == "csi_rate":
                if self.db:
                    self.db.insert_diagnostic(node_id, float(payload), None, True, 0)
            elif parts[2] == "availability" and self.registry:
                dev = self.registry.get(node_id)
                if dev:
                    dev.status = "online" if payload == "online" else "offline"
                    dev.last_seen = _now(); self.registry.upsert(dev)
            elif parts[2] == "device" and len(parts) > 3 and parts[3] == "info":
                if self.registry:
                    from whispy.device import DeviceInfo
                    dev = DeviceInfo.from_dict(json.loads(payload))
                    dev.status = "online"; dev.last_seen = _now()
                    self.registry.upsert(dev)
                    print(f"[backend] Device registered via MQTT: {node_id}")
        except Exception: pass

# =========================================================================
# FastAPI app
# =========================================================================
def _verify_api_key(api_key: str | None, provided: str | None) -> bool:
    """Constant-time comparison of API keys."""
    if not api_key:
        return True  # no auth configured
    if not provided:
        return False
    return hmac.compare_digest(api_key.encode(), provided.encode())


def create_app(
    api_key: str | None = None,
    admin_key: str | None = None,
    data_dir: str = "./whispy_backend_data",
    db_path: str = "./whispy_backend.db",
    registry_path: str = "./whispy_devices.json",
    broker: str | None = None,
    broker_port: int = 1883,
    broker_user: str | None = None,
    broker_password: str | None = None,
):
    """Create the FastAPI app.  All settings can also come from env vars."""
    try:
        from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI required — pip install whispy[backend]")

    from whispy.device import DeviceInfo, DeviceRegistry

    # resolve from env vars (Railway convention)
    api_key = api_key or os.environ.get("WHISPY_API_KEY")
    admin_key = admin_key or os.environ.get("WHISPY_ADMIN_KEY") or api_key
    data_dir = os.environ.get("WHISPY_DATA_DIR", data_dir)
    db_path = os.environ.get("WHISPY_DB_PATH", db_path)
    broker = broker or os.environ.get("WHISPY_MQTT_BROKER")

    app = FastAPI(
        title="Whispy Backend",
        version="0.1.0",
        description="Hosted SaaS backend for the Whispy WiFi CSI sensing network. "
                    "Devices authenticate with `X-API-Key` header.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    db = TimeSeriesDB(db_path)
    registry = DeviceRegistry(registry_path)
    upload_dir = Path(data_dir) / "uploads"; upload_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(data_dir) / "models"; model_dir.mkdir(parents=True, exist_ok=True)

    # optional MQTT subscriber (hybrid mode)
    mqtt_sub: MQTTSubscriber | None = None
    if broker:
        mqtt_sub = MQTTSubscriber(broker, broker_port, broker_user, broker_password, db, registry)

    @app.on_event("startup")
    async def startup():
        if mqtt_sub:
            mqtt_sub.start()

    @app.on_event("shutdown")
    async def shutdown():
        if mqtt_sub:
            mqtt_sub.stop()

    # ── auth helpers ────────────────────────────────────────
    def require_auth(x_api_key: str | None):
        if not _verify_api_key(api_key, x_api_key):
            raise HTTPException(401, "Invalid or missing API key")

    def require_admin(x_api_key: str | None):
        if not _verify_api_key(admin_key, x_api_key):
            raise HTTPException(403, "Admin access required")

    # ── health (public) ─────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "ts": _now(), "devices": registry.count,
                "db": db.stats(), "auth_required": api_key is not None}

    # ── REST ingest (devices push here) ─────────────────────
    @app.post("/ingest/prediction")
    async def ingest_prediction(
        body: dict,
        x_api_key: str | None = Header(None),
    ):
        """Device pushes a prediction. Body: {node_id, label, class, confidence, probabilities?}"""
        require_auth(x_api_key)
        node_id = body.get("node_id", "")
        if not node_id:
            raise HTTPException(422, "node_id required")
        db.insert_prediction(
            node_id=node_id,
            label=body.get("label", ""),
            class_idx=body.get("class", -1),
            confidence=body.get("confidence", 0),
            probs=body.get("probabilities"),
        )
        # update device last_seen
        dev = registry.get(node_id)
        if dev:
            dev.status = "online"; dev.last_seen = _now()
            registry.upsert(dev)
        return {"status": "ok"}

    @app.post("/ingest/diagnostics")
    async def ingest_diagnostics(
        body: dict,
        x_api_key: str | None = Header(None),
    ):
        """Device pushes diagnostics. Body: {node_id, csi_rate, cpu_temp?, esp32_alive, cache_mb}"""
        require_auth(x_api_key)
        node_id = body.get("node_id", "")
        if not node_id:
            raise HTTPException(422, "node_id required")
        db.insert_diagnostic(
            node_id=node_id,
            csi_rate=body.get("csi_rate", 0),
            cpu_temp=body.get("cpu_temp"),
            esp32_alive=body.get("esp32_alive", True),
            cache_mb=body.get("cache_mb", 0),
        )
        return {"status": "ok"}

    # ── device registry ─────────────────────────────────────
    @app.post("/devices/register")
    async def register_device(device: dict, x_api_key: str | None = Header(None)):
        require_auth(x_api_key)
        dev = DeviceInfo.from_dict(device)
        dev.status = "online"; dev.last_seen = _now()
        registry.upsert(dev)
        return {"status": "registered", "node_id": dev.node_id}

    @app.get("/devices")
    async def list_devices(
        location: str | None = Query(None),
        task: str | None = Query(None),
        status: str | None = Query(None),
        x_api_key: str | None = Header(None),
    ):
        require_auth(x_api_key)
        devs = registry.list()
        if location: devs = [d for d in devs if d.location == location]
        if task: devs = [d for d in devs if d.task == task]
        if status: devs = [d for d in devs if d.status == status]
        return [d.to_dict() for d in devs]

    @app.get("/devices/{node_id}")
    async def get_device(node_id: str, x_api_key: str | None = Header(None)):
        require_auth(x_api_key)
        dev = registry.get(node_id)
        if not dev: raise HTTPException(404, "Device not found")
        return dev.to_dict()

    @app.delete("/devices/{node_id}")
    async def remove_device(node_id: str, x_api_key: str | None = Header(None)):
        require_admin(x_api_key)
        if not registry.remove(node_id): raise HTTPException(404, "Not found")
        return {"status": "removed"}

    # ── data upload ─────────────────────────────────────────
    @app.post("/upload/{node_id}")
    async def upload_data(
        node_id: str,
        file: UploadFile = File(...),
        label: str = Query("unknown"),
        samples: int = Query(0),
        duration_sec: float = Query(0),
        x_api_key: str | None = Header(None),
    ):
        require_auth(x_api_key)
        dest = upload_dir / node_id; dest.mkdir(parents=True, exist_ok=True)
        fpath = dest / file.filename
        content = await file.read()
        with open(fpath, "wb") as f:
            f.write(content)
        db.insert_upload(node_id, file.filename, len(content), samples,
                         duration_sec, label, str(fpath))
        return {"status": "uploaded", "path": str(fpath), "size": len(content)}

    # ── query endpoints ─────────────────────────────────────
    @app.get("/predictions")
    async def get_predictions(
        node_id: str | None = Query(None),
        limit: int = Query(100),
        x_api_key: str | None = Header(None),
    ):
        require_auth(x_api_key)
        return db.query("predictions", node_id, limit)

    @app.get("/diagnostics")
    async def get_diagnostics(
        node_id: str | None = Query(None),
        limit: int = Query(100),
        x_api_key: str | None = Header(None),
    ):
        require_auth(x_api_key)
        return db.query("diagnostics", node_id, limit)

    @app.get("/uploads")
    async def get_uploads(
        node_id: str | None = Query(None),
        limit: int = Query(100),
        x_api_key: str | None = Header(None),
    ):
        require_auth(x_api_key)
        return db.query("uploads", node_id, limit)

    # ── FL coordination ─────────────────────────────────────
    @app.post("/fl/distribute")
    async def fl_distribute(
        model_file: UploadFile = File(...),
        task: str = Query("occupancy"),
        x_api_key: str | None = Header(None),
    ):
        require_admin(x_api_key)
        fpath = model_dir / f"global_{task}.pkl"
        with open(fpath, "wb") as f:
            f.write(await model_file.read())
        return {"status": "distributed", "path": str(fpath), "task": task,
                "target_devices": len(registry.find(task=task))}

    @app.get("/fl/model/{task}")
    async def fl_get_model(task: str, x_api_key: str | None = Header(None)):
        require_auth(x_api_key)
        from fastapi.responses import FileResponse
        fpath = model_dir / f"global_{task}.pkl"
        if not fpath.exists(): raise HTTPException(404, "No global model")
        return FileResponse(fpath, filename=f"global_{task}.pkl")

    @app.post("/fl/update/{node_id}")
    async def fl_upload_update(
        node_id: str,
        model_file: UploadFile = File(...),
        task: str = Query("occupancy"),
        x_api_key: str | None = Header(None),
    ):
        require_auth(x_api_key)
        dest = model_dir / "updates" / node_id; dest.mkdir(parents=True, exist_ok=True)
        fpath = dest / f"update_{task}_{int(time.time())}.pkl"
        with open(fpath, "wb") as f:
            f.write(await model_file.read())
        return {"status": "received", "node_id": node_id, "path": str(fpath)}

    return app

# =========================================================================
# Runner
# =========================================================================
def run_server(host="0.0.0.0", port=None, api_key=None, **kwargs):
    """Start the backend.  ``port`` defaults to $PORT (Railway) or 8000."""
    import uvicorn
    port = port or int(os.environ.get("PORT", 8000))
    app = create_app(api_key=api_key, **kwargs)
    uvicorn.run(app, host=host, port=port)
