"""Central backend server for global Whispy deployments.

Runs a FastAPI server with MQTT subscriber, device registry, data upload,
FL coordination, and dashboard API.

    whispy backend start --broker mqtt.example.com --port 8000
"""
from __future__ import annotations
import json, os, sqlite3, shutil, threading, time
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
def create_app(broker="localhost", broker_port=1883, broker_user=None,
               broker_password=None, data_dir="./whispy_backend_data",
               db_path="./whispy_backend.db", registry_path="./whispy_devices.json"):
    try:
        from fastapi import FastAPI, UploadFile, File, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI required — pip install whispy[backend]")

    from whispy.device import DeviceInfo, DeviceRegistry

    app = FastAPI(title="Whispy Backend", version="0.1.0",
                  description="Central server for global Whispy CSI sensing network")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    db = TimeSeriesDB(db_path)
    registry = DeviceRegistry(registry_path)
    upload_dir = Path(data_dir) / "uploads"; upload_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(data_dir) / "models"; model_dir.mkdir(parents=True, exist_ok=True)

    mqtt_sub = MQTTSubscriber(broker, broker_port, broker_user, broker_password, db, registry)

    @app.on_event("startup")
    async def startup(): mqtt_sub.start()
    @app.on_event("shutdown")
    async def shutdown(): mqtt_sub.stop()

    # ── health ──────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "ts": _now(), "devices": registry.count, "db": db.stats()}

    # ── devices ─────────────────────────────────────────────
    @app.post("/devices/register")
    async def register_device(device: dict):
        dev = DeviceInfo.from_dict(device)
        dev.status = "online"; dev.last_seen = _now()
        registry.upsert(dev)
        return {"status": "registered", "node_id": dev.node_id}

    @app.get("/devices")
    async def list_devices(location: str | None = Query(None),
                           task: str | None = Query(None),
                           status: str | None = Query(None)):
        devs = registry.list()
        if location: devs = [d for d in devs if d.location == location]
        if task: devs = [d for d in devs if d.task == task]
        if status: devs = [d for d in devs if d.status == status]
        return [d.to_dict() for d in devs]

    @app.get("/devices/{node_id}")
    async def get_device(node_id: str):
        dev = registry.get(node_id)
        if not dev: raise HTTPException(404, "Device not found")
        return dev.to_dict()

    @app.delete("/devices/{node_id}")
    async def remove_device(node_id: str):
        if not registry.remove(node_id): raise HTTPException(404, "Not found")
        return {"status": "removed"}

    # ── data upload ─────────────────────────────────────────
    @app.post("/upload/{node_id}")
    async def upload_data(node_id: str, file: UploadFile = File(...),
                          label: str = Query("unknown"), samples: int = Query(0),
                          duration_sec: float = Query(0)):
        dest = upload_dir / node_id; dest.mkdir(parents=True, exist_ok=True)
        fpath = dest / file.filename
        with open(fpath, "wb") as f:
            content = await file.read(); f.write(content)
        db.insert_upload(node_id, file.filename, len(content), samples,
                         duration_sec, label, str(fpath))
        return {"status": "uploaded", "path": str(fpath), "size": len(content)}

    # ── predictions / diagnostics queries ───────────────────
    @app.get("/predictions")
    async def get_predictions(node_id: str | None = Query(None),
                              limit: int = Query(100)):
        return db.query("predictions", node_id, limit)

    @app.get("/diagnostics")
    async def get_diagnostics(node_id: str | None = Query(None),
                              limit: int = Query(100)):
        return db.query("diagnostics", node_id, limit)

    @app.get("/uploads")
    async def get_uploads(node_id: str | None = Query(None),
                          limit: int = Query(100)):
        return db.query("uploads", node_id, limit)

    # ── FL coordination ─────────────────────────────────────
    @app.post("/fl/distribute")
    async def fl_distribute(model_file: UploadFile = File(...),
                            task: str = Query("occupancy")):
        fpath = model_dir / f"global_{task}.pkl"
        with open(fpath, "wb") as f: f.write(await model_file.read())
        return {"status": "distributed", "path": str(fpath), "task": task,
                "target_devices": len(registry.find(task=task))}

    @app.get("/fl/model/{task}")
    async def fl_get_model(task: str):
        from fastapi.responses import FileResponse
        fpath = model_dir / f"global_{task}.pkl"
        if not fpath.exists(): raise HTTPException(404, "No global model")
        return FileResponse(fpath, filename=f"global_{task}.pkl")

    @app.post("/fl/update/{node_id}")
    async def fl_upload_update(node_id: str, model_file: UploadFile = File(...),
                               task: str = Query("occupancy")):
        dest = model_dir / "updates" / node_id; dest.mkdir(parents=True, exist_ok=True)
        fpath = dest / f"update_{task}_{int(time.time())}.pkl"
        with open(fpath, "wb") as f: f.write(await model_file.read())
        return {"status": "received", "node_id": node_id, "path": str(fpath)}

    return app

# =========================================================================
# Runner
# =========================================================================
def run_server(host="0.0.0.0", port=8000, broker="localhost", broker_port=1883,
               broker_user=None, broker_password=None, **kwargs):
    import uvicorn
    app = create_app(broker=broker, broker_port=broker_port,
                     broker_user=broker_user, broker_password=broker_password, **kwargs)
    uvicorn.run(app, host=host, port=port)
