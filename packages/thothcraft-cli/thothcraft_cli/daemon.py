"""thothcraftd — the Thoth device daemon.

Handles sensor discovery, collection, local storage, prediction, cloud
synchronization, commands and heartbeats. The `thothcraft` CLI controls
it; on Linux it runs under systemd (`systemctl status thothcraftd`).

A Thoth device is any authenticated node implementing the device
protocol — this daemon is the reference implementation for computers
(Pi, laptop, Jetson) rather than dedicated hardware.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CONFIG_DIR = Path(os.getenv("THOTHCRAFT_CONFIG_DIR", Path.home() / ".thothcraft"))
DEVICE_FILE = CONFIG_DIR / "device.json"
HEARTBEAT_SECONDS = int(os.getenv("THOTHCRAFT_HEARTBEAT_SECONDS", "30"))
# Local API port — thothcraft.local() talks to this, same surface as the Pi
# dashboard. 0 disables it.
LOCAL_API_PORT = int(os.getenv("THOTHCRAFTD_PORT", "5000"))
CAMERA_FPS = float(os.getenv("THOTHCRAFTD_CAMERA_FPS", "5"))

_ACTIVE_MODELS: dict[str, Any] = {}
_RECENT_PREDICTIONS: list[dict[str, Any]] = []
_MODELS_LOCK = threading.Lock()


def _device_uuid() -> str:
    """Stable per-machine device UUID, persisted locally."""
    if DEVICE_FILE.exists():
        try:
            return json.loads(DEVICE_FILE.read_text())["device_uuid"]
        except (ValueError, KeyError):
            pass
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    new_uuid = str(uuid.uuid4())
    DEVICE_FILE.write_text(json.dumps({"device_uuid": new_uuid}))
    return new_uuid


def _load_device_token() -> str:
    try:
        return json.loads(DEVICE_FILE.read_text())["device_token"]
    except (OSError, ValueError, KeyError):
        return ""


# ── camera -------------------------------------------------------------------

class _Camera:
    """Latest-frame camera grabber (OpenCV). Starts on first request so the
    device camera is only held while someone is actually watching."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: bytes | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_error: str | None = None

    def _run(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            self.last_error = "opencv-python not installed (pip install thothcraft-cli[sensors])"
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.last_error = "no camera device"
            return
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.2)
                    continue
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    with self._lock:
                        self._frame = buf.tobytes()
                time.sleep(max(0.0, 1.0 / CAMERA_FPS - 0.01))
        finally:
            cap.release()

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def ensure_started(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="thothcraftd-camera", daemon=True)
            self._thread.start()


_CAMERA = _Camera()


# ── local API ------------------------------------------------------------------

_SENSOR_KEYS = {
    "Camera": "usb_camera",
    "Microphone": "microphone",
    "Wi-Fi": "wifi",
    "Bluetooth": "bluetooth",
    "Accelerometer": "accelerometer",
    "Radar": "radar",
    "CSI": "esp32_csi",
}


def _sensor_inventory() -> list[dict]:
    """probe.scan() results in the same shape the Pi dashboard returns."""
    from . import probe
    return [
        {
            "key": _SENSOR_KEYS.get(name, name.lower()),
            "name": name,
            "online": bool(available),
            "available": bool(available),
            "source": detail,
        }
        for name, (available, detail) in probe.scan().items()
    ]


def _register_model(model_config: dict[str, Any]) -> dict[str, Any]:
    """Register and instantiate a rule model with attached actuator."""
    from thothcraft.processors.base import RuleProcessor, ProcessorMeta
    model_id = str(model_config.get("id") or model_config.get("deployment_id") or uuid.uuid4())
    rule_config = model_config.get("rule_config") or model_config
    name = str(model_config.get("model_name") or model_config.get("name") or "rule-model")
    sensor = str(model_config.get("sensor") or rule_config.get("sensor") or "camera")
    meta = ProcessorMeta(
        name=name,
        processor_type="rule",
        sensor=sensor,
        task=str(model_config.get("task") or "classification"),
    )
    processor = RuleProcessor(rule_config, meta=meta)
    with _MODELS_LOCK:
        _ACTIVE_MODELS[model_id] = processor
    print(f"[thothcraftd] registered model {name} ({model_id}) on sensor '{sensor}'")
    return {"id": model_id, "name": name, "sensor": sensor, "status": "active"}


def _run_single_inference(processor: Any) -> dict[str, Any]:
    """Run model on current sensor input and record prediction."""
    from thothcraft.processors.base import SensorWindow
    sensor = processor.metadata().sensor
    window_data: dict[str, Any] = {}

    if sensor in ("camera", "video", "usb_camera"):
        _CAMERA.ensure_started()
        frame = _CAMERA.frame()
        if frame:
            window_data["camera"] = frame
    else:
        # Include probed system metrics
        for item in _sensor_inventory():
            window_data[item["key"]] = 1.0 if item["online"] else 0.0

    window = SensorWindow(window_data)
    prediction = processor.predict(window)
    res = {
        "model_name": processor.metadata().name,
        "sensor": sensor,
        "label": prediction.label,
        "confidence": prediction.confidence,
        "people_count": prediction.people_count,
        "extras": prediction.extras,
        "timestamp": time.time(),
    }
    with _MODELS_LOCK:
        _RECENT_PREDICTIONS.append(res)
        if len(_RECENT_PREDICTIONS) > 100:
            _RECENT_PREDICTIONS.pop(0)
    return res


def _dashboard_html(device_uuid: str) -> str:
    hostname = socket.gethostname()
    os_name = f"{platform.system()} {platform.release()}"
    token = _load_device_token()
    pairing_status = "Linked to Brain" if token else "Unpaired (Local Mode)"
    pairing_badge_class = "badge-linked" if token else "badge-unpaired"
    sensors = _sensor_inventory()
    with _MODELS_LOCK:
        models = [
            {"id": mid, "name": p.metadata().name, "sensor": p.metadata().sensor}
            for mid, p in _ACTIVE_MODELS.items()
        ]
        recent = list(_RECENT_PREDICTIONS)[-10:]
        recent.reverse()

    sensor_rows = ""
    for s in sensors:
        status_color = "#10b981" if s["online"] else "#64748b"
        status_text = "online" if s["online"] else "offline"
        sensor_rows += f"""
        <tr>
            <td style="font-weight:600;">{s['name']}</td>
            <td><code style="color:#0284c7;">{s['key']}</code></td>
            <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{status_color};margin-right:6px;"></span>{status_text}</td>
            <td style="color:#64748b;font-size:12px;">{s.get('source', '')}</td>
        </tr>
        """

    models_html = ""
    if models:
        for m in models:
            models_html += f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin-bottom:8px;">
                <div style="font-weight:600;">{m['name']}</div>
                <div style="font-size:12px;color:#64748b;">Target Sensor: <code>{m['sensor']}</code> | ID: {m['id']}</div>
            </div>
            """
    else:
        models_html = '<div style="color:#64748b;font-size:13px;">No models deployed yet. Deploy from Portal or SDK.</div>'

    preds_html = ""
    if recent:
        for p in recent:
            ts = time.strftime('%H:%M:%S', time.localtime(p.get('timestamp', time.time())))
            label = p.get('label', 'unknown')
            conf = p.get('confidence', 1.0)
            preds_html += f"""
            <div style="padding:8px 12px;border-bottom:1px solid #f1f5f9;display:flex;justify-content:space-between;font-size:13px;">
                <div><strong>{p.get('model_name', 'model')}</strong>: <span style="color:#0284c7;font-weight:600;">{label}</span> ({conf*100:.1f}%)</div>
                <div style="color:#94a3b8;font-size:12px;">{ts}</div>
            </div>
            """
    else:
        preds_html = '<div style="color:#64748b;font-size:13px;padding:12px;">No predictions emitted yet.</div>'

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ThothCraft Node — {hostname}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }}
        .container {{ max-width: 980px; margin: 0 auto; }}
        header {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
        h1 {{ margin: 0; font-size: 22px; font-weight: 700; color: #0f172a; }}
        .badge {{ padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
        .badge-linked {{ background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }}
        .badge-unpaired {{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }}
        .card h2 {{ margin: 0 0 14px; font-size: 16px; font-weight: 600; color: #1e293b; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ text-align: left; padding: 8px 10px; }}
        th {{ color: #64748b; font-weight: 600; border-bottom: 1px solid #e2e8f0; }}
        tr:not(:last-child) td {{ border-bottom: 1px solid #f8fafc; }}
        .btn {{ display: inline-block; background: #0f172a; color: #ffffff; padding: 8px 14px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 600; border: none; cursor: pointer; }}
        .btn:hover {{ background: #334155; }}
        .btn-outline {{ background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; }}
        .btn-outline:hover {{ background: #f8fafc; }}
        .camera-box {{ background: #000000; border-radius: 8px; overflow: hidden; position: relative; min-height: 180px; display: flex; align-items: center; justify-content: center; }}
        .camera-box img {{ max-width: 100%; max-height: 240px; display: block; }}
        pre {{ background: #0f172a; color: #f8fafc; padding: 12px; border-radius: 8px; font-size: 12px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>ThothCraft Node · {hostname}</h1>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                    UUID: <code style="color: #0f172a;">{device_uuid}</code> · OS: {os_name}
                </div>
            </div>
            <div>
                <span class="badge {pairing_badge_class}">{pairing_status}</span>
            </div>
        </header>

        <div class="grid">
            <div class="card" style="grid-column: span 2;">
                <h2>Connected Sensors</h2>
                <table>
                    <thead>
                        <tr><th>Sensor</th><th>Modality Key</th><th>Status</th><th>Driver / Probe Detail</th></tr>
                    </thead>
                    <tbody>
                        {sensor_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>Built-in Camera Preview</h2>
                <div class="camera-box">
                    <img id="camera-frame" src="/api/captures/live/video/frame" alt="Camera frame" onerror="this.style.display='none'; document.getElementById('cam-msg').style.display='block';" onload="this.style.display='block'; document.getElementById('cam-msg').style.display='none';">
                    <div id="cam-msg" style="display:none; color:#94a3b8; font-size:12px; padding:20px; text-align:center;">No frame or camera in use</div>
                </div>
                <div style="margin-top: 12px; display: flex; gap: 8px;">
                    <button class="btn btn-outline" onclick="document.getElementById('camera-frame').src='/api/captures/live/video/frame?t=' + Date.now();">Refresh Frame</button>
                    <a class="btn btn-outline" href="/api/captures/live/video/frame" target="_blank">Open Direct Feed</a>
                </div>
            </div>

            <div class="card">
                <h2>Active Models & Actuators</h2>
                {models_html}
                <div style="margin-top: 12px;">
                    <button class="btn" onclick="fetch('/api/models/predict', {{method:'POST'}}).then(r=>r.json()).then(d=>alert('Inference result: ' + JSON.stringify(d)));">Run Inference Now</button>
                </div>
            </div>

            <div class="card">
                <h2>Recent Predictions Stream</h2>
                <div style="max-height: 240px; overflow-y: auto;">
                    {preds_html}
                </div>
                <div style="margin-top: 10px;">
                    <button class="btn btn-outline" onclick="location.reload();">Refresh Log</button>
                </div>
            </div>

            <div class="card">
                <h2>Quick Access & Remote Control</h2>
                <p style="font-size: 13px; color: #475569; margin: 0 0 10px;">
                    This node provides an OpenSSH server (port 22) and local REST API on port 5000.
                </p>
                <div style="font-size: 12px; margin-bottom: 6px; font-weight: 600;">Link to ThothCraft Cloud:</div>
                <pre>thothcraft login
thothcraft pair</pre>
                <div style="font-size: 12px; margin-bottom: 6px; font-weight: 600; margin-top: 10px;">Python SDK Local Inspection:</div>
                <pre>import thothcraft
node = thothcraft.local("127.0.0.1")
print(node.sensors())</pre>
            </div>
        </div>
    </div>
</body>
</html>"""


def _make_handler(device_uuid: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes,
                  content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802 — stdlib handler name
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("", "/", "/dashboard"):
                html = _dashboard_html(device_uuid)
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif path in ("/health", "/api/health"):
                self._json({"status": "ok", "daemon": "thothcraftd",
                            "device_id": device_uuid})
            elif path == "/api/sensors":
                self._json({"sensors": _sensor_inventory()})
            elif path == "/api/settings":
                self._json({"device_id": device_uuid, "daemon": "thothcraftd",
                            "capabilities": _sensor_inventory()})
            elif path == "/api/models":
                with _MODELS_LOCK:
                    models_list = [
                        {
                            "id": mid,
                            "name": p.metadata().name,
                            "sensor": p.metadata().sensor,
                            "processor_type": p.metadata().processor_type,
                            "task": p.metadata().task,
                        }
                        for mid, p in _ACTIVE_MODELS.items()
                    ]
                self._json({"models": models_list})
            elif path == "/api/predictions":
                with _MODELS_LOCK:
                    preds = list(_RECENT_PREDICTIONS)
                self._json({"predictions": preds})
            elif path == "/api/captures/live/video/frame":
                _CAMERA.ensure_started()
                deadline = time.monotonic() + 8
                frame = _CAMERA.frame()
                while frame is None and time.monotonic() < deadline \
                        and _CAMERA.last_error is None:
                    time.sleep(0.1)
                    frame = _CAMERA.frame()
                if frame is None:
                    self._json({"error": _CAMERA.last_error or "no frame yet"},
                               code=404)
                else:
                    self._send(200, frame, "image/jpeg")
            else:
                self._json({"error": "not found"}, code=404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                body = {}

            if path == "/api/live/session":
                self._json({"success": True})
            elif path == "/api/models":
                res = _register_model(body)
                self._json({"success": True, "model": res})
            elif path in ("/api/models/predict", "/api/predict"):
                with _MODELS_LOCK:
                    models = list(_ACTIVE_MODELS.values())
                results = [_run_single_inference(m) for m in models]
                self._json({"success": True, "results": results})
            elif path == "/api/internal/prediction":
                # Manual prediction injection
                label = body.get("label", "occupied")
                conf = float(body.get("confidence", 1.0))
                with _MODELS_LOCK:
                    _RECENT_PREDICTIONS.append({
                        "model_name": body.get("model_name", "injected"),
                        "label": label,
                        "confidence": conf,
                        "timestamp": time.time(),
                    })
                self._json({"success": True, "injected": label})
            else:
                self._json({"error": "not found"}, code=404)

        def log_message(self, fmt: str, *args) -> None:
            pass  # keep stdout clean; heartbeats log their own failures

    return Handler


def _serve_local_api(device_uuid: str, port: int) -> ThreadingHTTPServer | None:
    if not port:
        return None
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(device_uuid))
    except OSError as exc:
        print(f"[thothcraftd] local API disabled: {exc}", file=sys.stderr)
        return None
    threading.Thread(target=server.serve_forever,
                     name="thothcraftd-api", daemon=True).start()
    print(f"[thothcraftd] local API on :{port} "
          f"(thothcraft.local('{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}'))")
    return server


def run(config_path: str = None) -> int:
    """Main daemon loop: local API → register → heartbeat → sync → commands."""
    from thothcraft.client import Client, DEFAULT_BASE_URL
    from . import probe

    device_uuid = _device_uuid()
    server = _serve_local_api(device_uuid, LOCAL_API_PORT)
    token = _load_device_token()
    if not token:
        print("[thothcraftd] No device credential — local API only; "
              "run `thothcraft pair` to link Brain",
              file=sys.stderr)

    base_url = Client.load_base_url() or DEFAULT_BASE_URL
    client = Client(base_url, token=token) if token else None
    capabilities = {k: v[0] for k, v in probe.scan().items()}

    print(f"[thothcraftd] device={device_uuid} brain={base_url}")
    print(f"[thothcraftd] capabilities: {capabilities}")

    stop = False

    def _term(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    def _inference_worker():
        while not stop:
            with _MODELS_LOCK:
                models = list(_ACTIVE_MODELS.values())
            for model in models:
                try:
                    _run_single_inference(model)
                except Exception as exc:
                    pass
            time.sleep(2.0)

    threading.Thread(target=_inference_worker, name="thothcraftd-inference", daemon=True).start()

    while not stop:
        if client is not None:
            try:
                hb_res = client._http.post_json("/api/device/heartbeat", body={
                    "device_id": device_uuid,
                    "capabilities": capabilities,
                    "daemon": "thothcraftd",
                })
                deployments = hb_res.get("pending_deployments") if isinstance(hb_res, dict) else None
                if isinstance(deployments, list):
                    for dep in deployments:
                        try:
                            _register_model(dep)
                            dep_id = dep.get("deployment_id")
                            if dep_id:
                                client._http.post_json(
                                    f"/api/datasets/models/deployments/{dep_id}/confirm",
                                    body={"status": "delivered"},
                                )
                        except Exception as e:
                            print(f"[thothcraftd] deployment failed: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[thothcraftd] heartbeat failed: {e}", file=sys.stderr)
        time.sleep(HEARTBEAT_SECONDS)

    if server is not None:
        server.shutdown()
    print("[thothcraftd] stopped")
    return 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="thothcraftd")
    parser.add_argument("--config", default=None, help="device.json path")
    args = parser.parse_args()
    raise SystemExit(run(args.config))
