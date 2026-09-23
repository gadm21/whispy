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


THOTH_NAMES = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "alex", "amina", "andre", "aria", "arjun", "ben", "carlos", "chen", "clara", "dario", "elena", "emma", "felix", "freya",
    "hana", "idris", "ivy", "jade", "jonas", "kai", "keiko", "leila", "liam", "luca", "mara", "mateo", "maya", "milo", "nina",
    "noah", "omar", "oscar", "priya", "ravi", "rosa", "sara", "soren", "theo", "uma", "vera", "yuki", "zara",
    "amsterdam", "athens", "austin", "berlin", "cairo", "chicago", "denver", "dublin", "geneva", "hanoi", "havana",
    "kyoto", "lisbon", "london", "madrid", "manila", "nairobi", "oslo", "paris", "perth", "prague", "quito", "reykjavik",
    "rome", "seoul", "sydney", "tokyo", "toronto", "venice", "vienna", "zurich"
]


def _device_hostname(device_uuid: str) -> str:
    """Consistent thoth-<name>.local hostname convention, persisted locally."""
    env_name = os.getenv("THOTH_HOSTNAME")
    if env_name:
        clean = env_name.strip().lower()
        return clean if clean.endswith(".local") else f"{clean}.local"

    if DEVICE_FILE.exists():
        try:
            data = json.loads(DEVICE_FILE.read_text())
            if data.get("device_hostname"):
                return data["device_hostname"]
            if data.get("device_name") and data["device_name"].lower().startswith("thoth-"):
                h = data["device_name"].lower()
                return h if h.endswith(".local") else f"{h}.local"
        except (ValueError, KeyError):
            pass

    sys_host = socket.gethostname().lower()
    if sys_host.startswith("thoth-"):
        chosen = f"{sys_host}.local"
    else:
        try:
            idx = int(uuid.UUID(device_uuid)) % len(THOTH_NAMES)
        except Exception:
            idx = sum(ord(c) for c in device_uuid) % len(THOTH_NAMES)
        chosen = f"thoth-{THOTH_NAMES[idx]}.local"

    try:
        data = {}
        if DEVICE_FILE.exists():
            data = json.loads(DEVICE_FILE.read_text())
        data["device_hostname"] = chosen
        if not data.get("device_name"):
            data["device_name"] = chosen.replace(".local", "")
        DEVICE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

    return chosen


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
    print(f"[thothcraft] registered model {name} ({model_id}) on sensor '{sensor}'")
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
    hostname = _device_hostname(device_uuid)
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
        status_color = "#238653" if s["online"] else "#8fa8ad"
        status_text = "online" if s["online"] else "offline"
        sensor_rows += f"""
        <tr>
            <td style="font-weight:600;color:#eef6f7;">{s['name']}</td>
            <td><code style="color:#4fd5cd;">{s['key']}</code></td>
            <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{status_color};margin-right:6px;"></span>{status_text}</td>
            <td style="color:#8fa8ad;font-size:12px;">{s.get('source', '')}</td>
        </tr>
        """

    models_html = ""
    if models:
        for m in models:
            models_html += f"""
            <div style="background:#faf8f2;border:1px solid #c9c4b9;border-radius:10px;padding:12px;margin-bottom:8px;">
                <div style="font-weight:600;color:#11110f;">{m['name']}</div>
                <div style="font-size:12px;color:#6d6961;margin-top:2px;">Target Sensor: <code style="color:#11110f;">{m['sensor']}</code> | ID: {m['id']}</div>
            </div>
            """
    else:
        models_html = '<div style="color:#6d6961;font-size:13px;">No models deployed yet. Deploy from Portal or SDK.</div>'

    preds_html = ""
    if recent:
        for p in recent:
            ts = time.strftime('%H:%M:%S', time.localtime(p.get('timestamp', time.time())))
            label = p.get('label', 'unknown')
            conf = p.get('confidence', 1.0)
            preds_html += f"""
            <div style="padding:8px 0;border-bottom:1px solid #c9c4b9;display:flex;justify-content:space-between;align-items:center;font-size:13px;">
                <div>
                    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#238653;margin-right:6px;"></span>
                    <strong>{p.get('model_name', 'model')}</strong>: <span style="font-weight:600;color:#11110f;">{label}</span> ({conf*100:.1f}%)
                </div>
                <div style="color:#6d6961;font-size:12px;font-family:ui-monospace,monospace;">{ts}</div>
            </div>
            """
    else:
        preds_html = '<div style="color:#6d6961;font-size:13px;padding:8px 0;">No predictions emitted yet.</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Thoth Device — {hostname}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#11110f">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --portal-ink: #11110f;
            --portal-paper: #f4f1e9;
            --portal-panel: #faf8f2;
            --portal-card: #ffffff;
            --portal-line: #c9c4b9;
            --portal-muted: #6d6961;
            --portal-success: #238653;
            --portal-danger: #a63730;
            --portal-teal: #4fd5cd;
            --portal-dark: #0d1517;
            --portal-dark-card: #122024;
            --portal-dark-line: #1b2a30;
            --portal-radius: 14px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--portal-paper);
            color: var(--portal-ink);
            -webkit-font-smoothing: antialiased;
            letter-spacing: -.01em;
        }}
        .site-header {{
            background: rgba(17,17,15,.96);
            color: var(--portal-paper);
            border-bottom: 1px solid #353530;
            backdrop-filter: blur(18px);
            padding: 14px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .site-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: -.03em;
        }}
        .site-mark {{
            width: 28px;
            height: 28px;
            display: grid;
            place-items: center;
            border: 1px solid #878279;
            border-radius: 50%;
            font-size: 11px;
            font-family: ui-monospace, monospace;
            font-weight: 700;
            color: #ffffff;
        }}
        .nav-host {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            color: var(--portal-teal);
            font-weight: 600;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 28px 24px 60px;
        }}
        .hero-banner {{
            background: var(--portal-panel);
            border: 1px solid var(--portal-line);
            border-radius: var(--portal-radius);
            padding: 24px 28px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .hero-banner h1 {{
            margin: 0;
            font-size: 26px;
            font-weight: 600;
            letter-spacing: -.04em;
        }}
        .hero-banner .meta {{
            font-size: 13px;
            color: var(--portal-muted);
            margin-top: 6px;
        }}
        .hero-banner .meta code {{
            color: var(--portal-ink);
            font-family: ui-monospace, monospace;
        }}
        .badge {{
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            font-family: ui-monospace, monospace;
            letter-spacing: .06em;
            text-transform: uppercase;
        }}
        .badge-linked {{
            background: rgba(35,134,83,.12);
            color: var(--portal-success);
            border: 1px solid rgba(35,134,83,.3);
        }}
        .badge-unpaired {{
            background: rgba(109,105,97,.12);
            color: var(--portal-muted);
            border: 1px solid var(--portal-line);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--portal-card);
            border: 1px solid var(--portal-line);
            border-radius: var(--portal-radius);
            padding: 22px 24px;
        }}
        .card.dark {{
            background: var(--portal-dark);
            border: 1px solid #22333a;
            color: #eef6f7;
        }}
        .card h2 {{
            margin: 0 0 16px;
            font-size: 17px;
            font-weight: 600;
            letter-spacing: -.02em;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .k-tag {{
            font: 700 11px ui-monospace, monospace;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: var(--portal-muted);
        }}
        .card.dark .k-tag {{
            color: #8fa8ad;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 8px;
        }}
        th {{
            color: #8fa8ad;
            font: 700 11px ui-monospace, monospace;
            letter-spacing: .12em;
            text-transform: uppercase;
            border-bottom: 1px solid #1b2a30;
        }}
        tr:not(:last-child) td {{
            border-bottom: 1px solid #1b2a30;
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 9px 18px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all .15s ease;
            border: 1px solid transparent;
        }}
        .btn-primary {{
            background: var(--portal-ink);
            color: #ffffff;
        }}
        .btn-primary:hover {{
            background: #2e2e2a;
        }}
        .btn-outline {{
            background: transparent;
            color: var(--portal-ink);
            border-color: var(--portal-line);
        }}
        .btn-outline:hover {{
            background: var(--portal-panel);
        }}
        .btn-teal {{
            background: var(--portal-teal);
            color: #071012;
            font-weight: 700;
        }}
        .btn-teal:hover {{
            opacity: .9;
        }}
        .btn-dark-outline {{
            background: #122024;
            color: #8fa8ad;
            border: 1px solid #22333a;
        }}
        .btn-dark-outline:hover {{
            color: #eef6f7;
            border-color: var(--portal-teal);
        }}
        .camera-box {{
            background: #000000;
            border: 1px solid #22333a;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .camera-box img {{
            max-width: 100%;
            max-height: 260px;
            display: block;
        }}
        pre {{
            background: #11110f;
            color: #f4f1e9;
            padding: 12px 14px;
            border-radius: 10px;
            font-size: 12px;
            overflow-x: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
    </style>
</head>
<body>
    <header class="site-header">
        <div class="site-brand">
            <span class="site-mark">&bull;</span>
            <span>Thoth Edge Node</span>
        </div>
        <div class="nav-host">{hostname}</div>
    </header>

    <div class="container">
        <div class="hero-banner">
            <div>
                <p class="k-tag" style="margin:0 0 6px;">Edge Sensor Node</p>
                <h1>{hostname}</h1>
                <div class="meta">
                    UUID: <code>{device_uuid}</code> &middot; OS: {os_name}
                </div>
            </div>
            <div>
                <span class="badge {pairing_badge_class}">{pairing_status}</span>
            </div>
        </div>

        <div class="grid">
            <div class="card dark" style="grid-column: span 2;">
                <h2>
                    <span>Connected Sensors</span>
                    <span class="k-tag">Hardware Discovery</span>
                </h2>
                <table>
                    <thead>
                        <tr><th>Sensor</th><th>Modality Key</th><th>Status</th><th>Driver / Probe Detail</th></tr>
                    </thead>
                    <tbody>
                        {sensor_rows}
                    </tbody>
                </table>
            </div>

            <div class="card dark">
                <h2>
                    <span>Built-in Camera Preview</span>
                    <span class="k-tag" style="color:var(--portal-teal);">Sensor Lab Stage</span>
                </h2>
                <div class="camera-box">
                    <img id="camera-frame" src="/api/captures/live/video/frame" alt="Camera frame" onerror="this.style.display='none'; document.getElementById('cam-msg').style.display='block';" onload="this.style.display='block'; document.getElementById('cam-msg').style.display='none';">
                    <div id="cam-msg" style="display:none; color:#8fa8ad; font-size:12px; padding:20px; text-align:center;">No active frame or camera in use</div>
                </div>
                <div style="margin-top: 14px; display: flex; gap: 8px;">
                    <button class="btn btn-dark-outline" onclick="document.getElementById('camera-frame').src='/api/captures/live/video/frame?t=' + Date.now();">Refresh Frame</button>
                    <a class="btn btn-teal" href="/api/captures/live/video/frame" target="_blank">Direct Stream</a>
                </div>
            </div>

            <div class="card">
                <h2>
                    <span>Active Models & Actuators</span>
                    <span class="k-tag">Edge Runtime</span>
                </h2>
                {models_html}
                <div style="margin-top: 14px;">
                    <button class="btn btn-primary" onclick="fetch('/api/models/predict', {{method:'POST'}}).then(r=>r.json()).then(d=>alert('Inference result: ' + JSON.stringify(d)));">Run Inference Now</button>
                </div>
            </div>

            <div class="card">
                <h2>
                    <span>Recent Predictions Stream</span>
                    <span class="k-tag">Real-Time</span>
                </h2>
                <div style="max-height: 240px; overflow-y: auto;">
                    {preds_html}
                </div>
                <div style="margin-top: 14px;">
                    <button class="btn btn-outline" onclick="location.reload();">Refresh Log</button>
                </div>
            </div>

            <div class="card">
                <h2>
                    <span>Quick Access & Connectivity</span>
                    <span class="k-tag">Local & Remote</span>
                </h2>
                <p style="font-size: 13px; color: var(--portal-muted); margin: 0 0 12px; line-height: 1.5;">
                    This node provides an OpenSSH server (port 22) and local REST API on port 5000 reachable at <code>http://{hostname}:5000</code> or <code>http://localhost:5000</code>.
                </p>
                <div style="font-size: 11px; margin-bottom: 4px; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; color: var(--portal-muted);">Link to ThothCraft Cloud:</div>
                <pre>thothcraft login
thothcraft pair</pre>
                <div style="font-size: 11px; margin-bottom: 4px; margin-top: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; color: var(--portal-muted);">Python SDK Local Inspection:</div>
                <pre>import thothcraft
node = thothcraft.local("{hostname}")
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
        print(f"[thothcraft] local API disabled: {exc}", file=sys.stderr)
        return None
    threading.Thread(target=server.serve_forever,
                     name="thothcraft-api", daemon=True).start()
    hostname = _device_hostname(device_uuid)
    print(f"[thothcraft] local API on :{port} ({hostname})")
    return server


# ── mDNS advertisement ---------------------------------------------------------

def _primary_ipv4() -> str:
    """Best-effort primary LAN IPv4 for the mDNS A record."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no traffic is actually sent
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _start_mdns(device_uuid: str, hostname: str, port: int):
    """Advertise ``thoth-<name>.local`` on the LAN via mDNS/Zeroconf.

    Publishes an A record (hostname → LAN IP) plus a ``_thoth._tcp.local.``
    service pointing at the local API port, so terminals and browsers resolve
    http://thoth-<name>.local:5000 with zero configuration. Returns the
    Zeroconf instance (caller must close it) or None when unavailable.
    """
    if not port:
        return None
    try:
        from zeroconf import IPVersion, ServiceInfo, Zeroconf  # type: ignore
    except ImportError:
        print(f"[thothcraft] zeroconf not installed — {hostname} will not "
              "resolve on the LAN (pip install zeroconf)", file=sys.stderr)
        return None
    try:
        host = hostname if hostname.endswith(".local") else f"{hostname}.local"
        short = host[:-len(".local")]
        ip = _primary_ipv4()
        info = ServiceInfo(
            "_thoth._tcp.local.",
            f"{short}._thoth._tcp.local.",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={
                "uuid": device_uuid,
                "daemon": "thothcraftd",
                "dashboard": f"http://{host}:{port}",
            },
            server=f"{host}.",
        )
        zc = Zeroconf(ip_version=IPVersion.V4Only)
        zc.register_service(info)
        print(f"[thothcraft] mDNS: {host} -> {ip} "
              f"(dashboard http://{host}:{port})")
        return zc
    except Exception as exc:
        print(f"[thothcraft] mDNS advertisement failed: {exc}", file=sys.stderr)
        return None


def run(config_path: str = None) -> int:
    """Main daemon loop: local API → register → heartbeat → sync → commands."""
    from thothcraft.client import Client, DEFAULT_BASE_URL
    from . import probe

    device_uuid = _device_uuid()
    hostname = _device_hostname(device_uuid)
    token = _load_device_token()
    if not token:
        # Fail fast before binding ports: the daemon's job is Brain sync, so
        # an unpaired node exits with guidance instead of idling forever.
        print("[thothcraft] No device credential — run `thothcraft pair` to link "
              "this node to your account, then start the daemon again.",
              file=sys.stderr)
        return 2
    server = _serve_local_api(device_uuid, LOCAL_API_PORT)
    zc = _start_mdns(device_uuid, hostname, LOCAL_API_PORT if server else 0)

    base_url = Client.load_base_url() or DEFAULT_BASE_URL
    client = Client(base_url, token=token) if token else None
    # Keyed capability map (usb_camera, radar, esp32_csi, ...) — Brain stores
    # it in hardware_info so the portal shows real sensor state.
    capabilities = {item["key"]: item["online"] for item in _sensor_inventory()}

    print(f"[thothcraft] device={device_uuid} ({hostname}) brain={base_url}")
    print(f"[thothcraft] capabilities: {capabilities}")

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

    threading.Thread(target=_inference_worker, name="thothcraft-inference", daemon=True).start()

    while not stop:
        if client is not None:
            try:
                hb_res = client._http.post_json("/api/device/heartbeat", body={
                    "device_id": device_uuid,
                    "device_hostname": hostname,
                    "capabilities": capabilities,
                    "daemon": "thothcraft",
                })
                deployments = hb_res.get("pending_deployments") if isinstance(hb_res, dict) else None
                if deployments is None and isinstance(hb_res, dict):
                    deployments = (hb_res.get("data") or {}).get("pending_deployments")
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
                            print(f"[thothcraft] deployment failed: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[thothcraft] heartbeat failed: {e}", file=sys.stderr)
        time.sleep(HEARTBEAT_SECONDS)

    if zc is not None:
        try:
            zc.unregister_all_services()
            zc.close()
        except Exception:
            pass
    if server is not None:
        server.shutdown()
    print("[thothcraft] stopped")
    return 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="thothcraft daemon")
    parser.add_argument("--config", default=None, help="device.json path")
    args = parser.parse_args()
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
