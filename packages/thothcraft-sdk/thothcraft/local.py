"""Local device access — talk to a Thoth node's dashboard directly.

    import thothcraft
    node = thothcraft.local("thoth-chen.local")   # or "192.168.1.42"
    node.sensors()                               # probed capabilities
    node.occupancy()                             # live radar occupancy
    for cap in node.captures():                  # iterate minute folders
        ...

The local dashboard (thoth runtime, port 5000) exposes the same
concepts as Brain — sensors, captures, models — without cloud auth.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from .errors import APIError


class _LocalHttp:
    def __init__(self, base_url: str, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_json(self, path: str, params: Optional[dict] = None) -> Any:
        url = self.base_url + path
        if params:
            qs = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
            if qs:
                url += "?" + qs
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            raise APIError(f"local request failed: {url}: {exc}") from exc

    def post_json(self, path: str, body: Optional[dict] = None) -> Any:
        data = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            raise APIError(f"local request failed: {path}: {exc}") from exc

    def get_bytes(self, path: str, params: Optional[dict] = None) -> bytes:
        url = self.base_url + path
        if params:
            qs = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
            if qs:
                url += "?" + qs
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as res:
                return res.read()
        except urllib.error.HTTPError as exc:
            raise APIError(f"local request failed: {url}: HTTP {exc.code}",
                           status_code=exc.code) from exc
        except Exception as exc:
            raise APIError(f"local request failed: {url}: {exc}") from exc


class LocalDevice:
    """A Thoth node reachable on the LAN (dashboard on port 5000)."""

    def __init__(self, host: str, port: int = 5000, timeout: int = 15):
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        self._http = _LocalHttp(f"{host}:{port}", timeout=timeout)
        self.host = host

    # -- introspection ----------------------------------------------------
    def sensors(self) -> Dict[str, Any]:
        """Probed sensor capabilities reported by the node."""
        return self._http.get_json("/api/sensors")

    def status(self) -> Dict[str, Any]:
        return self._http.get_json("/api/settings")

    def occupancy(self) -> Dict[str, Any]:
        """Live radar occupancy reading."""
        return self._http.get_json("/api/radar/occupancy")

    def radar_live(self) -> Dict[str, Any]:
        return self._http.get_json("/api/radar/live")

    def radar_snr(self) -> Dict[str, Any]:
        """Current detection SNR: ``{"snr_db", "threshold_db", "detected"}``.

        Values are None when the radar hasn't produced a frame yet."""
        # Prefer the per-frame sidecar (updates every processed frame);
        # fall back to the throttled full live state on older builds.
        try:
            state = self._http.get_json("/api/radar/snr")
            return {
                "snr_db": state.get("snr_db"),
                "threshold_db": state.get("threshold_db"),
                "peak_power_db": state.get("peak_power_db"),
                "noise_floor_db": state.get("noise_floor_db"),
                "detected": bool(state.get("detected")),
                "stale": bool(state.get("stale")),
                "updated_at": state.get("updated_at"),
                "age_seconds": state.get("age_seconds"),
            }
        except Exception:
            pass
        state = self.radar_live()
        det = ((state.get("intensity") or {}).get("example2_xy") or {}).get("detection") or {}
        return {
            "snr_db": det.get("snr_db"),
            "threshold_db": det.get("threshold_db", state.get("threshold_db")),
            "peak_power_db": det.get("peak_power_db"),
            "noise_floor_db": det.get("noise_floor_db"),
            "detected": bool(state.get("person_detected") or det.get("detected")),
            "stale": bool(state.get("stale")),
            "updated_at": state.get("updated_at"),
            "age_seconds": state.get("age_seconds"),
        }

    # -- live sensor streams --------------------------------------------------
    def csi_tail(self, cursor: str = "") -> Dict[str, Any]:
        """Incremental WiFi CSI feed.

        Pass back ``response['cursor']`` on the next call to receive only new
        samples. Samples are ``[monotonic_ns, rx_index, amplitude]`` at the
        receiver's native rate (~100 Hz)."""
        return self._http.get_json("/api/captures/live/csi/tail",
                                   {"cursor": cursor})

    def csi_stream(self, poll_s: float = 0.06,
                   max_batches: Optional[int] = None) -> Iterator[List[list]]:
        """Yield batches of CSI samples as they arrive."""
        cursor = ""
        batches = 0
        while max_batches is None or batches < max_batches:
            body = self.csi_tail(cursor)
            cursor = body.get("cursor") or cursor
            samples = body.get("samples") or []
            if samples:
                batches += 1
                yield samples
            else:
                time.sleep(poll_s)

    def sensehat(self) -> Dict[str, Any]:
        """Latest Sense HAT reading plus a trailing series.

        ``latest`` carries temperature_c, humidity_percent, pressure_mbar and
        the IMU vectors (acceleration/gyroscope/compass)."""
        return self._http.get_json("/api/captures/live/sensehat")

    def camera_frame(self) -> bytes:
        """Latest live camera frame as JPEG bytes."""
        return self._http.get_bytes("/api/captures/live/video/frame")

    def camera_stream(self, max_items: Optional[int] = None,
                      min_interval_s: float = 0.05) -> Iterator[bytes]:
        """Yield live camera JPEG frames as they are produced."""
        count = 0
        while max_items is None or count < max_items:
            try:
                yield self.camera_frame()
                count += 1
            except APIError:
                time.sleep(min_interval_s * 4)  # camera warming up
                continue
            time.sleep(min_interval_s)

    def matrix(self) -> Dict[str, Any]:
        """Current Sense HAT 8x8 LED matrix state."""
        return self._http.get_json("/api/sensehat/matrix")

    def set_matrix(self, *, pixels: Optional[list] = None,
                   color: Optional[list] = None, text: Optional[str] = None,
                   clear: bool = False, low_light: Optional[bool] = None,
                   rotation: Optional[int] = None,
                   speed: Optional[float] = None) -> Dict[str, Any]:
        """Drive the Sense HAT LED matrix.

        ``pixels`` accepts either 64 ``[r,g,b]`` entries or sparse
        ``[x, y, r, g, b]`` tuples."""
        body: Dict[str, Any] = {}
        if pixels is not None:
            body["pixels"] = pixels
        if color is not None:
            body["color"] = color
        if text is not None:
            body["text"] = text
        if clear:
            body["clear"] = True
        if low_light is not None:
            body["low_light"] = low_light
        if rotation is not None:
            body["rotation"] = rotation
        if speed is not None:
            body["speed"] = speed
        return self._http.post_json("/api/sensehat/matrix", body)

    # -- captures -----------------------------------------------------------
    def captures(self) -> List[Dict[str, Any]]:
        payload = self._http.get_json("/api/captures")
        return payload.get("captures") or payload.get("minutes") or []

    def capture(self, minute: str) -> Dict[str, Any]:
        return self._http.get_json(f"/api/captures/{minute}")

    def sensor_data(self, minute: str, sensor: str) -> Any:
        return self._http.get_json(f"/api/captures/{minute}/sensor/{sensor}")

    # -- control ------------------------------------------------------------
    def live_session(self, action: str = "start") -> Dict[str, Any]:
        return self._http.post_json("/api/live/session", {"action": action})

    def predict(self, label: str, confidence: float = 1.0, *,
                model_name: str = "SDK injection") -> Dict[str, Any]:
        """Inject a prediction on the node — drives linked actuators.

        ``node.predict("occupied")`` publishes the occupancy entity and turns
        the configured Home Assistant light on; ``"empty"`` turns it off."""
        return self._http.post_json("/api/internal/prediction", {
            "class": label,
            "confidence": confidence,
            "model_name": model_name,
        })

    def models(self) -> List[Dict[str, Any]]:
        """List models deployed on this node."""
        res = self._http.get_json("/api/models")
        return res.get("models") or []

    def deploy_rule_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy or register a rule-based sensor model on this node."""
        return self._http.post_json("/api/models", model_config)

    def predictions(self) -> List[Dict[str, Any]]:
        """Recent predictions emitted on this node."""
        res = self._http.get_json("/api/predictions")
        return res.get("predictions") or []

    def run_inference(self) -> Dict[str, Any]:
        """Evaluate deployed model(s) on current sensor readings."""
        return self._http.post_json("/api/models/predict", {})

    def __repr__(self) -> str:
        return f"LocalDevice({self.host})"


def local(host: str = "thoth.local", port: int = 5000,
          timeout: int = 15) -> LocalDevice:
    """Connect to a Thoth node on the LAN.

        node = thothcraft.local("thoth-april.local")
        node.occupancy()

    Works against the Pi dashboard and against ``thothcraftd`` running on
    any computer (laptop, Jetson) — the daemon exposes the same local API.
    """
    return LocalDevice(host, port=port, timeout=timeout)
