"""Actuator adapters, handles, commands, and LAN actuator transport."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import whispy
from whispy.actuators import Speak, Stop, SetVolume, ShowPattern, Clear
from whispy.actuators.base import (
    ActuatorAdapter, ActuatorHandle, ActuatorMeta,
)
from whispy.contracts import (
    ActionResult, ActionStatus, ActuatorCommand, ActuatorDescriptor,
)
from whispy.devices.local import LanDevice, LocalDevice


class _FakeSpeakerHandle(ActuatorHandle):
    def __init__(self, descriptor):
        self._desc = descriptor
        self.calls = []

    @property
    def info(self):
        return self._desc

    def execute(self, command):
        self.calls.append(command.operation)
        if command.operation == "speak":
            return ActionResult(status=ActionStatus.SUCCEEDED,
                                action_type="speaker",
                                detail=f"said {command.params.get('text')!r}")
        if command.operation == "explode":
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker", detail="boom")
        return ActionResult(status=ActionStatus.UNSUPPORTED,
                            action_type="speaker",
                            detail=f"unsupported {command.operation}")


class FakeSpeakerAdapter(ActuatorAdapter):
    def __init__(self):
        self.handle = _FakeSpeakerHandle(self._desc())

    @staticmethod
    def _desc():
        return ActuatorDescriptor(
            id="speaker-81a2", kind="speaker", adapter="fake-speaker",
            name="builtin-speaker", hardware_id="fake-speaker-0",
            operations=["speak", "stop", "set_volume"], stable=True)

    def metadata(self):
        return ActuatorMeta(name="fake-speaker", kinds=("speaker",))

    def discover(self):
        return [self._desc()]

    def connect(self, descriptor, config=None):
        return self.handle


def _device_with_speaker():
    return LocalDevice(device_id="laptop-01", drivers={}, adapters={},
                       actuator_adapters={"fake-speaker": FakeSpeakerAdapter()})


def test_local_actuator_inventory():
    dev = _device_with_speaker()
    acts = dev.actuators()
    assert [a.id for a in acts] == ["speaker-81a2"]
    assert acts[0].kind == "speaker"
    assert "speak" in acts[0].operations


def test_local_actuator_execute_speak():
    dev = _device_with_speaker()
    speaker = dev.actuator("speaker")
    result = speaker.execute(Speak("hello there"))
    assert result.status is ActionStatus.SUCCEEDED
    assert "hello there" in result.detail


def test_local_actuator_unknown_kind():
    dev = _device_with_speaker()
    with pytest.raises(KeyError):
        dev.actuator("matrix")


def test_actuator_command_roundtrip():
    cmd = Speak("test", rate=2)
    d = cmd.to_dict()
    cmd2 = ActuatorCommand.from_dict(d)
    assert cmd2.operation == "speak"
    assert cmd2.params["text"] == "test" and cmd2.params["rate"] == 2


def test_command_helpers():
    assert Speak("x").operation == "speak"
    assert Stop().operation == "stop"
    assert SetVolume(0.5).params["level"] == 0.5
    assert ShowPattern([[1]]).operation == "show"
    assert Clear().operation == "clear"


# -- LAN actuator transport -----------------------------------------------------

class _ActAPI(BaseHTTPRequestHandler):
    """Minimal stand-in for thoth's /api/actuators endpoints."""

    calls = []

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/actuators":
            return self._json(200, {"actuators": [FakeSpeakerAdapter._desc()
                                                  .to_dict()]})
        if self.path == "/api/device":
            return self._json(200, {"id": "rpi1", "stable_uuid": "rpi1",
                                    "name": "rpi1"})
        if self.path == "/api/sensors":
            return self._json(200, {"sensors": []})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/api/actuators/") and \
                self.path.endswith("/actions"):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            _ActAPI.calls.append((self.path, body))
            return self._json(200, {
                "status": "succeeded", "action_type": "speaker",
                "detail": f"remote {body.get('operation')}",
                "attempts": 1})
        return self._json(404, {"error": "not found"})

    def log_message(self, *a):
        pass


@pytest.fixture
def lan_server():
    _ActAPI.calls = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ActAPI)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port
    httpd.shutdown()
    httpd.server_close()


def test_lan_actuator_inventory_and_execute(lan_server):
    dev = LanDevice("127.0.0.1", port=lan_server)
    acts = dev.actuators()
    assert acts[0].id == "speaker-81a2"
    speaker = dev.actuator("speaker")
    result = speaker.execute(Speak("remote hello"))
    assert result.status is ActionStatus.SUCCEEDED
    path, body = _ActAPI.calls[-1]
    assert path == "/api/actuators/speaker-81a2/actions"
    assert body["operation"] == "speak"
    assert body["params"]["text"] == "remote hello"


def test_lan_actuator_unknown(lan_server):
    dev = LanDevice("127.0.0.1", port=lan_server)
    with pytest.raises(KeyError):
        dev.actuator("matrix")
