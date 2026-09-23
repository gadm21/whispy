"""Actuator semantics — explicit results, no fake success (§7.6, §46)."""
import json
import urllib.error

import pytest

from whispy.contracts import Action, ActionStatus, Prediction
from whispy.actuators import (
    DeviceActuator, HomeAssistantActuator, WebhookActuator, create_actuator,
    register_handler,
)
from whispy.actuators import device as device_mod


def _pred(label="occupied", confidence=0.95):
    return Prediction(label=label, confidence=confidence, device_id="d1")


# -- gating -----------------------------------------------------------------

def test_confidence_gate_fails_explicitly():
    action = Action(type="webhook", config={"url": "https://x"},
                    min_confidence=0.9)
    result = WebhookActuator({}).trigger(action, _pred(confidence=0.5))
    assert result.status is ActionStatus.FAILED
    assert "confidence" in result.detail


def test_label_filter_unsupported():
    action = Action(type="webhook", config={"url": "https://x"},
                    trigger_labels=["fall_detected"])
    result = WebhookActuator({}).trigger(action, _pred(label="occupied"))
    assert result.status is ActionStatus.UNSUPPORTED


# -- webhook ------------------------------------------------------------------

def test_webhook_missing_url_fails():
    action = Action(type="webhook", config={})
    result = WebhookActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.FAILED


def test_webhook_success_on_2xx(monkeypatch):
    class _Res:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"ok"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Res())
    action = Action(type="webhook", config={"url": "https://hooks.example/x"})
    result = WebhookActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.SUCCEEDED
    assert result.response["status"] == 200


def test_webhook_http_error_is_failed(monkeypatch):
    def _raise(*a, **k):
        raise urllib.error.HTTPError("https://x", 500, "boom", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    action = Action(type="webhook", config={"url": "https://x"})
    result = WebhookActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.FAILED
    assert result.response["status"] == 500


def test_webhook_body_template(monkeypatch):
    sent = {}
    class _Res:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"ok"
    def _capture(req, **k):
        sent["body"] = json.loads(req.data.decode())
        return _Res()
    monkeypatch.setattr(urllib.request, "urlopen", _capture)
    action = Action(type="webhook", config={
        "url": "https://x",
        "body": {"text": "{label} on {device_id}"}})
    WebhookActuator({}).trigger(action, _pred())
    assert sent["body"]["text"] == "occupied on d1"


# -- home assistant -------------------------------------------------------------

def test_ha_requires_url_and_service():
    action = Action(type="home_assistant", config={})
    result = HomeAssistantActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.FAILED


def test_ha_calls_service_endpoint(monkeypatch):
    sent = {}
    class _Res:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"[]"
    def _capture(req, **k):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        return _Res()
    monkeypatch.setattr(urllib.request, "urlopen", _capture)
    action = Action(type="home_assistant", config={
        "base_url": "http://ha.local:8123", "token": "t",
        "entity_id": "light.desk", "action": "light.turn_on"})
    result = HomeAssistantActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.SUCCEEDED
    assert "/api/services/light/turn_on" in sent["url"]
    assert sent["body"]["entity_id"] == "light.desk"


# -- device ---------------------------------------------------------------------

def test_device_gpio_unsupported_without_gpiozero():
    action = Action(type="device",
                    config={"action": "gpio_toggle", "pin": 18})
    result = DeviceActuator({}).trigger(action, _pred())
    # No gpiozero on Windows dev → must be unsupported, never fake success
    assert result.status in (ActionStatus.UNSUPPORTED, ActionStatus.FAILED)
    assert result.status is not ActionStatus.SUCCEEDED


def test_device_unknown_action_unsupported():
    action = Action(type="device", config={"action": "teleport"})
    result = DeviceActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.UNSUPPORTED


def test_device_handler_confirms(monkeypatch):
    calls = []
    register_handler("test_relay", lambda a, p: calls.append(p.label))
    try:
        action = Action(type="device", config={"action": "test_relay"})
        result = DeviceActuator({}).trigger(action, _pred())
        assert result.status is ActionStatus.SUCCEEDED
        assert calls == ["occupied"]
    finally:
        device_mod._HANDLERS.pop("test_relay", None)


def test_device_handler_exception_is_failed():
    def _boom(a, p):
        raise RuntimeError("hardware gone")
    register_handler("bad_relay", _boom)
    try:
        action = Action(type="device", config={"action": "bad_relay"})
        result = DeviceActuator({}).trigger(action, _pred())
        assert result.status is ActionStatus.FAILED
        assert "hardware gone" in result.detail
    finally:
        device_mod._HANDLERS.pop("bad_relay", None)


def test_device_shell_success():
    action = Action(type="device",
                    config={"action": "shell",
                            "command": ["python", "-c", "print(1)"]})
    result = DeviceActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.SUCCEEDED
    assert result.response["returncode"] == 0


# -- retry ------------------------------------------------------------------

def test_retry_policy_attempts(monkeypatch):
    calls = {"n": 0}
    def _flaky(req, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("down")
        class _Res:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"ok"
        return _Res()
    monkeypatch.setattr(urllib.request, "urlopen", _flaky)
    action = Action(type="webhook", config={"url": "https://x"},
                    retry_policy={"max_attempts": 3, "backoff_seconds": 0})
    from whispy.contracts import RetryPolicy
    action.retry_policy = RetryPolicy(max_attempts=3)
    result = WebhookActuator({}).trigger(action, _pred())
    assert result.status is ActionStatus.SUCCEEDED
    assert result.attempts == 3


def test_create_actuator_dispatch():
    assert isinstance(create_actuator(Action(type="webhook")), WebhookActuator)
    assert isinstance(create_actuator(Action(type="home_assistant")),
                      HomeAssistantActuator)
    assert isinstance(create_actuator(Action(type="device")), DeviceActuator)
    with pytest.raises(ValueError):
        create_actuator(Action(type="pigeon"))
