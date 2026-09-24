"""Client helper tests — ContextCache TTL, DeviceRegistry, IntegrationRegistry."""
import pytest

from whispy.cloud import ContextCache, DeviceRegistry
from whispy.integrations import IntegrationRegistry


def test_context_cache_fetches_once_within_ttl():
    calls = []
    snap = {"entities": [{"id": "space:lab", "kind": "space"}],
            "relationships": [], "states": [
                {"key": "spatial.occupancy/v1", "value": "occupied",
                 "entity_id": "space:lab", "confidence": 0.9}],
            "generated_at": 1.0}
    cache = ContextCache(lambda: (calls.append(1), snap)[1], ttl_s=60)
    assert cache.state("spatial.occupancy/v1")["value"] == "occupied"
    assert cache.state("spatial.occupancy/v1")["value"] == "occupied"
    assert len(calls) == 1
    assert cache.entities("space")[0]["id"] == "space:lab"


def test_context_cache_stale_refresh_and_error():
    calls = []
    cache = ContextCache(lambda: (calls.append(1), {"states": []})[1],
                         ttl_s=0)
    cache.snapshot()
    cache.snapshot()
    assert len(calls) == 2  # ttl_s=0 → always stale

    def boom():
        raise RuntimeError("brain unreachable")
    cache = ContextCache(boom)
    cache.snapshot()
    assert cache.last_error == "brain unreachable"


def test_device_registry_fetcher_and_probe_unknown():
    reg = DeviceRegistry(fetcher=lambda: [
        {"id": "pi1", "name": "Pi 1", "ip_address": "10.0.0.88"}])
    reg.refresh()
    assert reg.get("pi1")["name"] == "Pi 1"
    assert len(reg.list()) == 1
    res = reg.probe("unknown-device")
    assert res["reachable"] is False


def test_integration_registry():
    reg = IntegrationRegistry()
    reg.register("home-assistant", "both",
                 config_schema={"url": "str", "token": "str"})
    with pytest.raises(ValueError):
        reg.register("bad", "sideways")
    entry = reg.enable("home-assistant", {"url": "http://ha:8123"})
    assert entry["enabled"] is True
    assert reg.disable("home-assistant")["enabled"] is False
    with pytest.raises(KeyError):
        reg.enable("nope", {})
