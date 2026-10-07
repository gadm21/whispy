from whispy_sensor_zigbee import (
    ZigbeeSensorAdapter, _ha_zigbee_entities, _parse_line,
)


def test_parse_zb_data():
    out = _parse_line(b'ZB_DATA,1205,0x1a2b,00158d0001ab2cd3,0006,-62,110,"Desk lamp"')
    assert out["type"] == "zigbee_device"
    d = out["data"]
    assert d["ieee"] == "00158d0001ab2cd3"
    assert d["addr16"] == "0x1a2b"
    assert d["lqi"] == 110 and d["rssi"] == -62
    assert d["name"] == "Desk lamp"


def test_parse_join_leave():
    assert _parse_line(b'ZB_JOIN,9,0x0001,aabbccddeeff0011,"Plug"')["type"] == "zigbee_join"
    assert _parse_line(b'ZB_LEAVE,9,0x0001,aabbccddeeff0011')["type"] == "zigbee_leave"
    assert _parse_line(b'noise line') is None
    assert _parse_line(b'ZB_DATA,short') is None


def test_ha_entity_filter():
    states = [
        {"entity_id": "light.desk", "state": "on",
         "last_changed": "t",
         "attributes": {"friendly_name": "Desk", "ieee": "0x00158d0001ab2cd3", "lqi": 96}},
        {"entity_id": "light.plain", "state": "on",
         "attributes": {"friendly_name": "Plain"}},
    ]
    out = _ha_zigbee_entities(states)
    assert len(out) == 1
    assert out[0]["ieee"] == "0x00158d0001ab2cd3"
    assert out[0]["lqi"] == 96


def test_discover_from_sources():
    a = ZigbeeSensorAdapter(sources=[
        {"type": "ha", "ha_url": "http://h:8123", "ha_token": "t"},
        {"type": "serial", "serial_port": "/dev/ttyUSB9"},
        {"type": "bogus"},
    ])
    descs = a.discover()
    assert len(descs) == 2
    assert {d.metadata["type"] for d in descs} == {"ha", "serial"}
    assert all(d.modality == "zigbee" and d.stable for d in descs)


def test_discover_empty_without_config():
    assert ZigbeeSensorAdapter(sources=[]).discover() == []


def test_ha_state_transition_sample():
    a = ZigbeeSensorAdapter(sources=[])
    desc = a.discover()
    # build a handle manually against a fake HA source
    from whispy_sensor_zigbee import _ZigbeeHandle
    from whispy import SensorDescriptor
    d = SensorDescriptor(id="zigbee-test", modality="zigbee",
                         hardware_id="ha:x",
                         metadata={"type": "ha", "ha_url": "http://x",
                                   "ha_token": "t"})
    h = _ZigbeeHandle(d)
    import whispy_sensor_zigbee as z
    orig = z._ha_states
    z._ha_states = lambda u, t, timeout=10: [
        {"entity_id": "light.desk", "state": "on", "last_changed": "t",
         "attributes": {"ieee": "0xabc", "lqi": 50}},
    ]
    try:
        s = next(iter(h.stream(max_samples=1)))
        assert s.payload_type == "zigbee_state"  # first sighting = state change
        assert s.payload["ieee"] == "0xabc"
    finally:
        z._ha_states = orig
