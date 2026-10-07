from whispy.beacons import decode_beacon

IBEACON_MFG = ("4c00" + "0215" + "e2c56db5dffb48d2b060d0f5a71096e0"
               + "0001" + "0002" + "c5")          # company LE + ibeacon


def test_ibeacon():
    out = decode_beacon({"mfg": IBEACON_MFG, "rssi": -55})
    assert out == {
        "kind": "ibeacon",
        "uuid": "e2c56db5-dffb-48d2-b060-d0f5a71096e0",
        "major": 1, "minor": 2, "measured_power": -59,
    }


def test_thoth_identity():
    # 0xffff LE + b"th:gad"
    out = decode_beacon({"mfg": "ffff" + "74683a676164"})
    assert out == {"kind": "thoth_identity", "owner": "gad"}


def test_altbeacon():
    data = bytes.fromhex("beac") + bytes(range(20)) + bytes([0xc5, 0x00])
    out = decode_beacon({"mfg": "ffff" + data.hex()})
    # company 0xffff checked first → no th: prefix → altbeacon path
    assert out is not None and out["kind"] == "altbeacon"


def test_eddystone_service_data():
    uid = "00" + "c5" + "00112233445566778899" + "aabbccddeeff" + "0000"
    out = decode_beacon({"service_data": {"0000feaa-0000-1000-8000-00805f9b34fb": uid}})
    assert out is not None
    assert out["kind"] == "eddystone_uid"
    assert out["namespace"] == "00112233445566778899"
    assert out["measured_power"] == -59


def test_eddystone_url():
    # frame 0x10, tx 0xc5, prefix https:// (0x03), "x" + .com suffix 0x07
    out = decode_beacon({"service_data": {"feaa": "10c503780 7".replace(" ", "")}})
    assert out is not None
    assert out["kind"] == "eddystone_url"
    assert out["url"] == "https://x.com"


def test_eddystone_uuid_only():
    out = decode_beacon({"service_uuids": ["0000feaa-0000-1000-8000-00805f9b34fb"]})
    assert out == {"kind": "eddystone"}


def test_non_beacon():
    assert decode_beacon({"mfg": "ffff1234"}) is None
    assert decode_beacon({"mfg": None, "name": "random"}) is None
    assert decode_beacon({}) is None
    assert decode_beacon({"mfg": "zz"}) is None
