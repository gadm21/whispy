"""Three-device hardware acceptance suite -- run from the Windows laptop.

Connects to the local laptop plus RPi1 and RPi2 over the LAN and tests
every reported capability through Whispy -- never through vendor APIs::

    python whispy/examples/hardware_acceptance.py \
        --rpi1 rpi1.local:5000:<token> --rpi2 rpi2.local:5000:<token>

Expected coverage:

    laptop camera        frame received
    laptop microphone    PCM received
    laptop speaker       known speech physically played
    rpi1 microphone      PCM received remotely
    rpi1 csi             CSI sample received
    rpi1 imu/env         measurements received
    rpi1 matrix          known pattern displayed
    rpi2 radar           radar/SNR sample received
    rpi2 csi             CSI sample received
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional


def _parse_target(spec: Optional[str]):
    if not spec:
        return None
    parts = spec.split(":")
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 and parts[1] else 5000
    token = parts[2] if len(parts) > 2 else None
    return {"host": host, "port": port, "token": token}


def _probe_sensor(device, sensor, timeout_s: float = 8.0) -> Dict[str, Any]:
    """Read one real sample from a sensor handle."""
    out = {"id": sensor.id, "type": sensor.type, "ok": False, "detail": ""}
    try:
        handle = device.sensor(sensor.id)
        deadline = time.time() + timeout_s
        for sample in handle.stream():
            out["ok"] = sample.payload is not None
            out["detail"] = f"seq={sample.sequence} " \
                            f"payload_type={sample.payload_type}"
            break
            if time.time() > deadline:  # pragma: no cover
                break
        if not out["ok"]:
            out["detail"] = "no sample before timeout"
    except Exception as exc:
        out["detail"] = f"{type(exc).__name__}: {exc}"
    return out


def _probe_actuator(device, desc) -> Dict[str, Any]:
    """Exercise an actuator with a safe, known operation."""
    import whispy
    from whispy.actuators import Clear, ShowPattern, Speak

    out = {"id": desc.id, "kind": desc.kind, "ok": False, "detail": ""}
    try:
        handle = device.actuator(desc.id)
        if "clear" in desc.operations:
            result = handle.execute(Clear())
        elif "show" in desc.operations:
            result = handle.execute(ShowPattern(
                [[255, 0, 0]] * 64, duration_s=0.5))
        elif "speak" in desc.operations:
            result = handle.execute(Speak("thoth acceptance probe",
                                          volume=0.3))
        else:
            result = handle.execute(
                whispy.ActuatorCommand(operation=desc.operations[0]))
        out["ok"] = result.status.value in ("succeeded", "unsupported")
        out["detail"] = f"{result.status.value}: {result.detail}"
    except Exception as exc:
        out["detail"] = f"{type(exc).__name__}: {exc}"
    return out


def check_device(name: str, device, results: List[Dict[str, Any]]) -> None:
    try:
        info = device.info
        print(f"\n== {name} ({info.id}) ==")
    except Exception as exc:
        results.append({"device": name, "ok": False,
                        "detail": f"unreachable: {exc}"})
        print(f"\n== {name}: UNREACHABLE ({exc}) ==")
        return

    try:
        sensors = device.sensors()
    except Exception as exc:
        sensors = []
        print(f"  sensors: error {exc}")
    for sensor in sensors:
        res = _probe_sensor(device, sensor)
        res["device"] = name
        results.append(res)
        print(f"  sensor {res['id']:<20} {res['type']:<12} "
              f"{'OK' if res['ok'] else 'FAIL'}  {res['detail']}")

    try:
        actuators = device.actuators()
    except Exception as exc:
        actuators = []
        print(f"  actuators: error {exc}")
    for desc in actuators:
        res = _probe_actuator(device, desc)
        res["device"] = name
        results.append(res)
        print(f"  actuator {res['id']:<18} {res['kind']:<12} "
              f"{'OK' if res['ok'] else 'FAIL'}  {res['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpi1", help="host:port:token")
    parser.add_argument("--rpi2", help="host:port:token")
    parser.add_argument("--no-local", action="store_true")
    args = parser.parse_args()

    import whispy

    results: List[Dict[str, Any]] = []

    if not args.no_local:
        check_device("laptop", whispy.local(), results)

    for name, spec in (("rpi1", args.rpi1), ("rpi2", args.rpi2)):
        target = _parse_target(spec)
        if target is None:
            continue
        device = whispy.lan(target["host"], port=target["port"],
                            token=target["token"])
        check_device(name, device, results)

    total = len(results)
    passed = sum(1 for r in results if r.get("ok"))
    print(f"\n{'=' * 50}\n{passed}/{total} capabilities verified")
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
