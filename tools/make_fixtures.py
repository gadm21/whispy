#!/usr/bin/env python3
"""Write the canonical contract fixtures under contracts/fixtures/.

These fixtures are the shared cross-language test vectors (Python,
TypeScript, Dart must all parse/serialize them identically).
Regenerate: python tools/make_fixtures.py
"""

import json
from pathlib import Path

FX = Path(__file__).resolve().parent.parent / "contracts" / "fixtures"

FIXTURES = {
    "observation": {
        "observation_id": "obs-1", "source_id": "camera-f91a",
        "device_id": "local-1", "timestamp": 1790242800.5,
        "schema": "physical.frame/v1", "sequence": 7,
        "payload_type": "jpeg", "payload": "<bytes>",
        "sample_rate": 30.0, "units": {},
        "quality": {"exposure": "ok"},
        "provenance": {"adapter": "opencv-camera"}, "metadata": {},
        "confidence": None, "accuracy": None, "spatial_reference": None,
        "privacy_classification": "sensitive_raw",
    },
    "source-descriptor": {
        "id": "camera-f91a", "modality": "camera", "adapter": "opencv-camera",
        "name": "Integrated Camera", "hardware_id": "USB\\VID_1234",
        "capabilities": ["frames"], "config_schema": {}, "stable": True,
        "source_class": "sensor", "health": {"status": "ok"}, "metadata": {},
    },
    "device-descriptor": {
        "id": "local-abc", "stable_uuid": "local-abc", "name": "laptop",
        "platform": "windows", "architecture": "AMD64", "online": True,
        "capabilities": [], "sensors": [], "health": {},
        "compute": {
            "architecture": "AMD64", "logical_cpu_count": 16,
            "memory_total_mb": 32768, "memory_available_mb": 12000,
            "gpu": [], "accelerators": [], "vram_mb": None,
            "storage_available_mb": 200000, "battery": {"percent": 80},
            "charging": True, "thermal": None, "network": {"lan": True},
        },
    },
    "compute-capability": {
        "architecture": "AMD64", "logical_cpu_count": 16,
        "memory_total_mb": 32768, "memory_available_mb": 12000,
        "gpu": [], "accelerators": [], "vram_mb": None,
        "storage_available_mb": 200000, "battery": {"percent": 80},
        "charging": True, "thermal": None, "network": {"lan": True},
    },
    "prediction": {
        "id": "pred-1", "device_id": "local-abc", "runtime_model_id": "rm-1",
        "task": "person_presence", "timestamp": 1790242801.0,
        "label": "person", "confidence": 0.9, "scores": {"person": 0.9},
        "source_window": None, "people_count": 1, "metadata": {},
    },
    "action-request": {
        "operation": "speak", "params": {"text": "hello"},
        "timeout_seconds": 10.0,
    },
    "action-result": {
        "status": "succeeded", "action_type": "speaker", "detail": "played",
        "attempts": 1, "started_at": 1790242800.0,
        "finished_at": 1790242801.0, "response": {},
    },
    "inference-request": {
        "request_id": "req-1", "model_id": "whisper-stt",
        "bindings": {"audio": "microphone-ab12"},
        "policy": {
            "allowed": ["local", "trusted_edge"],
            "preferred": ["trusted_edge"], "max_latency_ms": 5000,
            "max_cost": None, "queue_if_unavailable": False, "privacy": {},
        },
        "source_device": "pi1", "target": None,
        "window_seconds": 4.0, "config": {},
    },
    "inference-trace": {
        "model_id": "whisper-stt", "model_version": "1.0.0",
        "artifact_hash": "abc123", "runtime_id": "rt-1",
        "execution_device": "laptop", "execution_class": "trusted_edge",
        "input_bindings": {"audio": "microphone-ab12"},
        "input_interval": {"start": 1790242800.0, "end": 1790242804.0},
        "inference_timestamp": 1790242805.0, "latency_ms": 800.0,
        "confidence": 0.95, "cpu_percent": None, "gpu_percent": None,
        "estimated_cost": None, "actual_cost": None,
    },
    "inference-result": {
        "request_id": "req-1", "status": "succeeded", "error": "",
        "prediction": {
            "id": "p1", "label": "seven", "confidence": 0.9,
            "device_id": "", "runtime_model_id": "whisper-stt",
            "task": "stt", "timestamp": 1790242805.0, "scores": {},
            "source_window": None, "people_count": None, "metadata": {},
        },
        "trace": {"model_id": "whisper-stt", "execution_class": "trusted_edge"},
        "outputs": {"transcript": "seven"},
    },
    "context-evidence": {
        "id": "ev-1", "key": "spatial.presence/v1", "value": "occupied",
        "timestamp": 1790242800.0, "source_id": "radar-main",
        "device_id": "pi2", "prediction_id": "pred-1", "observation_id": "",
        "model_id": "occupancy-v2", "model_version": "2.1",
        "confidence": 0.95, "execution_class": "local", "provenance": {},
    },
    "context-state": {
        "id": "st-1", "key": "semantic.working/v1", "value": "working",
        "confidence": 0.88, "since": 1790242400.0, "entity_id": "person:gad",
        "evidence_ids": ["ev-1", "ev-2"], "estimator": "weighted-v1",
        "valid_until": None,
    },
    "context-event": {
        "id": "cev-1", "key": "spatial.occupancy/v1", "event_type": "entered",
        "timestamp": 1790242800.0, "entity_id": "space:lab",
        "state_id": "st-9", "value": "occupied", "previous_value": "vacant",
        "confidence": 0.93, "provenance": {},
    },
    "relationship": {
        "id": "rel-1", "subject": "person:gad", "predicate": "carries",
        "object": "device:phone", "valid_from": 1790240000.0,
        "valid_until": None, "confidence": 0.91, "source": "ble-proximity",
        "provenance": {},
    },
    "minute-manifest": {
        "format": "thoth-minute/v1", "minute_id": "20260924_1014",
        "device_id": "pi2", "start_timestamp": 1790242440.0,
        "end_timestamp": 1790242500.0, "duration_seconds": 60.0,
        "sources": [{
            "source_id": "radar-main", "modality": "radar",
            "timestamps": [1790242440.1, 1790242441.1],
            "values": [[1, 2], [3, 4]], "second_offsets": [0.1, 1.1],
            "units": {}, "quality": {}, "metadata": {},
        }],
        "predictions": [], "events": [], "annotations": [], "labels": {},
        "quality": {}, "source_metadata": {},
        "files": {"npz": "capture.npz"}, "checksums": {}, "metadata": {},
    },
}


def main() -> int:
    FX.mkdir(parents=True, exist_ok=True)
    for name, data in FIXTURES.items():
        (FX / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
