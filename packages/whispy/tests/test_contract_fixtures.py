"""Cross-language contract fixtures (§64).

Every JSON fixture under ``contracts/fixtures/`` must round-trip through
the Python contract dataclasses. The same fixtures are parsed by the
generated TypeScript and Dart bindings — semantic drift fails CI.
"""

import json
from pathlib import Path

import pytest

from whispy import (
    ActionRequest, ActionResult, ComputeCapability, ContextEvent,
    ContextEvidence, ContextState, Device, InferenceRequest,
    InferenceResult, InferenceTrace, MinuteManifest, Observation,
    Prediction, Relationship, SourceDescriptor,
)

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "contracts" / "fixtures"

# fixture name -> (contract class, dict-key the class reads)
CONTRACTS = {
    "observation": Observation,
    "source-descriptor": SourceDescriptor,
    "device-descriptor": Device,
    "compute-capability": ComputeCapability,
    "prediction": Prediction,
    "action-request": ActionRequest,
    "action-result": ActionResult,
    "inference-request": InferenceRequest,
    "inference-trace": InferenceTrace,
    "inference-result": InferenceResult,
    "context-evidence": ContextEvidence,
    "context-state": ContextState,
    "context-event": ContextEvent,
    "relationship": Relationship,
    "minute-manifest": MinuteManifest,
}


def test_all_fixtures_have_contracts():
    files = {p.stem for p in FIXTURE_DIR.glob("*.json")}
    assert files == set(CONTRACTS), (
        f"fixture/contract drift: missing={set(CONTRACTS) - files} "
        f"unmapped={files - set(CONTRACTS)}")


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_fixture_round_trip(name):
    cls = CONTRACTS[name]
    raw = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    obj = cls.from_dict(raw)
    out = obj.to_dict()
    # Required schema fields must survive the round-trip.
    schema = json.loads(
        (FIXTURE_DIR.parents[0] / "schemas" / f"{name}.schema.json")
        .read_text())
    for req in schema.get("required") or []:
        assert req in out or req == "format", f"{name}: lost required {req}"
        if req in out:
            assert out[req] == raw[req], f"{name}: {req} mutated"
    # Re-parse the emitted dict — idempotent serialization.
    obj2 = cls.from_dict(out)
    assert obj2.to_dict() == out
