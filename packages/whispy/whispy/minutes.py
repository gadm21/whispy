"""Canonical minute reading — ``thoth-minute/v1`` (§9/§10).

Reads both canonical ``thoth-minute/v1`` manifests and every legacy
chunk-oriented manifest (``thoth-minute-manifest/v7`` and earlier),
normalizing in memory to :class:`MinuteManifest`.

Legacy terminology mapping:

- ``expected_chunks`` / ``expected_seconds`` → expected second count
- ``chunk_index`` / ``second_index``         → second offset index
- ``chunks`` / ``seconds``                   → per-second entries
- ``chunk_seconds``                          → seconds-per-entry granularity

Low-level byte/frame buffering inside drivers may still be called
"chunks"; this module only removes chunk as the *domain* temporal unit.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    MINUTE_MANIFEST_FORMAT, MinuteManifest, MinuteSourceData,
)

MINUTE_DIR_RE = re.compile(r"^\d{8}_\d{4}$")
LEGACY_MANIFEST_SCHEMAS = frozenset({
    "thoth-minute-manifest/v7", "thoth-minute-manifest/v6",
    "thoth-minute-manifest/v5",
})

#: Legacy chunk-era keys normalized at the compatibility boundary. The
#: canonical model speaks seconds/minutes only — these renames are applied
#: recursively to every legacy-derived payload before it can leave this
#: module, so domain-level ``chunk`` terminology never reaches API output.
_LEGACY_KEY_MAP = {
    "chunk_index": "second_index",
    "expected_chunks": "expected_seconds",
    "stored_chunks": "stored_seconds",
    "analyzed_chunks": "analyzed_seconds",
    "chunk_seconds": "seconds_per_entry",
    "chunk_count": "second_count",
    "chunks": "seconds",
}


def _strip_chunk_terms(value: Any) -> Any:
    """Recursively rename legacy chunk keys → second terminology.

    Applied to every legacy-derived structure (progress blocks, prediction
    timelines, per-source metadata) so the canonical MinuteManifest never
    carries domain-level ``chunk`` fields. Driver-internal byte buffering
    may still use the word; it must not appear here.
    """
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            new_key = _LEGACY_KEY_MAP.get(key, key)
            out[new_key] = _strip_chunk_terms(item)
        return out
    if isinstance(value, list):
        return [_strip_chunk_terms(v) for v in value]
    return value


def _parse_iso(value: Any) -> Optional[float]:
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError):
        return None


def minute_id_timestamp(minute_id: str) -> Optional[float]:
    """Epoch seconds encoded by a ``YYYYMMDD_HHMM`` minute id (local tz)."""
    try:
        return _dt.datetime.strptime(minute_id, "%Y%m%d_%H%M").timestamp()
    except (ValueError, TypeError):
        return None


def _legacy_seconds(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-second entries from a legacy manifest (chunks or seconds)."""
    radar = (manifest.get("outputs") or {}).get("radar") or {}
    entries = radar.get("seconds") or radar.get("chunks") or []
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("second_index", entry.get("chunk_index"))
        out.append({
            "second_index": int(idx) if idx is not None else None,
            "status": entry.get("status"),
            "occupied": entry.get("occupied"),
            "classification": entry.get("classification"),
            "score": entry.get("score"),
            "ratio": entry.get("ratio"),
            "location": entry.get("location"),
            "detected_frames": entry.get("detected_frames"),
            "evaluated_frames": entry.get("evaluated_frames"),
            "error": entry.get("error"),
        })
    return out


def _legacy_predictions(minute_dir: Path,
                        manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize predictions.json timeline + manifest model_predictions."""
    out: List[Dict[str, Any]] = []
    pred_path = minute_dir / "predictions.json"
    if pred_path.exists():
        try:
            payload = json.loads(pred_path.read_text(encoding="utf-8"))
            for entry in (payload.get("timeline") or []):
                if not isinstance(entry, dict):
                    continue
                e = dict(entry)
                # Normalize chunk_index → second_index.
                if "second_index" not in e and "chunk_index" in e:
                    e["second_index"] = e.pop("chunk_index")
                out.append(e)
        except Exception:
            pass
    for entry in (manifest.get("model_predictions") or []):
        if isinstance(entry, dict):
            e = dict(entry)
            if "second_index" not in e and "chunk_index" in e:
                e["second_index"] = e.pop("chunk_index")
            out.append(e)
    return out


def read_minute(minute_dir: Any) -> MinuteManifest:
    """Read one minute directory into a canonical :class:`MinuteManifest`.

    Accepts canonical ``thoth-minute/v1`` manifests (``minute.json``, as
    written by :func:`write_minute_manifest`) directly and legacy v5–v7
    manifests (``manifest.json``) via normalization. When both exist the
    canonical file wins — it is the migrated source of truth. Never
    mutates files on disk.
    """
    minute_dir = Path(minute_dir)
    canonical_path = minute_dir / "minute.json"
    if canonical_path.exists():
        try:
            data = json.loads(canonical_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            return MinuteManifest.from_dict(data)

    manifest_path = minute_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}

    minute_id = str(manifest.get("minute_id") or manifest.get("folder_minute")
                    or minute_dir.name)

    start = (_parse_iso(manifest.get("capture_started"))
             or _parse_iso(manifest.get("scheduled_start"))
             or minute_id_timestamp(minute_id) or 0.0)
    end = _parse_iso(manifest.get("capture_finished"))
    duration = manifest.get("duration_seconds")
    if duration is None and start and end:
        duration = end - start

    # -- sources -------------------------------------------------------------
    sources: List[MinuteSourceData] = []
    outputs = manifest.get("outputs") or {}
    if isinstance(outputs, dict):
        for key, out in outputs.items():
            if not isinstance(out, dict):
                continue
            if key == "wifi_csi":
                receivers = out.get("receivers") or [out]
                for i, rx in enumerate(receivers, 1):
                    if not isinstance(rx, dict):
                        continue
                    sources.append(MinuteSourceData(
                        source_id=str(rx.get("device_id") or f"csi-{i}"),
                        modality="csi",
                        metadata={
                            "port": rx.get("device") or rx.get("port"),
                            "sample_count": rx.get("sample_count")
                                            or rx.get("samples"),
                            "average_sampling_rate_hz":
                                rx.get("average_sampling_rate_hz"),
                        },
                    ))
                continue
            sources.append(MinuteSourceData(
                source_id=str(out.get("device_id") or out.get("device") or key),
                modality=key,
                metadata={
                    "sample_count": out.get("sample_count"),
                    "average_sampling_rate_hz":
                        out.get("average_sampling_rate_hz"),
                    "files": out.get("files"),
                },
            ))

    seconds = _legacy_seconds(manifest)
    predictions = _legacy_predictions(minute_dir, manifest)

    files: Dict[str, str] = {}
    for name in ("capture.npz", "usb_camera.mp4", "predictions.json",
                 "xy-tracking.json", "wifi_csi.csv"):
        if (minute_dir / name).exists():
            key = {"capture.npz": "npz", "usb_camera.mp4": "video",
                   "predictions.json": "predictions"}.get(name, name)
            files[key] = name

    return MinuteManifest(
        minute_id=minute_id,
        device_id=str(manifest.get("device_id") or manifest.get("host") or ""),
        start_timestamp=start,
        end_timestamp=end,
        duration_seconds=duration,
        sources=sources,
        predictions=predictions,
        events=[],
        annotations=[],
        labels={"labels": manifest.get("labels") or []},
        quality={
            "status": manifest.get("status"),
            "warnings": manifest.get("warnings") or [],
            "errors": manifest.get("errors") or [],
            "expected_seconds": (manifest.get("expected_seconds")
                                 or manifest.get("expected_chunks")),
            "seconds": seconds,
        },
        source_metadata=_strip_chunk_terms({
            "schema": manifest.get("schema"),
            "sensors_enabled": manifest.get("sensors_enabled") or [],
            "container": manifest.get("container"),
            "device_name": manifest.get("device_name"),
        }),
        files=files,
        checksums={},
        metadata={"legacy_schema": manifest.get("schema"),
                  "progress": _strip_chunk_terms(
                      manifest.get("progress") or {})},
    )


def write_minute_manifest(minute_dir: Any, manifest: MinuteManifest) -> Path:
    """Write a canonical ``thoth-minute/v1`` manifest (new writers only)."""
    minute_dir = Path(minute_dir)
    path = minute_dir / "minute.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2),
                    encoding="utf-8")
    return path


def iter_minute_dirs(root: Any) -> List[Path]:
    """All minute directories under a capture root (label subdirs included)."""
    root = Path(root)
    out: List[Path] = []
    if not root.exists():
        return out
    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        if MINUTE_DIR_RE.match(item.name):
            out.append(item)
            continue
        for child in sorted(item.iterdir()):
            if child.is_dir() and MINUTE_DIR_RE.match(child.name):
                out.append(child)
    return out


__all__ = [
    "MINUTE_DIR_RE", "LEGACY_MANIFEST_SCHEMAS", "read_minute",
    "write_minute_manifest", "iter_minute_dirs", "minute_id_timestamp",
]
