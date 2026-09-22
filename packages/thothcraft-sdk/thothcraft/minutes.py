"""Captured-minute decoding: radar, CSI, camera, environmental, predictions."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .errors import NotFoundError


def extract_predictions(metadata: dict) -> list:
    """Extract predictions from minute metadata or a live-chunk response."""
    result = []
    for key in ('model_predictions', 'minute_predictions'):
        value = metadata.get(key)
        if value:
            result.extend(value if isinstance(value, list) else [value])
    for chunk in metadata.get('chunks') or metadata.get('seconds') or []:
        if isinstance(chunk, dict):
            result.extend(extract_predictions(chunk))
    return result


def _blob_at(data: np.ndarray, offsets: np.ndarray, index: int) -> bytes:
    if index < 0 or index + 1 >= len(offsets):
        return b""
    return data[int(offsets[index]):int(offsets[index + 1])].tobytes()


class _Container:
    """Lazy decoder for a capture.npz byte string."""

    def __init__(self, content: bytes):
        self._archive = np.load(io.BytesIO(content), allow_pickle=False)
        raw = self._archive["metadata_json"].astype(np.uint8).tobytes()
        self.metadata = json.loads(raw.decode("utf-8"))

    def close(self):
        try:
            self._archive.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class SensorData:
    """A per-sensor series: parallel arrays of samples + timestamps.

    Attributes
    ----------
    samples : list of per-sample payloads (bytes for radar, str/dict for CSI/sense).
    unix_ns : np.ndarray of capture timestamps (ns), aligned with ``samples``.
    second_index : np.ndarray of per-sample second offsets within the minute.
    """

    def __init__(self, samples: list, unix_ns=None, second_index=None, extra: dict | None = None):
        self.samples = samples
        self.unix_ns = np.asarray(unix_ns) if unix_ns is not None else np.zeros(len(samples), dtype=np.int64)
        self.second_index = np.asarray(second_index) if second_index is not None else np.zeros(len(samples), dtype=np.int64)
        self.extra = extra or {}

    def __len__(self) -> int:
        return len(self.samples)

    def to_numpy(self, dtype=None) -> np.ndarray:
        """Stack numeric samples; decode radar bytes and CSI real/imag samples.

        Ragged samples require explicit padding before conversion. CSI output
        has shape (samples, 2, subcarriers), with real then imaginary channels.
        """
        rows = []
        for sample in self.samples:
            if isinstance(sample, bytes):
                if self.extra.get('sensor') == 'camera':
                    raise ValueError('Decode camera JPEG bytes before numeric conversion')
                from .sensors.radar import parse_frame
                sample = parse_frame(sample)['samples']
            elif isinstance(sample, dict):
                if 'real' in sample and 'imag' in sample:
                    sample = [sample['real'], sample['imag']]
                else:
                    sample = [sample[k] for k in sorted(sample)]
            rows.append(np.asarray(sample, dtype=dtype))
        return np.stack(rows) if rows else np.empty((0,), dtype=dtype or float)

    def to_dataframe(self):
        """Return flattened samples indexed by capture time (requires [pandas])."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError('Install thothcraft-sdk[pandas]') from exc
        values = self.to_numpy()
        values = values.reshape(len(self), -1) if len(self) else np.empty((0, 0))
        return pd.DataFrame(values, index=pd.Index(self.unix_ns, name='unix_ns'))

    def to_torch(self, dtype=None):
        """Return a CPU tensor (requires thothcraft-sdk[dl])."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError('Install thothcraft-sdk[dl]') from exc
        return torch.as_tensor(self.to_numpy(), dtype=dtype)

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self) -> str:
        return f"SensorData(n={len(self.samples)})"


class Minute:
    """A captured minute on a device, decoded from its capture.npz."""

    def __init__(self, http, minute_id: str, device_id: Optional[str] = None):
        self._http = http
        self.id = minute_id
        self.device_id = device_id
        self._container: _Container | None = None
        self._metadata: dict | None = None
        self._labels: dict = {}

    def _params(self, extra: dict | None = None) -> dict:
        p = {"device_id": self.device_id}
        if extra:
            p.update(extra)
        return p

    def _load_container(self) -> _Container:
        if self._container is None:
            content = self._http.get_bytes(
                f"/api/file/minute/{self.id}/download", self._params())
            with zipfile.ZipFile(io.BytesIO(content)) as bundle:
                matches = [n for n in bundle.namelist() if n.endswith('capture.npz')]
                if len(matches) != 1:
                    raise ValueError('Minute bundle must contain exactly one capture.npz')
                self._container = _Container(bundle.read(matches[0]))
        return self._container

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            try:
                payload = self._http.get_json(
                    f"/api/file/minute/{self.id}/container/metadata", self._params())
                self._metadata = payload.get("metadata") or payload
            except NotFoundError:
                self._metadata = self._load_container().metadata
        return self._metadata

    @property
    def predictions(self) -> list:
        md = self.metadata
        return extract_predictions(md)

    @property
    def labels(self) -> dict:
        return self._labels

    def label(self, **kwargs) -> None:
        """Attach labels to this minute (e.g. ``minute.label(occupied=True)``)."""
        self._labels.update(kwargs)

    # -- per-sensor accessors -------------------------------------------
    def radar(self) -> SensorData:
        """Raw radar frames (12-byte header + uint12 payload) + timestamps."""
        c = self._load_container()
        if c is not None:
            a = c._archive
            if "radar_sample_bytes" not in a.files:
                return SensorData([])
            payload, offsets = a["radar_sample_bytes"], a["radar_sample_offsets"]
            n = len(a["radar_sample_second_index"])
            samples = [_blob_at(payload, offsets, i) for i in range(n)]
            return SensorData(
                samples,
                unix_ns=a["radar_sample_unix_ns"] if "radar_sample_unix_ns" in a.files else None,
                second_index=a["radar_sample_second_index"],
                extra={"sequence": a["radar_sample_sequence"] if "radar_sample_sequence" in a.files else None},
            )

    def csi(self, parse: bool = True) -> SensorData:
        """CSI samples; ``parse=True`` decodes subcarrier amplitudes."""
        c = self._load_container()
        if c is not None:
            a = c._archive
            if "csi_sample_bytes" not in a.files:
                return SensorData([])
            payload, offsets = a["csi_sample_bytes"], a["csi_sample_offsets"]
            n = len(a["csi_sample_second_index"])
            lines = [_blob_at(payload, offsets, i).decode("utf-8", "replace") for i in range(n)]
            receivers = a["csi_sample_receiver_index"] if "csi_sample_receiver_index" in a.files else None
            samples: list[Any] = lines
            if parse:
                from .sensors.csi import parse_csi_line
                parsed = []
                for i, line in enumerate(lines):
                    row = parse_csi_line(line)
                    if row is not None and receivers is not None:
                        row["receiver_index"] = int(receivers[i])
                    parsed.append(row if row is not None else line)
                samples = parsed
            return SensorData(
                samples,
                unix_ns=a["csi_sample_unix_ns"] if "csi_sample_unix_ns" in a.files else None,
                second_index=a["csi_sample_second_index"],
                extra={"receiver_index": receivers},
            )

    def camera(self) -> list:
        """JPEG bytes, one per captured second (empty where no frame)."""
        c = self._load_container()
        if c is not None:
            a = c._archive
            if "camera_jpeg_bytes" not in a.files:
                return []
            present = a["camera_present"]
            payload, offsets = a["camera_jpeg_bytes"], a["camera_jpeg_offsets"]
            return [_blob_at(payload, offsets, i) if bool(present[i]) else b""
                    for i in range(len(present))]

    def sense(self) -> list:
        """Parsed environmental sensor JSON rows."""
        c = self._load_container()
        if c is not None:
            a = c._archive
            if "sense_sample_bytes" not in a.files:
                return []
            payload, offsets = a["sense_sample_bytes"], a["sense_sample_offsets"]
            n = len(a["sense_sample_second_index"])
            rows = []
            for i in range(n):
                try:
                    row = json.loads(_blob_at(payload, offsets, i).decode("utf-8", "replace"))
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            return rows

    # -- persistence -----------------------------------------------------
    def download(self, out_dir: str | Path = ".") -> Path:
        """Download the raw minute bundle (zip) to ``out_dir/<id>.zip``.

        Requires the ``download_data`` entitlement (Home/Research).
        """
        content = self._http.get_bytes(
            f"/api/file/minute/{self.id}/download", self._params())
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.id}.zip"
        path.write_bytes(content)
        return path

    def __repr__(self) -> str:
        return f"Minute({self.id}, device={self.device_id})"

    def __getitem__(self, sensor: str) -> SensorData:
        accessors = {'radar': self.radar, 'csi': self.csi,
                     'camera': self.camera, 'sense': self.sense}
        if sensor not in accessors:
            raise KeyError(sensor)
        data = accessors[sensor]()
        return data if isinstance(data, SensorData) else SensorData(data, extra={'sensor': sensor})

    def __iter__(self):
        return iter(('radar', 'csi', 'camera', 'sense'))

    def __len__(self) -> int:
        return 4

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
