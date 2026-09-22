"""Radar frame decoding and range/Doppler helpers."""

from __future__ import annotations

import numpy as np

HEADER_BYTES = 12


def unpack_uint12(payload: bytes) -> np.ndarray:
    """Decode packed uint12 radar samples to uint16."""
    data = np.frombuffer(payload, dtype=np.uint8)
    if len(data) % 3:
        data = data[: len(data) - (len(data) % 3)]
    triples = data.reshape(-1, 3).astype(np.uint16)
    out = np.empty(len(triples) * 2, dtype=np.uint16)
    out[0::2] = (triples[:, 0] << 4) | (triples[:, 1] >> 4)
    out[1::2] = ((triples[:, 1] & 0x0F) << 8) | triples[:, 2]
    return out


def parse_frame(raw: bytes) -> dict:
    """Split a raw radar frame into header + uint12 payload."""
    return {
        "header": raw[:HEADER_BYTES],
        "samples": unpack_uint12(raw[HEADER_BYTES:]),
    }
