"""Camera frame helpers."""

from __future__ import annotations


def present_frames(jpegs: list) -> list:
    """Filter a per-second JPEG list down to seconds with a real frame."""
    return [j for j in jpegs if j]


def save_frames(jpegs: list, out_dir: str, prefix: str = "frame") -> list:
    """Write non-empty JPEG frames to disk; returns written paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, jpg in enumerate(jpegs):
        if not jpg:
            continue
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.jpg")
        with open(path, "wb") as fh:
            fh.write(jpg)
        paths.append(path)
    return paths
