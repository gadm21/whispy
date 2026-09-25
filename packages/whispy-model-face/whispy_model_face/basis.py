"""Eigenface basis math — pure numpy, shared by Brain and edge devices.

Ported from gadm21/Face-recognition-using-PCA-and-SVD (reference impl,
Python 2.7): images are flattened to rows of a data matrix ``A``, the
mean face is subtracted, and PCA eigenvectors form the projection basis.
A face is recognized by projecting it into the basis and finding the
nearest stored projection by Euclidean distance.

The reference repo always returns the closest image — it has no
unknown-rejection. Here ``max_distance`` provides the "very close"
cutoff: distances above it yield ``person:unknown``.

Basis serialization is ``.npz`` bytes (mean, eigenvectors, image_size,
max_distance) so a basis trained anywhere — Brain ``/v1/faces/basis``
or a local dataset — travels as one artifact.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_IMAGE_SIZE = 64          # grayscale edge, matching Olivetti 64x64
DEFAULT_COMPONENTS = 40
DEFAULT_THRESHOLD_SIGMA = 2.0    # mean + k*std of enrolled self-distances


def normalize_image(image: Any, image_size: int = DEFAULT_IMAGE_SIZE
                    ) -> Optional[np.ndarray]:
    """Any grayscale/BGR/RGB ndarray → flattened float32 vector in [0,1].

    Returns ``None`` for undecodable input. Color is averaged to gray —
    the reference repo used BGR channels directly, but gray keeps the
    basis small (image_size² dims) and is what face datasets provide.
    """
    try:
        arr = np.asarray(image, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if arr.shape != (image_size, image_size):
        # Nearest-neighbor resize without cv2/PIL dependency.
        ys = (np.linspace(0, arr.shape[0] - 1, image_size)).astype(int)
        xs = (np.linspace(0, arr.shape[1] - 1, image_size)).astype(int)
        arr = arr[np.ix_(ys, xs)]
    if arr.max() > 1.5:            # 0-255 range → normalize
        arr = arr / 255.0
    return arr.astype(np.float32).flatten()


def fit_basis(images: Sequence[np.ndarray],
              image_size: int = DEFAULT_IMAGE_SIZE,
              n_components: int = DEFAULT_COMPONENTS
              ) -> Dict[str, Any]:
    """Fit an eigenface basis from flattened/normalizable images.

    Returns ``{mean, eigenvectors, image_size, n_components}`` where
    ``eigenvectors`` is (n_components, image_size²). Uses SVD on the
    centered data matrix — same math as ``cv2.PCACompute`` in the
    reference repo, without the OpenCV dependency.
    """
    rows = [normalize_image(im, image_size) for im in images]
    rows = [r for r in rows if r is not None]
    return fit_basis_normalized(rows, image_size, n_components)


def fit_basis_normalized(vectors: Sequence[np.ndarray],
                         image_size: int = DEFAULT_IMAGE_SIZE,
                         n_components: int = DEFAULT_COMPONENTS
                         ) -> Dict[str, Any]:
    """Fit a basis from already-normalized vectors (``normalize_image``
    output). Shared by Brain's enrollment path so the SVD math lives in
    exactly one place.
    """
    rows = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
    if len(rows) < 2:
        raise ValueError("need at least 2 decodable images to fit a basis")
    A = np.stack(rows)
    mean = A.mean(axis=0)
    centered = A - mean
    k = max(1, min(int(n_components), min(A.shape) - 1))
    # Economy SVD: eigenvectors of the covariance are the right singular
    # vectors of the centered data matrix.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return {
        "mean": mean.astype(np.float32),
        "eigenvectors": vt[:k].astype(np.float32),
        "image_size": int(image_size),
        "n_components": int(k),
    }


def project(basis: Dict[str, Any], image: Any) -> Optional[np.ndarray]:
    """Project an image into the basis → weight vector (n_components,)."""
    vec = normalize_image(image, int(basis["image_size"]))
    if vec is None:
        return None
    centered = vec - np.asarray(basis["mean"], dtype=np.float32)
    return np.dot(centered, np.asarray(basis["eigenvectors"]).T)


def distance(w1: Sequence[float], w2: Sequence[float]) -> float:
    """Euclidean distance between projections — the repo's find_distance."""
    a, b = np.asarray(w1, dtype=np.float32), np.asarray(w2, dtype=np.float32)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def nearest(gallery: Dict[str, List[Sequence[float]]],
            projection: Sequence[float]
            ) -> Tuple[Optional[str], float]:
    """Nearest gallery projection → (name, distance). None if empty."""
    best_name, best_d = None, float("inf")
    for name, projections in gallery.items():
        for stored in projections:
            d = distance(stored, projection)
            if d < best_d:
                best_name, best_d = name, d
    return best_name, best_d


def calibrate_threshold(gallery: Dict[str, List[Sequence[float]]],
                        sigma: float = DEFAULT_THRESHOLD_SIGMA) -> float:
    """Default ``max_distance``: mean + sigma·std of each enrolled
    projection's distance to its own person's centroid.

    The reference repo has no threshold; this is the calibrated stand-in
    for "very close". Tighten/loosen via the ``max_distance`` config.
    """
    dists: List[float] = []
    for projections in gallery.values():
        if not projections:
            continue
        arr = np.asarray(projections, dtype=np.float32)
        centroid = arr.mean(axis=0)
        dists.extend(float(np.sqrt(np.sum((p - centroid) ** 2)))
                     for p in arr)
    if not dists:
        return 0.0
    return float(np.mean(dists) + sigma * np.std(dists))


def serialize_basis(basis: Dict[str, Any]) -> bytes:
    """Basis dict → .npz bytes for storage/transport."""
    buf = io.BytesIO()
    np.savez(buf,
             mean=np.asarray(basis["mean"], dtype=np.float32),
             eigenvectors=np.asarray(basis["eigenvectors"],
                                     dtype=np.float32),
             image_size=np.int64(basis["image_size"]),
             max_distance=np.float64(basis.get("max_distance") or 0.0))
    return buf.getvalue()


def deserialize_basis(data: bytes) -> Dict[str, Any]:
    """.npz bytes → basis dict."""
    with np.load(io.BytesIO(data)) as z:
        return {
            "mean": z["mean"].astype(np.float32),
            "eigenvectors": z["eigenvectors"].astype(np.float32),
            "image_size": int(z["image_size"]),
            "n_components": int(z["eigenvectors"].shape[0]),
            "max_distance": float(z["max_distance"]),
        }


def gallery_from_persons(persons: Iterable[Dict[str, Any]]
                         ) -> Dict[str, List[List[float]]]:
    """Brain ``/v1/faces/gallery`` person rows → {name: [projection]}."""
    gallery: Dict[str, List[List[float]]] = {}
    for row in persons:
        name = str(row.get("name") or "")
        proj = row.get("projection")
        if name and proj:
            gallery.setdefault(name, []).append(list(proj))
    return gallery


__all__ = [
    "DEFAULT_IMAGE_SIZE", "DEFAULT_COMPONENTS", "DEFAULT_THRESHOLD_SIGMA",
    "normalize_image", "fit_basis", "project", "distance", "nearest",
    "calibrate_threshold", "serialize_basis", "deserialize_basis",
    "gallery_from_persons",
]
