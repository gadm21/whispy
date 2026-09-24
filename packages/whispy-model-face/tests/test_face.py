"""whispy-model-face tests — eigenface math, recognizer, conformance."""
import numpy as np
import pytest

from whispy.contracts import ModalityState, SensorSample, SensorWindow
from whispy.conformance import check_model_plugin
from whispy_model_face import EigenfaceRecognizer, HaarFaceModel
from whispy_model_face import basis as B


def _face(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.rand(size, size) * 40
    yy, xx = np.mgrid[0:size, 0:size]
    img += 120 * np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
                        / (2 * (size / (3 + seed % 3)) ** 2))
    return np.clip(img, 0, 255).astype(np.float32)


def _window(frame) -> SensorWindow:
    s = SensorSample.now("dev0", "cam0", "camera", frame,
                         payload_type="ndarray")
    return SensorWindow(start_timestamp=0.0, end_timestamp=1.0,
                        samples={"cam0": [s]},
                        modalities={"cam0": ModalityState(
                            sensor_id="cam0", state="ok")})


@pytest.fixture
def trained():
    """Basis + gallery fitted on two synthetic people."""
    people = {"gad": [_face(0), _face(0) + 2], "sara": [_face(3), _face(3) + 2]}
    all_imgs = [im for imgs in people.values() for im in imgs]
    basis = B.fit_basis(all_imgs, image_size=64, n_components=3)
    gallery = {name: [B.project(basis, im).tolist() for im in imgs]
               for name, imgs in people.items()}
    return basis, gallery


def test_fit_project_roundtrip():
    basis = B.fit_basis([_face(0), _face(1), _face(2)], n_components=2)
    assert basis["eigenvectors"].shape == (2, 64 * 64)
    proj = B.project(basis, _face(0))
    assert proj.shape == (2,)


def test_serialize_roundtrip(trained):
    basis, _ = trained
    blob = B.serialize_basis({**basis, "max_distance": 1.5})
    back = B.deserialize_basis(blob)
    np.testing.assert_allclose(back["mean"], basis["mean"])
    assert back["image_size"] == 64
    assert back["max_distance"] == 1.5


def test_recognizer_matches_enrolled(trained):
    basis, gallery = trained
    rec = EigenfaceRecognizer({
        "basis": basis, "gallery": gallery, "detect": False})
    pred = rec.predict(_window(_face(0)))          # gad's enrolled pattern
    assert pred.label == "person:gad"
    assert pred.metadata["matched"] is True
    assert pred.metadata["distance"] >= 0


def test_recognizer_rejects_unknown(trained):
    basis, gallery = trained
    rec = EigenfaceRecognizer({
        "basis": basis, "gallery": gallery,
        "max_distance": 0.5, "detect": False})     # tight threshold
    pred = rec.predict(_window(np.random.RandomState(99)
                             .rand(64, 64) * 255))  # noise ≠ enrolled
    assert pred.label == "person:unknown"
    assert pred.metadata["matched"] is False


def test_recognizer_no_basis_errors():
    rec = EigenfaceRecognizer({"detect": False})
    pred = rec.predict(_window(_face(0)))
    assert pred.label == "person:unknown"
    assert "no basis" in pred.metadata["error"]


def test_set_gallery_hotswap(trained):
    basis, gallery = trained
    rec = EigenfaceRecognizer({"basis": basis, "detect": False})
    rec.set_gallery(gallery, max_distance=100.0)
    assert rec.health()["persons"] == 2
    pred = rec.predict(_window(_face(3)))
    assert pred.label == "person:sara"


def test_conformance():
    assert check_model_plugin(HaarFaceModel)["passed"]
    assert check_model_plugin(EigenfaceRecognizer)["passed"]


def test_gallery_from_persons():
    g = B.gallery_from_persons([
        {"name": "gad", "projection": [1.0, 2.0]},
        {"name": "gad", "projection": [1.1, 2.1]},
        {"name": "sara", "projection": [9.0, 9.0]},
        {"name": "", "projection": [0.0]},          # skipped
    ])
    assert len(g["gad"]) == 2 and len(g["sara"]) == 1
