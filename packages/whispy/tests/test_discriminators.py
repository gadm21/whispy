"""Built-in discriminators — guided + auto calibration over descriptors."""
import random

import pytest

from whispy.discriminators import (
    RECIPES, CalibrationSession, Discriminator, DiscriminatorBank,
    features_from_descriptors,
)


def _radar_desc(std, mean=10.0):
    return {"radar-1": {"type": "radar", "n": 20, "fields": {
        "snr_db": {"mean": mean, "std": std, "min": 0, "max": 20},
        "range_profile_mean": {"mean": 1.0, "std": std / 2,
                               "min": 0, "max": 2}}}}


def _windows(rng, std_center, n, mean=10.0):
    return [_radar_desc(max(0.01, rng.gauss(std_center, 0.2)),
                        mean=rng.gauss(mean, 0.5)) for _ in range(n)]


def test_features_flatten_and_average_same_type():
    d = {"a": {"type": "radar", "fields": {"snr_db": {"mean": 2, "n": 5}}},
         "b": {"type": "radar", "fields": {"snr_db": {"mean": 4}}},
         "c": {"type": "ble_scan", "scan": {"emitters": 3}},
         "d": {"type": None, "n": 0}}
    f = features_from_descriptors(d)
    assert f == {"radar.snr_db.mean": 3.0, "ble_scan.emitters": 3.0}


def test_uncalibrated_never_guesses():
    disc = Discriminator(RECIPES["occupancy-radar"])
    assert disc.predict(_radar_desc(1.0)) is None


def test_guided_session_calibrates_and_separates():
    rng = random.Random(0)
    disc = Discriminator(RECIPES["occupancy-radar"])
    sess = CalibrationSession(disc)
    assert sess.current.label == "empty"
    for w in _windows(rng, 0.5, 30):
        sess.record(w)
    assert sess.next_step().label == "occupied"
    for w in _windows(rng, 3.0, 30, mean=12.0):
        sess.record(w)
    assert sess.next_step() is None
    assert sess.status()["recorded"] == {"empty": 30, "occupied": 30}
    cal = sess.finish()
    assert cal.method == "guided" and cal.windows == 60
    p = disc.predict(_radar_desc(3.2, mean=12.0), device_id="n1")
    assert p.label == "occupied" and p.confidence > 0.9
    assert p.runtime_model_id == "builtin:occupancy-radar"
    assert p.task == "occupancy" and "z" in p.metadata
    assert disc.predict(_radar_desc(0.4)).label == "empty"


def test_auto_calibration_two_means():
    rng = random.Random(1)
    hist = [features_from_descriptors(w) for w in
            _windows(rng, 0.5, 40) + _windows(rng, 3.0, 40)]
    disc = Discriminator(RECIPES["occupancy-radar"])
    cal = disc.calibrate_auto(hist)
    assert cal.method == "auto"
    assert disc.predict(_radar_desc(3.1)).label == "occupied"
    assert disc.predict(_radar_desc(0.5)).label == "empty"


def test_auto_calibration_needs_enough_varied_windows():
    disc = Discriminator(RECIPES["occupancy-radar"])
    with pytest.raises(ValueError):
        disc.calibrate_auto([{"radar.snr_db.std": 1.0}] * 5)
    with pytest.raises(ValueError):
        disc.calibrate_auto([{"radar.snr_db.std": 1.0}] * 100)


def test_guided_requires_every_label():
    disc = Discriminator(RECIPES["occupancy-radar"])
    with pytest.raises(ValueError):
        disc.calibrate_guided({"empty": [{"radar.snr_db.std": 0.5}]})


def test_bank_persists_and_predicts(tmp_path):
    rng = random.Random(2)
    bank = DiscriminatorBank(tmp_path)
    assert {d["name"] for d in bank.to_list()} == set(RECIPES)
    disc = bank.get("occupancy-radar")
    disc.calibrate_guided({
        "empty": [features_from_descriptors(w) for w in _windows(rng, 0.5, 20)],
        "occupied": [features_from_descriptors(w) for w in _windows(rng, 3, 20)],
    })
    bank.save("occupancy-radar")
    again = DiscriminatorBank(tmp_path)
    assert again.get("occupancy-radar").calibrated
    preds = again.predict_all(_radar_desc(3.0))
    assert [p.runtime_model_id for p in preds] == ["builtin:occupancy-radar"]
    again.reset("occupancy-radar")
    assert not DiscriminatorBank(tmp_path).get("occupancy-radar").calibrated
    with pytest.raises(KeyError):
        again.get("nope")
