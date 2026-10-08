"""Built-in local discriminators over physical descriptors.

A discriminator answers one binary question (occupied?, present?,
moving?, noisy?) from the descriptor dicts produced by
:func:`whispy.descriptors.window_descriptors`. They are deliberately
small — Gaussian class models over a handful of named physical
features — so they run on any node, calibrate in a minute and explain
themselves (per-feature z-distances).

Feature names are ``<sensor_type>.<field>.<stat>`` (e.g.
``radar.snr_db.std``) or ``<sensor_type>.emitters`` for radio scans;
sensors of the same type are averaged.

Calibration (tailored per data type via :data:`RECIPES`):

* **guided** — :class:`CalibrationSession` walks the user through the
  recipe's steps ("leave the room empty", "walk around"…), recording
  labelled descriptor windows, then fits class means/stds.
* **auto** — unlabelled history; a 1-D 2-means split on the recipe's
  primary feature assigns pseudo-labels (the paper's ``acc_km``
  centroid-midpoint idea), then the same fit runs.

An uncalibrated discriminator never guesses: ``predict`` returns
``None``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .contracts import Prediction

MIN_STD = 1e-3


# ---------------------------------------------------------------------------
# Recipes — one per data type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationStep:
    label: str
    instruction: str
    seconds: float = 60.0


@dataclass(frozen=True)
class Recipe:
    name: str
    task: str
    data_type: str
    labels: tuple                       # (negative, positive)
    features: tuple
    primary: str                        # feature used for auto split
    positive_high: bool                 # positive class has higher primary
    steps: tuple
    auto_min_windows: int = 60

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


RECIPES: Dict[str, Recipe] = {r.name: r for r in (
    Recipe(
        name="occupancy-radar", task="occupancy", data_type="radar",
        labels=("empty", "occupied"),
        features=("radar.snr_db.std", "radar.snr_db.mean",
                  "radar.range_profile_mean.std"),
        primary="radar.snr_db.std", positive_high=True,
        steps=(CalibrationStep("empty", "Leave the room empty — nobody in "
                               "the radar's field of view.", 60),
               CalibrationStep("occupied", "Stay in the room: sit, then "
                               "move around normally.", 60))),
    Recipe(
        name="occupancy-csi", task="occupancy", data_type="csi",
        labels=("empty", "occupied"),
        features=("csi.value.std", "csi.amplitude_mean.std", "csi.rssi.std"),
        primary="csi.value.std", positive_high=True,
        steps=(CalibrationStep("empty", "Leave the room empty between the "
                               "Wi-Fi transmitter and receiver.", 60),
               CalibrationStep("occupied", "Walk and sit between the "
                               "transmitter and receiver.", 60))),
    Recipe(
        name="presence-ble", task="presence", data_type="ble_scan",
        labels=("away", "present"),
        features=("ble_scan.emitters", "radio.emitters"),
        primary="ble_scan.emitters", positive_high=True,
        steps=(CalibrationStep("away", "Take your phone/watch out of "
                               "range (another room or outside).", 45),
               CalibrationStep("present", "Bring your phone/watch next to "
                               "the node.", 45)),
        auto_min_windows=120),
    Recipe(
        name="activity-imu", task="activity", data_type="imu",
        labels=("still", "moving"),
        features=("imu.magnitude.std", "imu.value.std", "imu.magnitude.max"),
        primary="imu.magnitude.std", positive_high=True,
        steps=(CalibrationStep("still", "Keep the device/watch still.", 30),
               CalibrationStep("moving", "Walk around wearing or holding "
                               "the device.", 30))),
    Recipe(
        name="noise-mic", task="noise", data_type="microphone",
        labels=("quiet", "noisy"),
        features=("microphone.value.std", "microphone.value.max"),
        primary="microphone.value.std", positive_high=True,
        steps=(CalibrationStep("quiet", "Keep the room quiet.", 30),
               CalibrationStep("noisy", "Talk or play audio at normal "
                               "volume.", 30))),
)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def features_from_descriptors(descriptors: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten window descriptors into ``type.field.stat`` features,
    averaging sensors of the same type."""
    acc: Dict[str, List[float]] = {}
    for desc in (descriptors or {}).values():
        if not isinstance(desc, Mapping) or not desc.get("type"):
            continue
        t = str(desc["type"])
        for fname, stats in (desc.get("fields") or {}).items():
            if isinstance(stats, Mapping):
                for stat, v in stats.items():
                    if isinstance(v, (int, float)) and stat != "n":
                        acc.setdefault(f"{t}.{fname}.{stat}", []).append(float(v))
        scan = desc.get("scan")
        if isinstance(scan, Mapping) and isinstance(scan.get("emitters"), (int, float)):
            acc.setdefault(f"{t}.emitters", []).append(float(scan["emitters"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Calibration + discriminator
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    method: str                                       # guided | auto
    classes: Dict[str, Dict[str, Dict[str, float]]]   # label → feat → {mean,std,n}
    prior: Dict[str, float]
    created_at: float = field(default_factory=time.time)
    windows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Calibration":
        return cls(method=str(d.get("method", "guided")),
                   classes=dict(d.get("classes") or {}),
                   prior=dict(d.get("prior") or {}),
                   created_at=float(d.get("created_at") or time.time()),
                   windows=int(d.get("windows") or 0))


def _fit(recipe: Recipe, labelled: Mapping[str, Sequence[Mapping[str, float]]],
         method: str) -> Calibration:
    classes: Dict[str, Dict[str, Dict[str, float]]] = {}
    total = sum(len(v) for v in labelled.values())
    for label in recipe.labels:
        rows = list(labelled.get(label) or [])
        if not rows:
            raise ValueError(f"no windows recorded for {label!r}")
        stats: Dict[str, Dict[str, float]] = {}
        for feat in recipe.features:
            xs = [r[feat] for r in rows if feat in r]
            if not xs:
                continue
            mean = sum(xs) / len(xs)
            std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
            stats[feat] = {"mean": mean, "std": max(std, MIN_STD,
                                                    abs(mean) * 0.02),
                           "n": len(xs)}
        if not stats:
            raise ValueError(f"no recipe features present for {label!r} "
                             f"(need any of {list(recipe.features)})")
        classes[label] = stats
    shared = set.intersection(*(set(s) for s in classes.values()))
    if not shared:
        raise ValueError("classes share no features — check sensors")
    for label in classes:
        classes[label] = {f: classes[label][f] for f in shared}
    prior = {label: max(len(labelled.get(label) or []), 1) / max(total, 1)
             for label in recipe.labels}
    return Calibration(method=method, classes=classes, prior=prior,
                       windows=total)


def _two_means(xs: Sequence[float], iters: int = 50) -> float:
    """1-D 2-means — returns the centroid midpoint threshold."""
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        raise ValueError("feature is constant — cannot auto-calibrate")
    c0, c1 = lo, hi
    for _ in range(iters):
        mid = (c0 + c1) / 2
        a = [x for x in xs if x <= mid]
        b = [x for x in xs if x > mid]
        if not a or not b:
            break
        n0, n1 = sum(a) / len(a), sum(b) / len(b)
        if abs(n0 - c0) < 1e-12 and abs(n1 - c1) < 1e-12:
            break
        c0, c1 = n0, n1
    return (c0 + c1) / 2


class Discriminator:
    """Gaussian discriminator over a recipe's physical features."""

    def __init__(self, recipe: Recipe, calibration: Optional[Calibration] = None):
        self.recipe = recipe
        self.calibration = calibration

    @property
    def name(self) -> str:
        return self.recipe.name

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    # -- calibration -------------------------------------------------------
    def calibrate_guided(self, labelled: Mapping[str, Sequence[Mapping[str, float]]]
                         ) -> Calibration:
        self.calibration = _fit(self.recipe, labelled, "guided")
        return self.calibration

    def calibrate_auto(self, history: Sequence[Mapping[str, float]]) -> Calibration:
        rows = [r for r in history if self.recipe.primary in r]
        if len(rows) < self.recipe.auto_min_windows:
            raise ValueError(f"need >= {self.recipe.auto_min_windows} windows "
                             f"with {self.recipe.primary} (have {len(rows)})")
        thr = _two_means([r[self.recipe.primary] for r in rows])
        neg, pos = self.recipe.labels
        hi, lo = (pos, neg) if self.recipe.positive_high else (neg, pos)
        labelled = {hi: [r for r in rows if r[self.recipe.primary] > thr],
                    lo: [r for r in rows if r[self.recipe.primary] <= thr]}
        self.calibration = _fit(self.recipe, labelled, "auto")
        return self.calibration

    # -- inference ---------------------------------------------------------
    def score(self, features: Mapping[str, float]) -> Optional[Dict[str, Any]]:
        if not self.calibration:
            return None
        loglik: Dict[str, float] = {}
        z: Dict[str, Dict[str, float]] = {}
        for label, stats in self.calibration.classes.items():
            ll = math.log(max(self.calibration.prior.get(label, 0.5), 1e-6))
            used = 0
            for feat, s in stats.items():
                if feat not in features:
                    continue
                d = (features[feat] - s["mean"]) / s["std"]
                ll += -0.5 * d * d - math.log(s["std"])
                z.setdefault(label, {})[feat] = round(d, 3)
                used += 1
            if used == 0:
                return None
            loglik[label] = ll
        m = max(loglik.values())
        exp = {k: math.exp(v - m) for k, v in loglik.items()}
        tot = sum(exp.values())
        probs = {k: v / tot for k, v in exp.items()}
        return {"probs": probs, "z": z}

    def predict(self, descriptors: Mapping[str, Any], *,
                device_id: str = "") -> Optional[Prediction]:
        feats = features_from_descriptors(descriptors)
        out = self.score(feats)
        if out is None:
            return None
        label = max(out["probs"], key=out["probs"].get)
        return Prediction(
            label=label, confidence=round(out["probs"][label], 4),
            device_id=device_id, runtime_model_id=f"builtin:{self.name}",
            task=self.recipe.task,
            scores={k: round(v, 4) for k, v in out["probs"].items()},
            metadata={"discriminator": self.name,
                      "calibration": self.calibration.method,
                      "z": out["z"]})

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "task": self.recipe.task,
                "data_type": self.recipe.data_type,
                "labels": list(self.recipe.labels),
                "calibrated": self.calibrated,
                "calibration": (self.calibration.to_dict()
                                if self.calibration else None),
                "recipe": self.recipe.to_dict()}


class CalibrationSession:
    """Guided calibration — walk the recipe steps, record windows."""

    def __init__(self, discriminator: Discriminator):
        self.discriminator = discriminator
        self.recipe = discriminator.recipe
        self.step_index = 0
        self.recorded: Dict[str, List[Dict[str, float]]] = {
            label: [] for label in self.recipe.labels}
        self.started_at = time.time()

    @property
    def current(self) -> Optional[CalibrationStep]:
        steps = self.recipe.steps
        return steps[self.step_index] if self.step_index < len(steps) else None

    def record(self, descriptors: Mapping[str, Any]) -> int:
        step = self.current
        if step is None:
            return 0
        feats = features_from_descriptors(descriptors)
        if any(f in feats for f in self.recipe.features):
            self.recorded[step.label].append(feats)
        return len(self.recorded[step.label])

    def next_step(self) -> Optional[CalibrationStep]:
        self.step_index += 1
        return self.current

    def finish(self) -> Calibration:
        return self.discriminator.calibrate_guided(self.recorded)

    def status(self) -> Dict[str, Any]:
        step = self.current
        return {"discriminator": self.recipe.name,
                "step": self.step_index, "steps": len(self.recipe.steps),
                "current": asdict(step) if step else None,
                "recorded": {k: len(v) for k, v in self.recorded.items()},
                "started_at": self.started_at}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def calibration_dir() -> Path:
    root = Path(os.getenv("WHISPY_HOME", "~/.whispy")).expanduser()
    path = root / "calibration"
    path.mkdir(parents=True, exist_ok=True)
    return path


class DiscriminatorBank:
    """All recipes as discriminators + persisted calibrations."""

    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory) if directory else calibration_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.items: Dict[str, Discriminator] = {}
        for name, recipe in RECIPES.items():
            self.items[name] = Discriminator(recipe, self._load(name))

    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def _load(self, name: str) -> Optional[Calibration]:
        try:
            return Calibration.from_dict(json.loads(self._path(name).read_text()))
        except (OSError, ValueError):
            return None

    def save(self, name: str) -> None:
        cal = self.items[name].calibration
        if cal is None:
            self._path(name).unlink(missing_ok=True)
            return
        self._path(name).write_text(json.dumps(cal.to_dict(), indent=1))

    def reset(self, name: str) -> None:
        self.items[name].calibration = None
        self.save(name)

    def get(self, name: str) -> Discriminator:
        if name not in self.items:
            raise KeyError(f"unknown discriminator {name!r}; "
                           f"known: {sorted(self.items)}")
        return self.items[name]

    def predict_all(self, descriptors: Mapping[str, Any], *,
                    device_id: str = "",
                    enabled: Optional[Iterable[str]] = None) -> List[Prediction]:
        names = set(enabled) if enabled is not None else set(self.items)
        out = []
        for name, disc in self.items.items():
            if name in names:
                pred = disc.predict(descriptors, device_id=device_id)
                if pred is not None:
                    out.append(pred)
        return out

    def to_list(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self.items.values()]


__all__ = ["RECIPES", "Recipe", "CalibrationStep", "Calibration",
           "Discriminator", "CalibrationSession", "DiscriminatorBank",
           "features_from_descriptors"]
