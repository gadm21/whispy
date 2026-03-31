"""Load built-in datasets from HuggingFace (gadgadgad/* repos).

Each dataset is a collection of raw ESP32 CSI CSV files hosted on
HuggingFace Hub. This module downloads them, parses them through the
core pipeline, and returns ready-to-use (X, y) arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from whispy.core import parse_csi_file, resample, select_subcarriers, set_seed
from whispy.pipeline import Pipeline

# =========================================================================
# Registry of built-in datasets
# =========================================================================
@dataclass
class DatasetInfo:
    repo_id: str
    files: list[tuple[str, str, str]]  # (filename, label, split)
    task: str = ""
    environment: str = ""
    n_classes: int = 0


DATASETS: dict[str, DatasetInfo] = {
    "OfficeLocalization": DatasetInfo(
        repo_id="gadgadgad/OfficeLocalization",
        task="localization",
        environment="office",
        n_classes=4,
        files=[
            ("empty_1.csv", "empty", "train"),
            ("empty_2.csv", "empty", "test"),
            ("five_1.csv", "five", "train"),
            ("five_2.csv", "five", "test"),
            ("one_1.csv", "one", "train"),
            ("one_2.csv", "one", "test"),
            ("two_1.csv", "two", "train"),
            ("two_2.csv", "two", "test"),
        ],
    ),
    "OfficeHAR": DatasetInfo(
        repo_id="gadgadgad/OfficeHAR",
        task="har",
        environment="office",
        n_classes=4,
        files=[
            ("eat.csv", "eat", "percentage"),
            ("empty.csv", "empty", "percentage"),
            ("watch.csv", "watch", "percentage"),
            ("work.csv", "work", "percentage"),
        ],
    ),
    "HomeHAR": DatasetInfo(
        repo_id="gadgadgad/HomeHAR",
        task="har",
        environment="home",
        n_classes=7,
        files=[
            # train (session 1-2)
            ("drink_1.csv", "drink", "train"), ("drink_2.csv", "drink", "train"),
            ("eat_1.csv", "eat", "train"), ("eat_2.csv", "eat", "train"),
            ("smoke_1.csv", "smoke", "train"), ("smoke_2.csv", "smoke", "train"),
            ("watch_1.csv", "watch", "train"), ("watch_2.csv", "watch", "train"),
            ("work_1.csv", "work", "train"), ("work_2.csv", "work", "train"),
            ("sleep_1.csv", "sleep", "train"), ("sleep_2.csv", "sleep", "train"),
            ("empty_1.csv", "empty", "train"), ("empty_2.csv", "empty", "train"),
            # test (session 3)
            ("drink_3.csv", "drink", "test"), ("eat_3.csv", "eat", "test"),
            ("smoke_3.csv", "smoke", "test"), ("watch_3.csv", "watch", "test"),
            ("work_3.csv", "work", "test"), ("sleep_3.csv", "sleep", "test"),
            ("empty_3.csv", "empty", "test"),
        ],
    ),
    "HomeOccupation": DatasetInfo(
        repo_id="gadgadgad/HomeOccupation",
        task="occupation",
        environment="home",
        n_classes=3,
        files=[
            ("empty_1.csv", "empty", "train"), ("empty_2.csv", "empty", "test"),
            ("sleep_1.csv", "sleep", "train"), ("sleep_2.csv", "sleep", "test"),
            ("work_1.csv", "work", "train"), ("work_2.csv", "work", "train"),
            ("work_3.csv", "work", "test"),
        ],
    ),
}


def list_datasets() -> list[str]:
    """Return names of all built-in datasets."""
    return list(DATASETS.keys())


# =========================================================================
# Dataset container
# =========================================================================
@dataclass
class CSIDataset:
    """Container for a loaded CSI dataset split."""

    X: np.ndarray
    y: np.ndarray
    label_map: dict[str, int]
    name: str = ""
    split: str = ""

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim == 2 else 0

    @property
    def n_classes(self) -> int:
        return len(self.label_map)

    @property
    def labels(self) -> list[str]:
        return sorted(self.label_map, key=self.label_map.get)

    def balance(self, seed: int = 42) -> "CSIDataset":
        """Return a new dataset with classes balanced to the smallest class."""
        rng = np.random.RandomState(seed)
        classes, counts = np.unique(self.y, return_counts=True)
        min_n = counts.min()
        idx = []
        for c in classes:
            ci = np.where(self.y == c)[0]
            idx.append(rng.choice(ci, min_n, replace=False) if len(ci) > min_n else ci)
        idx = np.sort(np.concatenate(idx))
        return CSIDataset(
            X=self.X[idx], y=self.y[idx],
            label_map=self.label_map, name=self.name, split=self.split,
        )


# =========================================================================
# Download + load
# =========================================================================
def _download_files(info: DatasetInfo, cache_dir: str) -> list[tuple[str, str, str]]:
    """Download dataset CSVs from HuggingFace Hub.

    Returns list of (local_path, label, split).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required: pip install whispy[hf]"
        )

    os.makedirs(cache_dir, exist_ok=True)
    downloaded = []
    for fname, label, split in info.files:
        local = hf_hub_download(
            repo_id=info.repo_id,
            filename=fname,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        downloaded.append((local, label, split))
    return downloaded


def load(
    name: str,
    pipeline: Pipeline | None = None,
    guaranteed_sr: int = 150,
    window_len: int = 500,
    var_window: int = 20,
    cache_dir: str | None = None,
    balance_train: bool = True,
    train_pct: float = 0.8,
    verbose: bool = False,
) -> tuple[CSIDataset, CSIDataset]:
    """Load a built-in dataset, process it, and return (train, test).

    Parameters
    ----------
    name : one of list_datasets().
    pipeline : custom Pipeline. If None, uses RollingVariance(var_window)
        + Window(window_len) + Flatten().
    guaranteed_sr : resampling rate.
    window_len : window length for default pipeline.
    var_window : rolling variance window for default pipeline.
    cache_dir : HF cache directory. Default ~/.cache/whispy.
    balance_train : balance training classes.
    train_pct : fraction used as train for 'percentage' split files.
    verbose : print progress.
    """
    if name not in DATASETS:
        avail = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset '{name}'. Available: {avail}")

    info = DATASETS[name]
    cache_dir = cache_dir or os.path.join(Path.home(), ".cache", "whispy")

    if pipeline is None:
        from whispy.pipeline import RollingVariance, Window, Flatten
        pipeline = Pipeline([RollingVariance(var_window), Window(window_len), Flatten()])

    if verbose:
        print(f"[whispy] Loading '{name}' ({info.n_classes} classes, {info.task})")
        print(f"[whispy] Pipeline: {pipeline!r}")

    files = _download_files(info, cache_dir)
    label_map = {lbl: i for i, lbl in enumerate(sorted(set(l for _, l, _ in info.files)))}

    train_parts: list[tuple[np.ndarray, int]] = []
    test_parts: list[tuple[np.ndarray, int]] = []

    for fpath, label, split in files:
        if verbose:
            print(f"  {os.path.basename(fpath):25s}  label={label:8s}  split={split}")
        data = parse_csi_file(fpath)
        data = resample(data, target_sr=guaranteed_sr, keys=("mag",))
        data = select_subcarriers(data, keys=("mag",))
        mag = data["mag"]  # (T, 52)

        X = pipeline(mag)
        if X is None or len(X) == 0:
            if verbose:
                print(f"    SKIP — no windows")
            continue

        y_val = label_map[label]

        if split == "train":
            train_parts.append((X, y_val))
        elif split == "test":
            test_parts.append((X, y_val))
        elif split == "percentage":
            n_train = max(1, int(X.shape[0] * train_pct))
            train_parts.append((X[:n_train], y_val))
            if X.shape[0] > n_train:
                test_parts.append((X[n_train:], y_val))

    def _concat(parts: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
        if not parts:
            return np.empty((0,)), np.empty((0,), dtype=np.int64)
        Xs = [p[0] for p in parts]
        ys = [np.full(p[0].shape[0], p[1], dtype=np.int64) for p in parts]
        return np.concatenate(Xs), np.concatenate(ys)

    X_tr, y_tr = _concat(train_parts)
    X_te, y_te = _concat(test_parts)

    train_ds = CSIDataset(X=X_tr, y=y_tr, label_map=label_map, name=name, split="train")
    test_ds = CSIDataset(X=X_te, y=y_te, label_map=label_map, name=name, split="test")

    if balance_train and train_ds.n_samples > 0:
        train_ds = train_ds.balance()

    if verbose:
        print(f"[whispy] Train: {train_ds.X.shape}  Test: {test_ds.X.shape}  "
              f"Classes: {train_ds.n_classes} {train_ds.labels}")

    return train_ds, test_ds
