"""Dataset construction from captured minutes.

A Dataset is a selection of Minutes + labels + a split manifest +
a preprocessing definition — the unit Labs and experiments consume.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterator, List, Optional


class Dataset:
    """A labeled selection of minutes with a split manifest."""

    def __init__(self, name: str = "dataset"):
        self.name = name
        self.minutes: List[Any] = []
        self.manifest: Dict[str, Any] = {"splits": {}, "preprocessing": None}

    def add(self, minute) -> "Dataset":
        self.minutes.append(minute)
        return self

    def extend(self, minutes) -> "Dataset":
        self.minutes.extend(minutes)
        return self

    def split(self, train: float = 0.7, val: float = 0.15,
              seed: int = 0, by: str = "minute") -> "Dataset":
        """Assign train/val/test splits.

        ``by='minute'`` randomizes per minute. For leakage-safe splits
        use ``by='session'`` or ``by='subject'`` once those fields exist
        on the minute metadata — prevents temporal/subject leakage.
        """
        rng = random.Random(seed)
        keys = list(range(len(self.minutes)))
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * train)
        n_val = int(n * val)
        for rank, idx in enumerate(keys):
            split = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
            self.manifest["splits"][getattr(self.minutes[idx], "id", idx)] = split
        return self

    def split_manifest(self) -> Dict[str, list]:
        out: Dict[str, list] = {"train": [], "val": [], "test": []}
        for mid, split in self.manifest["splits"].items():
            out.setdefault(split, []).append(mid)
        return out

    def save_manifest(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"name": self.name, **self.manifest}, fh, indent=2)
        return path

    def __len__(self) -> int:
        return len(self.minutes)

    def to_xy(self, sensor: str = 'radar', label: str = 'label'):
        """One flattened fixed-size row per minute; reject ragged/missing labels.

        Use explicit preprocessing/windowing before this method if capture
        lengths differ. No implicit padding or label guessing is performed.
        """
        import numpy as np
        rows, labels = [], []
        for minute in self.minutes:
            if label not in minute.labels:
                raise KeyError(f'Minute {minute.id} has no label {label!r}')
            rows.append(minute[sensor].to_numpy().reshape(-1))
            labels.append(minute.labels[label])
        return (np.stack(rows) if rows else np.empty((0, 0)), np.asarray(labels))

    def __iter__(self) -> Iterator:
        return iter(self.minutes)
