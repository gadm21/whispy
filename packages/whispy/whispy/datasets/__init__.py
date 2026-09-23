"""Local dataset helpers — recorded captures as training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..contracts import SensorSample


class Dataset:
    """A local collection of labeled SensorSample sequences.

    Stored as a directory of ``*.jsonl`` files — one sample per line —
    plus an optional ``labels.json`` mapping.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def add(self, sample: SensorSample, split: str = "train") -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / f"{split}.jsonl").open("a") as fh:
            fh.write(json.dumps(sample.to_dict()) + "\n")

    def samples(self, split: str = "train") -> Iterator[SensorSample]:
        file = self.path / f"{split}.jsonl"
        if not file.exists():
            return
        for line in file.read_text().splitlines():
            if line.strip():
                yield SensorSample.from_dict(json.loads(line))

    def __len__(self) -> int:
        return sum(1 for _ in self.samples())


__all__ = ["Dataset"]
