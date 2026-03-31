"""Composable processing pipeline for CSI feature engineering.

Usage:
    pipe = Pipeline([
        Resample(in_sr=200, out_sr=150),
        RollingVariance(window=20),
        Window(length=500),
        Flatten(),
    ])
    X = pipe(raw_mag)  # (N, 52) → (n_windows, 500*52)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from whispy.core import resample, rolling_variance, window_array


# =========================================================================
# Base
# =========================================================================
class Step(ABC):
    """A single processing step in a pipeline."""

    @abstractmethod
    def __call__(self, arr: np.ndarray, **ctx: Any) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


# =========================================================================
# Concrete steps
# =========================================================================
class Resample(Step):
    """Resample along time axis (expects dict context with 'timestamp')."""

    def __init__(self, out_sr: int = 150):
        self.out_sr = out_sr

    def __call__(self, arr: np.ndarray, **ctx: Any) -> np.ndarray:
        ts = ctx.get("timestamp")
        if ts is None:
            return arr
        data = {"mag": arr, "timestamp": ts}
        result = resample(data, target_sr=self.out_sr, keys=("mag",))
        ctx["timestamp"] = result["timestamp"]
        if "resampling_meta" in result:
            ctx["resampling_meta"] = result["resampling_meta"]
        return result["mag"]


class RollingVariance(Step):
    """Apply rolling variance per subcarrier."""

    def __init__(self, window: int = 20):
        self.window = window

    def __call__(self, arr: np.ndarray, **ctx: Any) -> np.ndarray:
        return rolling_variance(arr, self.window)


class Window(Step):
    """Slice into fixed-length windows."""

    def __init__(self, length: int = 500, stride: int | None = None):
        self.length = length
        self.stride = stride or length

    def __call__(self, arr: np.ndarray, **ctx: Any) -> np.ndarray:
        result = window_array(arr, self.length, self.stride, flatten=False)
        if result is None:
            raise ValueError(
                f"Too few samples ({arr.shape[0]}) for window length {self.length}"
            )
        return result


class Flatten(Step):
    """Flatten 3-D windows (n, L, C) → (n, L*C)."""

    def __call__(self, arr: np.ndarray, **ctx: Any) -> np.ndarray:
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], -1)
        return arr


# =========================================================================
# Pipeline
# =========================================================================
class Pipeline:
    """Chain of processing steps applied sequentially.

    Parameters
    ----------
    steps : list of Step objects.

    Usage
    -----
    >>> pipe = Pipeline([RollingVariance(20), Window(500), Flatten()])
    >>> X = pipe(mag_array)
    """

    def __init__(self, steps: list[Step]):
        self.steps = steps

    def __call__(self, arr: np.ndarray, **ctx: Any) -> np.ndarray:
        for step in self.steps:
            arr = step(arr, **ctx)
        return arr

    def __repr__(self) -> str:
        lines = [f"  {i}: {s!r}" for i, s in enumerate(self.steps)]
        return "Pipeline([\n" + "\n".join(lines) + "\n])"

    @staticmethod
    def from_name(name: str) -> "Pipeline":
        """Create a pipeline from a short name.

        Supported names:
            'amplitude'  — raw amplitude, window, flatten
            'rv20'       — rolling variance W=20, window, flatten
            'rv200'      — rolling variance W=200, window, flatten
            'rv2000'     — rolling variance W=2000, window, flatten
        """
        presets: dict[str, list[Step]] = {
            "amplitude": [Window(500), Flatten()],
            "rv20": [RollingVariance(20), Window(500), Flatten()],
            "rv200": [RollingVariance(200), Window(500), Flatten()],
            "rv2000": [RollingVariance(2000), Window(500), Flatten()],
        }
        if name not in presets:
            raise ValueError(f"Unknown pipeline '{name}'. Choose from: {list(presets)}")
        return Pipeline(presets[name])
