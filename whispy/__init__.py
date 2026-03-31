"""Whispy — WiFi Intelligence on ESP.

A Python toolkit for WiFi CSI sensing research with ESP32 microcontrollers.
"""

__version__ = "0.1.0"

from whispy.core import (
    CSI_SUBCARRIER_MASK,
    parse_csi_line,
    parse_csi_file,
    resample,
    select_subcarriers,
    rolling_variance,
    window_array,
)
from whispy.pipeline import Pipeline, Resample, RollingVariance, Window, Flatten

__all__ = [
    "CSI_SUBCARRIER_MASK",
    "parse_csi_line",
    "parse_csi_file",
    "resample",
    "select_subcarriers",
    "rolling_variance",
    "window_array",
    "Pipeline",
    "Resample",
    "RollingVariance",
    "Window",
    "Flatten",
]
