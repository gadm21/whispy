"""Automatic annotation helpers for dataset engineering labs."""

from __future__ import annotations

from typing import Any, List, Optional


class Detection:
    """A single detection in a camera frame."""

    def __init__(self, label: str, confidence: float, box: Optional[list] = None):
        self.label = label
        self.confidence = confidence
        self.box = box

    @property
    def person(self) -> bool:
        return self.label == "person"


class PersonDetector:
    """Camera-based person detector used for automatic occupancy labels.

    The default implementation is a stub — plug in a real model
    (e.g. a torchvision detector or an on-device model) by subclassing
    and overriding ``detect``.
    """

    def __call__(self, jpeg_frames) -> List[Detection]:
        frames = jpeg_frames if isinstance(jpeg_frames, list) else [jpeg_frames]
        detections: List[Detection] = []
        for frame in frames:
            if frame:
                detections.extend(self.detect(frame))
        return detections

    def detect(self, jpeg: bytes) -> List[Detection]:
        """Override with a real detector. Returns detections for one frame."""
        return []


def auto_label_occupancy(minute, detector: Optional[PersonDetector] = None) -> bool:
    """Label a minute occupied/empty from camera detections.

    Sets ``minute.label(occupied=...)`` and returns the label.
    """
    detector = detector or PersonDetector()
    detections = detector(minute.camera())
    occupied = any(d.person for d in detections)
    minute.label(occupied=occupied)
    return occupied
