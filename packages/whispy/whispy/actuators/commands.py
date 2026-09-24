"""Typed actuator commands — ergonomic constructors over ActuatorCommand.

    speaker.execute(Speak("hello"))
    speaker.execute(SetVolume(0.7))
    matrix.execute(ShowPattern([[1,0,1],[0,1,0],[1,0,1]]))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..contracts import ActuatorCommand


class Speak(ActuatorCommand):
    """Speak ``text`` through a speaker actuator (TTS)."""

    def __init__(self, text: str, *, rate: Optional[int] = None,
                 volume: Optional[float] = None, **kw: Any):
        params: Dict[str, Any] = {"text": text}
        if rate is not None:
            params["rate"] = rate
        if volume is not None:
            params["volume"] = volume
        super().__init__(operation="speak", params=params, **kw)


class Play(ActuatorCommand):
    """Play raw/encoded audio through a speaker actuator."""

    def __init__(self, audio: Any, *, encoding: str = "pcm_s16le",
                 sample_rate: int = 16000, **kw: Any):
        super().__init__(
            operation="play",
            params={"audio": audio, "encoding": encoding,
                    "sample_rate": sample_rate}, **kw)


class Stop(ActuatorCommand):
    """Stop any in-flight output on the actuator."""

    def __init__(self, **kw: Any):
        super().__init__(operation="stop", params={}, **kw)


class SetVolume(ActuatorCommand):
    """Set output volume (0.0–1.0)."""

    def __init__(self, level: float, **kw: Any):
        super().__init__(operation="set_volume",
                         params={"level": float(level)}, **kw)


class ShowPattern(ActuatorCommand):
    """Display a pixel pattern on a matrix actuator.

    ``pixels`` is an 8×8 (or device-sized) array of ``(r,g,b)`` tuples or
    0/1 cells for monochrome matrices.
    """

    def __init__(self, pixels: List[Any], *, duration_s: float = 0.0,
                 **kw: Any):
        super().__init__(operation="show",
                         params={"pixels": pixels,
                                 "duration_s": duration_s}, **kw)


class ShowMessage(ActuatorCommand):
    """Scroll a text message on a matrix actuator."""

    def __init__(self, text: str, *, speed: float = 0.1,
                 color: Optional[List[int]] = None, **kw: Any):
        params: Dict[str, Any] = {"text": text, "speed": speed}
        if color is not None:
            params["color"] = color
        super().__init__(operation="show_message", params=params, **kw)


class Clear(ActuatorCommand):
    """Clear/blank a matrix or display actuator."""

    def __init__(self, **kw: Any):
        super().__init__(operation="clear", params={}, **kw)


__all__ = [
    "Speak", "Play", "Stop", "SetVolume",
    "ShowPattern", "ShowMessage", "Clear",
]
