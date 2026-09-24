"""whisper-stt — installable speech-to-text model for Whispy.

Consumes microphone windows (``pcm_s16le`` payloads, any source device —
local mic, LAN Pi mic, recorded fixture) and emits a ``speech_to_text``
prediction whose ``attributes["text"]`` carries the transcript.

Backends (first configured/available wins):

- ``faster-whisper`` (CTranslate2 — CPU INT8, no torch needed)
- ``openai-whisper`` (torch)

Config::

    {
        "variant": "base.en",      # model size
        "device": "cpu",
        "compute_type": "int8",
        "backend": "faster",       # faster | openai
        "input": "audio",          # bound input name
        "language": "en",
    }

The Pi that captured the audio needs none of these dependencies — only
the execution node installs this package.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

from whispy.contracts import Prediction, SensorWindow
from whispy.models.runner import bound_samples
from whispy.processors.base import Processor, ProcessorMeta

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


class WhisperSttModel(Processor):
    """Speech-to-text over PCM audio windows via Whisper."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self._input = str(self._config.get("input") or "audio")
        self._variant = str(self._config.get("variant") or "base.en")
        self._device = str(self._config.get("device") or "cpu")
        self._compute = str(self._config.get("compute_type") or "int8")
        self._backend = str(self._config.get("backend") or "faster")
        self._language = self._config.get("language", "en")
        # Peak-normalize quiet captures (remote mics, AGC-less dongles)
        # toward ``target_peak`` before inference — tiny variants need
        # the signal well above the noise floor.
        self._normalize = bool(self._config.get("normalize_gain", True))
        self._target_peak = float(self._config.get("target_peak") or 0.5)
        self._model = None
        self._load_error = ""

    # -- setup ------------------------------------------------------------------
    def metadata(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="whisper-stt",
            version="0.1.0",
            processor_type="whisper-stt",
            sensor="microphone",
            task="speech_to_text",
            inputs=("microphone",),
            outputs=("text",),
            hardware_reqs={"cpu": True, "gpu": False,
                           "memory_mb": 500},
            config_schema={
                "type": "object",
                "properties": {
                    "variant": {"type": "string"},
                    "device": {"type": "string"},
                    "compute_type": {"type": "string"},
                    "backend": {"type": "string"},
                    "language": {"type": "string"},
                    "input": {"type": "string"},
                },
            },
        )

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._backend == "faster":
            try:
                from faster_whisper import WhisperModel  # type: ignore
                self._model = WhisperModel(
                    self._variant, device=self._device,
                    compute_type=self._compute)
                return True
            except Exception as exc:
                self._load_error = f"faster-whisper: {exc}"
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model(
                self._variant.replace(".en", ".en"),
                device=self._device)
            self._backend = "openai"
            return True
        except Exception as exc:
            self._load_error += f" | openai-whisper: {exc}"
            return False

    def health(self) -> Dict[str, Any]:
        if self._model is not None:
            return {"status": "ok", "backend": self._backend,
                    "variant": self._variant}
        try:
            import faster_whisper  # noqa: F401
            return {"status": "ok", "backend": "faster", "loaded": False}
        except ImportError:
            pass
        try:
            import whisper  # noqa: F401
            return {"status": "ok", "backend": "openai", "loaded": False}
        except ImportError:
            return {"status": "error",
                    "detail": "no whisper backend installed "
                              "(faster-whisper or openai-whisper)"}

    # -- audio assembly -------------------------------------------------------------
    def _pcm_bytes(self, samples) -> bytes:
        chunks: List[bytes] = []
        for s in samples:
            payload = s.payload
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, str):
                    try:
                        chunks.append(base64.b64decode(data))
                    except Exception:
                        continue
                elif isinstance(data, (bytes, bytearray)):
                    chunks.append(bytes(data))
            elif isinstance(payload, (bytes, bytearray)):
                chunks.append(bytes(payload))
            elif isinstance(payload, (list, tuple)) and np is not None:
                arr = np.asarray(payload)
                if arr.dtype != np.int16:
                    arr = (arr.astype(np.float32).clip(-1, 1)
                           * 32767).astype(np.int16)
                chunks.append(arr.tobytes())
        return b"".join(chunks)

    def _audio_float(self, pcm: bytes):
        if np is None:
            raise RuntimeError("numpy required for whisper-stt")
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # -- inference ------------------------------------------------------------------
    def predict(self, window: SensorWindow) -> Prediction:
        samples = bound_samples(window, self._input)
        if not samples:
            for sid, chunk in window.samples.items():
                if chunk and chunk[-1].sensor_type == "microphone":
                    samples = chunk
                    break
        if not samples:
            return Prediction(
                label="", confidence=0.0, task="speech_to_text",
                metadata={"error": "no microphone samples in window",
                          "text": ""})

        pcm = self._pcm_bytes(samples)
        if not pcm:
            return Prediction(
                label="", confidence=0.0, task="speech_to_text",
                metadata={"error": "empty audio payload", "text": ""})

        if not self._load():
            return Prediction(
                label="", confidence=0.0, task="speech_to_text",
                metadata={"error": f"whisper backend unavailable: "
                                   f"{self._load_error}", "text": ""})

        audio = self._audio_float(pcm)
        if self._normalize and np is not None and len(audio):
            peak = float(np.abs(audio).max())
            if 1e-5 < peak < self._target_peak:
                audio = audio * (self._target_peak / peak)
        duration = len(audio) / 16000.0
        try:
            if self._backend == "faster":
                segments, info = self._model.transcribe(
                    audio, language=self._language,
                    vad_filter=True, beam_size=5)
                text = " ".join(seg.text.strip()
                                for seg in segments).strip()
                confidence = float(getattr(info, "language_probability", 0.0)
                                   or 0.0)
            else:
                result = self._model.transcribe(
                    audio, language=self._language, fp16=False)
                text = str(result.get("text") or "").strip()
                confidence = 0.0
        except Exception as exc:
            return Prediction(
                label="", confidence=0.0, task="speech_to_text",
                metadata={"error": str(exc), "text": ""})

        return Prediction(
            label=text or "", confidence=confidence, task="speech_to_text",
            metadata={
                "text": text,
                "duration_s": round(duration, 3),
                "variant": self._variant,
                "backend": self._backend,
                "audio_sensor": samples[-1].sensor_id,
                "audio_device": samples[-1].device_id,
            })


__all__ = ["WhisperSttModel"]
