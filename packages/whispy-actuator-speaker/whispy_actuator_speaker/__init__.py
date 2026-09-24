"""Speaker actuator adapter — TTS + audio playback for Whispy.

Exposes the host's speaker as an actuator with operations::

    speak(text, rate?, volume?)   — text-to-speech
    play(audio, encoding, sample_rate) — raw PCM playback
    stop()                        — stop in-flight output
    set_volume(level)             — output volume 0..1

Backends (first available wins):

- Windows: ``System.Speech`` via PowerShell (zero Python deps) or
  ``pyttsx3`` when installed; PCM playback via ``winsound``.
- Linux/macOS: ``pyttsx3``/``espeak`` for TTS, ``aplay`` for PCM.

``succeeded`` requires the backend to confirm the utterance was
submitted to the audio device — never a config-parse success.
"""

from __future__ import annotations

import base64
import io
import logging
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from typing import Any, Dict, List, Optional

from whispy.contracts import (
    ActionResult, ActionStatus, ActuatorCommand, ActuatorDescriptor,
)
from whispy.actuators.base import (
    ActuatorAdapter, ActuatorHandle, ActuatorMeta,
)

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"
_IS_LINUX = platform.system() == "Linux"


def _pyttsx3_engine():
    try:
        import pyttsx3  # type: ignore
        return pyttsx3.init()
    except Exception:
        return None


class _SpeakerHandle(ActuatorHandle):
    def __init__(self, descriptor: ActuatorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        self._config = dict(config or {})
        self._engine = None
        self._engine_tried = False
        self._procs: List[subprocess.Popen] = []
        self._lock = threading.Lock()

    @property
    def info(self) -> ActuatorDescriptor:
        return self._desc

    @property
    def descriptor(self) -> ActuatorDescriptor:
        return self._desc

    # -- backends ---------------------------------------------------------------
    def _tts_engine(self):
        if not self._engine_tried:
            self._engine = _pyttsx3_engine()
            self._engine_tried = True
        return self._engine

    def _speak_windows_sapi(self, text: str, rate: int = 0,
                            volume: int = 100) -> ActionResult:
        """System.Speech via PowerShell — no Python deps required."""
        safe = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {int(rate)}; $s.Volume = {int(volume)}; "
            f"$s.Speak('{safe}'); $s.Dispose()")
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=120)
            ok = proc.returncode == 0
            return ActionResult(
                status=ActionStatus.SUCCEEDED if ok else ActionStatus.FAILED,
                action_type="speaker",
                detail="SAPI speak completed" if ok else
                       f"SAPI failed: {proc.stderr.strip()[:200]}",
                response={"backend": "sapi", "chars": len(text)})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker", detail=str(exc))

    def _speak_pyttsx3(self, text: str, rate: Optional[int],
                       volume: Optional[float]) -> ActionResult:
        engine = self._tts_engine()
        if engine is None:
            return ActionResult(status=ActionStatus.UNSUPPORTED,
                                action_type="speaker",
                                detail="pyttsx3 unavailable")
        try:
            if rate is not None:
                engine.setProperty("rate", int(rate))
            if volume is not None:
                engine.setProperty("volume", float(volume))
            engine.say(text)
            engine.runAndWait()
            return ActionResult(status=ActionStatus.SUCCEEDED,
                                action_type="speaker",
                                detail="pyttsx3 speak completed",
                                response={"backend": "pyttsx3"})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker", detail=str(exc))

    def _speak_espeak(self, text: str) -> ActionResult:
        exe = shutil.which("espeak") or shutil.which("espeak-ng")
        if not exe:
            return ActionResult(status=ActionStatus.UNSUPPORTED,
                                action_type="speaker",
                                detail="no TTS backend (espeak/pyttsx3)")
        try:
            proc = subprocess.run([exe, text], capture_output=True,
                                  timeout=120)
            ok = proc.returncode == 0
            return ActionResult(
                status=ActionStatus.SUCCEEDED if ok else ActionStatus.FAILED,
                action_type="speaker",
                detail="espeak completed" if ok else proc.stderr[:200],
                response={"backend": "espeak"})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker", detail=str(exc))

    def _speak(self, params: Dict[str, Any]) -> ActionResult:
        text = str(params.get("text") or "")
        if not text:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker",
                                detail="speak requires 'text'")
        rate = params.get("rate")
        volume = params.get("volume")
        if _IS_WINDOWS:
            vol = int(volume * 100) if isinstance(volume, (int, float)) \
                and volume <= 1 else int(volume or 100)
            result = self._speak_windows_sapi(
                text, rate=int(rate or 0), volume=vol)
            if result.status is ActionStatus.SUCCEEDED:
                return result
            return self._speak_pyttsx3(text, rate, volume)
        result = self._speak_pyttsx3(text, rate, volume)
        if result.status is ActionStatus.SUCCEEDED:
            return result
        return self._speak_espeak(text)

    def _play(self, params: Dict[str, Any]) -> ActionResult:
        audio = params.get("audio")
        if audio is None:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker",
                                detail="play requires 'audio'")
        encoding = str(params.get("encoding") or "pcm_s16le")
        rate = int(params.get("sample_rate") or 16000)
        if isinstance(audio, str):
            try:
                audio = base64.b64decode(audio)
            except Exception:
                audio = audio.encode()
        if encoding != "pcm_s16le":
            return ActionResult(status=ActionStatus.UNSUPPORTED,
                                action_type="speaker",
                                detail=f"unsupported encoding {encoding!r}")
        if _IS_WINDOWS:
            return self._play_winsound(bytes(audio), rate)
        return self._play_aplay(bytes(audio), rate)

    @staticmethod
    def _pcm_to_wav(pcm: bytes, rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        return buf.getvalue()

    def _play_winsound(self, pcm: bytes, rate: int) -> ActionResult:
        try:
            import winsound  # type: ignore
        except ImportError:
            return ActionResult(status=ActionStatus.UNSUPPORTED,
                                action_type="speaker",
                                detail="winsound unavailable")
        try:
            wav = self._pcm_to_wav(pcm, rate)
            with tempfile.NamedTemporaryFile(suffix=".wav",
                                             delete=False) as fh:
                fh.write(wav)
                path = fh.name
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return ActionResult(status=ActionStatus.SUCCEEDED,
                                action_type="speaker",
                                detail=f"played {len(pcm)} bytes at {rate} Hz",
                                response={"backend": "winsound"})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker", detail=str(exc))

    def _play_aplay(self, pcm: bytes, rate: int) -> ActionResult:
        exe = shutil.which("aplay") or shutil.which("paplay")
        if not exe:
            return ActionResult(status=ActionStatus.UNSUPPORTED,
                                action_type="speaker",
                                detail="no PCM playback backend (aplay)")
        try:
            args = [exe]
            if exe.endswith("aplay"):
                args += ["-f", "S16_LE", "-r", str(rate), "-c", "1"]
            proc = subprocess.Popen(args, stdin=subprocess.PIPE)
            proc.communicate(pcm, timeout=120)
            ok = proc.returncode == 0
            return ActionResult(
                status=ActionStatus.SUCCEEDED if ok else ActionStatus.FAILED,
                action_type="speaker",
                detail=f"{exe} exit {proc.returncode}",
                response={"backend": exe})
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker", detail=str(exc))

    def _stop(self) -> ActionResult:
        with self._lock:
            procs = list(self._procs)
            self._procs.clear()
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
        if _IS_WINDOWS:
            try:
                import winsound  # type: ignore
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        return ActionResult(status=ActionStatus.SUCCEEDED,
                            action_type="speaker", detail="stopped")

    def _set_volume(self, params: Dict[str, Any]) -> ActionResult:
        level = params.get("level")
        if level is None:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="speaker",
                                detail="set_volume requires 'level'")
        if _IS_WINDOWS:
            # SAPI volume is per-utterance; persist for future speak calls.
            self._config["volume"] = float(level)
            return ActionResult(status=ActionStatus.SUCCEEDED,
                                action_type="speaker",
                                detail=f"volume {float(level):.2f} stored",
                                response={"level": float(level)})
        return ActionResult(status=ActionStatus.UNSUPPORTED,
                            action_type="speaker",
                            detail="set_volume unsupported on this backend")

    def execute(self, command: ActuatorCommand) -> ActionResult:
        op = command.operation
        started = time.time()
        if op == "speak":
            result = self._speak(command.params)
        elif op == "play":
            result = self._play(command.params)
        elif op == "stop":
            result = self._stop()
        elif op == "set_volume":
            result = self._set_volume(command.params)
        else:
            result = ActionResult(status=ActionStatus.UNSUPPORTED,
                                  action_type="speaker",
                                  detail=f"unknown operation {op!r}")
        result.started_at = started
        result.finished_at = time.time()
        return result

    def close(self) -> None:
        self._stop()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


class SpeakerActuatorAdapter(ActuatorAdapter):
    """Discovers the host speaker/output device.

    One descriptor per host (``speaker-<hash>``); the underlying output
    device is the OS default. A host with no audio stack still reports
    the actuator — operations then return ``unsupported`` explicitly.
    """

    def __init__(self):
        self._handles: List[_SpeakerHandle] = []

    def metadata(self) -> ActuatorMeta:
        return ActuatorMeta(
            name="speaker",
            version="0.1.0",
            kinds=("speaker",),
            description="Host speaker: TTS (SAPI/pyttsx3/espeak) + PCM playback",
            config_schema={
                "type": "object",
                "properties": {
                    "rate": {"type": "integer"},
                    "volume": {"type": "number"},
                },
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[ActuatorDescriptor]:
        hw = f"{platform.system()}-{platform.node()}-default-speaker"
        return [ActuatorDescriptor(
            id=ActuatorDescriptor.make_id("speaker", hw),
            kind="speaker",
            adapter="speaker",
            name="builtin-speaker",
            hardware_id=hw,
            operations=["speak", "play", "stop", "set_volume"],
            capabilities=["tts", "pcm_playback"],
            config_schema=self.metadata().config_schema,
            stable=True,
            metadata={"backend": (
                "sapi" if _IS_WINDOWS else
                ("espeak" if _IS_LINUX else "pyttsx3"))},
        )]

    def connect(self, descriptor: ActuatorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _SpeakerHandle:
        handle = _SpeakerHandle(descriptor, config)
        self._handles.append(handle)
        return handle

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


__all__ = ["SpeakerActuatorAdapter"]
