"""Microphone sensor adapter — normalized PCM audio for Whispy.

Enumerates real capture devices via ``sounddevice`` (PortAudio: WASAPI
on Windows, ALSA on Linux/Pi, CoreAudio on macOS) and exports each input
device as a :class:`SensorDescriptor` with a stable id derived from the
host-API + device name.

Every stream is normalized to the Whispy audio contract::

    16 kHz · mono · signed 16-bit PCM · 100 ms samples

    payload = {"encoding": "pcm_s16le", "sample_rate": 16000,
               "channels": 1, "data": "<base64 pcm>"}

The adapter knows nothing about STT — the same stream can feed Whisper,
a VAD, a recorder, or a LAN client simultaneously.
"""

from __future__ import annotations

import base64
import itertools
import logging
import queue
import time
from typing import Any, Dict, Iterator, List, Optional

from whispy.contracts import SensorDescriptor, SensorSample
from whispy.devices.base import SensorHandle
from whispy.sensors.base import HealthReport, SensorAdapter, SensorMeta

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - dependency of this package
    sd = None  # type: ignore

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 100


class _MicrophoneHandle(SensorHandle):
    """Streams 100 ms PCM chunks from one input device."""

    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        self._config = dict(config or {})
        self._seq = itertools.count()
        self._stream = None
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _open(self) -> None:
        if self._stream is not None:
            return
        rate = int(self._config.get("sample_rate") or SAMPLE_RATE)
        device = self._desc.metadata.get("device_index")
        # ALSA (unlike WASAPI) does not resample: a USB mic that only
        # supports 44.1/48 kHz rejects a 16 kHz request with
        # paInvalidSampleRate. Capture at a rate the device accepts and
        # resample to the contract rate in software.
        native = rate
        try:
            sd.check_input_settings(device=device, samplerate=rate,
                                    channels=CHANNELS)
        except Exception:
            native = int(float(
                self._desc.metadata.get("default_sample_rate") or 44100))
            logger.info("mic %s: %d Hz unsupported; capturing at %d Hz "
                        "and resampling", self._desc.id, rate, native)
        self._native_rate = native
        block = int(native * CHUNK_MS / 1000)
        q = self._queue

        def _cb(indata, frames, _time, status):
            chunk = indata[:, 0].copy() if np is not None else bytes(indata)
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass                     # slow consumer → drop, never block

        self._stream = sd.InputStream(
            device=device, samplerate=native, channels=CHANNELS,
            dtype="float32" if np is not None else "int16",
            blocksize=block, callback=_cb)
        self._stream.start()

    def _to_pcm(self, chunk, rate: int) -> bytes:
        """float32 native-rate chunk → int16 PCM at the contract rate.

        Resampling is stateful: the fractional input position and the
        unconsumed tail of the previous chunk carry over, so the output
        stream is continuous — naive per-chunk interpolation clicks every
        100 ms and garbles speech.
        """
        if np is None:
            return bytes(chunk)
        native = getattr(self, "_native_rate", rate)
        if native != rate and len(chunk) > 0:
            carry = getattr(self, "_rs_carry", None)
            pos = getattr(self, "_rs_pos", 0.0)
            buf = np.concatenate([carry, chunk]) if carry is not None \
                and len(carry) else chunk
            step = native / rate
            n_out = int((len(buf) - 1 - pos) / step)
            if n_out > 0:
                idx = pos + np.arange(n_out) * step
                i0 = idx.astype(np.int64)
                frac = (idx - i0).astype(np.float32)
                out = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac
                used = int(idx[-1]) + 1
                self._rs_pos = idx[-1] + step - used
                self._rs_carry = buf[used:]
                chunk = out.astype(np.float32)
            else:
                self._rs_pos = pos
                self._rs_carry = buf
                return b""
        return (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    def stream(self, max_samples: Optional[int] = None) -> Iterator[SensorSample]:
        self._open()
        rate = int(self._config.get("sample_rate") or SAMPLE_RATE)
        count = 0
        while True:
            try:
                chunk = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue
            pcm = self._to_pcm(chunk, rate)
            yield SensorSample(
                device_id="",
                sensor_id=self._desc.id,
                sensor_type="microphone",
                timestamp=time.time(),
                sequence=next(self._seq),
                payload_type="pcm_s16le",
                payload={
                    "encoding": "pcm_s16le",
                    "sample_rate": rate,
                    "channels": CHANNELS,
                    "duration_s": CHUNK_MS / 1000.0,
                    "data": base64.b64encode(pcm).decode("ascii"),
                },
                sample_rate=rate,
                metadata={"adapter": self._desc.adapter,
                          "hardware_id": self._desc.hardware_id},
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                return

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class SoundDeviceMicrophoneAdapter(SensorAdapter):
    """Discovers audio capture devices via sounddevice/PortAudio.

    Works for ALSA, USB audio, I²S HATs, and Windows WASAPI devices —
    the Pi 3B+ needs nothing heavier than this package.
    """

    def __init__(self):
        self._handles: List[_MicrophoneHandle] = []

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="microphone",
            version="0.1.0",
            modalities=("microphone",),
            description="Audio capture devices via sounddevice/PortAudio",
            config_schema={
                "type": "object",
                "properties": {"sample_rate": {"type": "integer"}},
            },
            maintainer="thothcraft",
        )

    def discover(self) -> List[SensorDescriptor]:
        if sd is None:
            return []
        try:
            devices = sd.query_devices()
        except Exception:
            return []
        out: List[SensorDescriptor] = []
        for index, dev in enumerate(devices):
            if int(dev.get("max_input_channels") or 0) < 1:
                continue
            hostapi = ""
            try:
                hostapi = sd.query_hostapis(dev.get("hostapi", 0))["name"]
            except Exception:
                pass
            hw = f"{hostapi}:{dev.get('name', f'device-{index}')}"
            out.append(SensorDescriptor(
                id=SensorDescriptor.make_id("microphone", hw),
                modality="microphone",
                adapter="microphone",
                name=str(dev.get("name") or f"Microphone {index}"),
                hardware_id=hw,
                capabilities=["pcm_audio", "pcm_s16le", "mono"],
                config_schema=self.metadata().config_schema,
                stable=True,
                metadata={
                    "device_index": index,
                    "hostapi": hostapi,
                    "default_sample_rate": dev.get("default_samplerate"),
                },
            ))
        return out

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _MicrophoneHandle:
        handle = _MicrophoneHandle(descriptor, config)
        self._handles.append(handle)
        return handle

    def health(self) -> HealthReport:
        return HealthReport(
            status="ok" if sd is not None else "error",
            detail="" if sd is not None else "sounddevice not installed")

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


__all__ = ["SoundDeviceMicrophoneAdapter"]
