"""MMW-HAT radar adapter — BGT60TR13C driven by the custom spidev stack.

Unlike :mod:`whispy_sensor_dreamhat` (which wraps ``ifxradarsdk``), the
MMW-HAT ships its own Python driver — ``utility/BGT60TR13C.py`` inside an
``MMW-HAT-Release`` tree — that bit-bangs the chip over ``spidev`` plus two
GPIOs (reset/IRQ) and publishes complete frames on a queue.

The adapter locates that tree, imports the driver *in place* (so upstream
fixes in the repo apply without a reinstall), configures the radar from a
``radar_config`` register/settings pair, and emits one ``SensorSample`` per
frame shaped ``[rx, chirps, samples]``::

    WHISPY_MMWHAT_DIR=/home/gad/Desktop/thoth/WS/MMW-HAT/MMW-HAT-Release

    payload = {"encoding": "radar_frame", "shape": [3, 64, 128],
               "snr_db": ..., "range_profile": [...], "xy_map": [[...]]}

``discover()`` probes the chip id over SPI; it returns ``[]`` when the
release tree, ``spidev``, or the shield itself is absent.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import logging
import os
import queue
import re
import sys
import time
import types
from typing import Any, Dict, Iterator, List, Optional, Tuple

from whispy.contracts import SensorDescriptor, SensorSample
from whispy.devices.base import SensorHandle
from whispy.sensors.base import HealthReport, SensorAdapter, SensorMeta

logger = logging.getLogger(__name__)

HARDWARE_ID = "mmwhat-bgt60tr13c"
ENV_RELEASE_DIR = "WHISPY_MMWHAT_DIR"

_REG_CFG_RE = re.compile(r"BGT60TR13C_export_registers_\d{8}-\d{6}\.txt")
_SETTINGS_RE = re.compile(r"BGT60TR13C_settings_\d{8}-\d{6}\.json")

#: Locations probed for an MMW-HAT-Release tree (env var wins).
_DEFAULT_RELEASE_DIRS = (
    "~/Desktop/thoth/WS/MMW-HAT/MMW-HAT-Release",
    "~/Desktop/MMW-HAT/MMW-HAT-Release",
    "~/MMW-HAT-Release",
    "/opt/mmwhat/MMW-HAT-Release",
)
_DEFAULT_CONFIG_NAME = "config_3rx_3m"

_driver_cache: Dict[str, Any] = {}


def _release_candidates(explicit: Optional[str]) -> Iterator[str]:
    if explicit:
        yield os.path.expanduser(explicit)
    env = os.getenv(ENV_RELEASE_DIR, "").strip()
    if env:
        yield os.path.expanduser(env)
    for path in _DEFAULT_RELEASE_DIRS:
        yield os.path.expanduser(path)


def _find_release_dir(explicit: Optional[str] = None) -> Optional[str]:
    for path in _release_candidates(explicit):
        if (os.path.isfile(os.path.join(path, "utility", "BGT60TR13C.py"))
                and os.path.isdir(os.path.join(path, "radar_config"))):
            return path
    return None


def _load_driver(release_dir: str):
    """Import ``utility.BGT60TR13C`` from the release tree, once per path.

    The driver does ``from utility.BGT60TR13C_CONST import *``; both modules
    are loaded under a synthetic ``utility`` package so no ``sys.path``
    mutation is needed and the release tree stays untouched.
    """
    cached = _driver_cache.get(release_dir)
    if cached is not None:
        return cached
    util_dir = os.path.join(release_dir, "utility")
    const_path = os.path.join(util_dir, "BGT60TR13C_CONST.py")
    drv_path = os.path.join(util_dir, "BGT60TR13C.py")

    if "utility" not in sys.modules:
        pkg = types.ModuleType("utility")
        pkg.__path__ = [util_dir]  # type: ignore[attr-defined]
        sys.modules["utility"] = pkg

    for mod_name, path in (("utility.BGT60TR13C_CONST", const_path),
                           ("utility.BGT60TR13C", drv_path)):
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {mod_name} from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    cls = sys.modules["utility.BGT60TR13C"].BGT60TR13C
    _driver_cache[release_dir] = cls
    return cls


def _find_one(directory: str, pattern: re.Pattern) -> str:
    matches = [f for f in os.listdir(directory) if pattern.match(f)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected 1 file matching {pattern.pattern} in {directory}, "
            f"found {len(matches)}")
    return os.path.join(directory, matches[0])


def _resolve_config_dir(release_dir: str,
                        config_dir: Optional[str]) -> Optional[str]:
    """Pick a radar_config dir containing both register + settings files."""
    candidates: List[str] = []
    if config_dir:
        candidates.append(config_dir
                          if os.path.isabs(config_dir)
                          else os.path.join(release_dir, "radar_config",
                                            config_dir))
    else:
        candidates.append(os.path.join(release_dir, "radar_config",
                                       _DEFAULT_CONFIG_NAME))
        base = os.path.join(release_dir, "radar_config")
        try:
            candidates += [os.path.join(base, d) for d in sorted(
                os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
        except OSError:
            pass
    for cand in candidates:
        try:
            _find_one(cand, _REG_CFG_RE)
            _find_one(cand, _SETTINGS_RE)
            return cand
        except (OSError, RuntimeError):
            continue
    return None


def _parse_radar_cfg(setting: Dict[str, Any]) -> Dict[str, Any]:
    """Chirp/frame geometry from a settings JSON (numba-free helper port)."""
    frame_sequence = setting["sequence"][0]["sequence"]
    frame_rate = 1.0 / setting["sequence"][0]["repetition_time_s"]
    frame = frame_sequence[0]
    if frame.get("type") != "loop":
        raise ValueError("invalid frame type in settings")
    chirp = frame["sequence"][0]
    if chirp.get("type") != "chirp":
        raise ValueError("invalid chirp type in settings")
    rx_mask = chirp["rx_mask"]
    return {
        "frame_rate": frame_rate,
        "num_chirps_per_frame": frame["num_repetitions"],
        "num_samples_per_chirp": chirp["num_samples"],
        "rx_mask": rx_mask,
        "num_antennas": bin(rx_mask).count("1"),
        "sample_rate": chirp["sample_rate_Hz"],
        "bandwidth": chirp["end_frequency_Hz"] - chirp["start_frequency_Hz"],
    }


def _frame_size(setting: Dict[str, Any]) -> int:
    """Total ADC samples per frame (matches helper.calculate_frame_size)."""
    total = 0
    for frame in setting["sequence"][0]["sequence"]:
        if frame["type"] != "loop":
            continue
        chirp_samples = sum(
            c["num_samples"] * bin(c["rx_mask"]).count("1")
            for c in frame["sequence"] if c["type"] == "chirp")
        total += chirp_samples * frame["num_repetitions"]
    return total


def _read_uint12(raw: bytes) -> Any:
    import numpy as np
    data = np.frombuffer(raw, dtype=np.uint8)
    fst, mid, lst = np.reshape(
        data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    hi = (fst << 4) + (mid >> 4)
    lo = ((mid % 16) << 8) + lst
    return np.reshape(np.concatenate(
        (hi[:, None], lo[:, None]), axis=1), 2 * hi.shape[0]).astype(np.float32)


def _decode_frame(raw: bytes, cfg: Dict[str, Any]) -> Optional[Any]:
    """version|seq|len header + 12-bit data → [rx, chirps, samples] array."""
    import numpy as np
    if len(raw) < 12:
        return None
    version = int.from_bytes(raw[0:4], "little")
    if version != 0:
        return None
    data_len = int.from_bytes(raw[8:12], "little")
    if len(raw) - 12 != data_len:
        return None
    adc = _read_uint12(raw[12:])
    frames = adc.reshape((1, cfg["num_chirps_per_frame"],
                          cfg["num_samples_per_chirp"], cfg["num_antennas"]))
    return np.transpose(frames[0], (2, 0, 1))


def _frame_payload(arr: Any) -> Dict[str, Any]:
    """One frame → normalized payload (same keys as the dreamhat adapter)."""
    import numpy as np
    magnitude = np.abs(arr.astype(np.complex128)
                     if np.iscomplexobj(arr) else arr.astype(float))
    signal = float(magnitude.max()) if magnitude.size else 0.0
    noise = float(np.median(magnitude)) if magnitude.size else 0.0
    snr_db = 20.0 * float(np.log10((signal + 1e-9) / (noise + 1e-9)))
    # Range profile: energy per range bin (last axis), averaged over
    # antennas + chirps.
    if arr.ndim >= 3:
        range_profile = magnitude.mean(axis=tuple(range(arr.ndim - 1)))
    else:
        range_profile = magnitude
    # Downsampled range×chirp energy map (first antenna), 24×24.
    xy_map: List[List[float]] = []
    if arr.ndim >= 3:
        m2 = magnitude[0]
    elif arr.ndim == 2:
        m2 = magnitude
    else:
        m2 = None
    if m2 is not None:
        target = 24
        ys = np.linspace(0, m2.shape[0], target + 1).astype(int)
        xs = np.linspace(0, m2.shape[1], target + 1).astype(int)
        for i in range(target):
            row = []
            for j in range(target):
                block = m2[ys[i]:max(ys[i + 1], ys[i] + 1),
                           xs[j]:max(xs[j + 1], xs[j] + 1)]
                row.append(float(block.mean()))
            xy_map.append(row)
    return {
        "encoding": "radar_frame",
        "shape": list(arr.shape),
        "snr_db": round(snr_db, 3),
        "range_profile": range_profile.ravel()[:128].tolist(),
        "xy_map": xy_map,
        "energy": float((magnitude ** 2).mean()) if magnitude.size else 0.0,
    }


class _MmwhatHandle(SensorHandle):
    """Streams decoded radar frames straight off the driver's queue."""

    def __init__(self, descriptor: SensorDescriptor,
                 config: Optional[Dict[str, Any]] = None):
        self._desc = descriptor
        self._config = dict(config or {})
        self._seq = itertools.count()
        self._dev = None
        self._cfg: Optional[Dict[str, Any]] = None

    @property
    def info(self):
        return self._desc.to_sensor()

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._desc

    def _open(self) -> None:
        if self._dev is not None:
            return
        release_dir = str(self._config.get("release_dir")
                          or self._desc.metadata.get("release_dir") or "")
        if not release_dir:
            raise RuntimeError("MMW-HAT release dir unknown")
        cls = _load_driver(release_dir)
        dev = cls(
            spi_bus=int(self._config.get("spi_bus", 0)),
            spi_dev=int(self._config.get("spi_dev", 0)),
            spi_speed=int(self._config.get("spi_speed", 50_000_000)),
            rst_pin=int(self._config.get("rst_pin", 12)),
            irq_pin=int(self._config.get("irq_pin", 25)),
        )
        try:
            if dev.check_chip_id() != 0:
                raise RuntimeError("BGT60TR13C chip id check failed")
            cfg_dir = str(self._config.get("config_dir")
                          or self._desc.metadata.get("config_dir"))
            reg_fn = _find_one(cfg_dir, _REG_CFG_RE)
            dev.load_register_config_file(reg_fn)
            with open(_find_one(cfg_dir, _SETTINGS_RE)) as fh:
                setting = json.load(fh)
            self._cfg = _parse_radar_cfg(setting)
            dev.set_fifo_parameters(
                _frame_size(setting),
                int(self._config.get("num_samples_irq", 4096)),
                int(self._config.get("num_samples_burst", 2048)),
            )
            if dev.start() != 0:
                raise RuntimeError("radar start failed (soft reset timeout)")
        except Exception:
            dev.stop()
            dev.close()
            raise
        self._dev = dev

    def stream(self, max_samples: Optional[int] = None
               ) -> Iterator[SensorSample]:
        self._open()
        assert self._dev is not None and self._cfg is not None
        count = 0
        try:
            while True:
                try:
                    raw = bytes(self._dev.frame_buffer.get(timeout=1.0))
                except queue.Empty:
                    continue
                arr = _decode_frame(raw, self._cfg)
                if arr is None:
                    continue
                yield SensorSample(
                    device_id="",
                    sensor_id=self._desc.id,
                    sensor_type="radar",
                    timestamp=time.time(),
                    sequence=next(self._seq),
                    payload_type="radar_frame",
                    payload=_frame_payload(arr),
                    metadata={"adapter": "mmwhat-radar",
                              "hardware_id": self._desc.hardware_id},
                )
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
        finally:
            # Release SPI + GPIOs when the consumer stops reading so other
            # processes (the GUI examples) can claim the radar.
            self.close()

    def latest(self) -> Optional[SensorSample]:
        for sample in self.stream(max_samples=1):
            return sample
        return None

    def close(self) -> None:
        if self._dev is not None:
            try:
                self._dev.stop()
            except Exception:
                pass
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None


class MmwhatRadarAdapter(SensorAdapter):
    """Discovers the MMW-HAT BGT60TR13C shield on this host.

    Requires an ``MMW-HAT-Release`` tree (``WHISPY_MMWHAT_DIR`` or the
    default locations) plus ``spidev``/``gpiozero`` — absent hardware or
    driver yields ``[]`` rather than a phantom sensor.
    """

    def __init__(self, release_dir: Optional[str] = None,
                 config_dir: Optional[str] = None):
        self._release_dir = release_dir
        self._config_dir = config_dir
        self._handles: List[_MmwhatHandle] = []

    def metadata(self) -> SensorMeta:
        return SensorMeta(
            name="mmwhat-radar",
            version="0.1.0",
            modalities=("radar",),
            description="MMW-HAT BGT60TR13C 60 GHz radar (spidev driver)",
            config_schema={
                "type": "object",
                "properties": {
                    "release_dir": {"type": "string"},
                    "config_dir": {"type": "string"},
                    "spi_bus": {"type": "integer"},
                    "spi_dev": {"type": "integer"},
                    "spi_speed": {"type": "integer"},
                    "rst_pin": {"type": "integer"},
                    "irq_pin": {"type": "integer"},
                },
            },
            maintainer="thothcraft",
        )

    def _resolve(self) -> Tuple[Optional[str], Optional[str],
                                Optional[Dict[str, Any]]]:
        release_dir = _find_release_dir(self._release_dir)
        if release_dir is None:
            return None, None, None
        cfg_dir = _resolve_config_dir(release_dir, self._config_dir)
        if cfg_dir is None:
            return release_dir, None, None
        try:
            with open(_find_one(cfg_dir, _SETTINGS_RE)) as fh:
                cfg = _parse_radar_cfg(json.load(fh))
        except Exception:
            return release_dir, cfg_dir, None
        return release_dir, cfg_dir, cfg

    def _probe_chip(self, release_dir: str) -> bool:
        try:
            dev = _load_driver(release_dir)(spi_speed=10_000_000)
        except Exception as exc:
            logger.debug("mmwhat driver unavailable: %s", exc)
            return False
        try:
            return dev.check_chip_id() == 0
        except Exception as exc:
            logger.debug("mmwhat chip probe failed: %s", exc)
            return False
        finally:
            try:
                dev.close()
            except Exception:
                pass

    def discover(self) -> List[SensorDescriptor]:
        release_dir, cfg_dir, cfg = self._resolve()
        if release_dir is None or cfg is None:
            return []
        if not self._probe_chip(release_dir):
            return []
        return [SensorDescriptor(
            id=SensorDescriptor.make_id("radar", HARDWARE_ID),
            modality="radar",
            adapter="mmwhat-radar",
            name="MMW-HAT BGT60TR13C",
            hardware_id=HARDWARE_ID,
            capabilities=["radar_frame", "snr_db", "range_profile"],
            config_schema=self.metadata().config_schema,
            stable=True,
            metadata={
                "release_dir": release_dir,
                "config_dir": cfg_dir,
                "num_antennas": cfg["num_antennas"],
                "num_chirps_per_frame": cfg["num_chirps_per_frame"],
                "num_samples_per_chirp": cfg["num_samples_per_chirp"],
                "frame_rate": round(cfg["frame_rate"], 2),
                "rx_mask": cfg["rx_mask"],
            },
        )]

    def connect(self, descriptor: SensorDescriptor,
                config: Optional[Dict[str, Any]] = None) -> _MmwhatHandle:
        handle = _MmwhatHandle(descriptor, config)
        self._handles.append(handle)
        return handle

    def health(self) -> HealthReport:
        release_dir = _find_release_dir(self._release_dir)
        if release_dir is None:
            return HealthReport(status="error",
                                detail="MMW-HAT-Release tree not found")
        try:
            _load_driver(release_dir)
        except Exception as exc:
            return HealthReport(status="error", detail=str(exc))
        return HealthReport(status="ok")

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()


__all__ = ["MmwhatRadarAdapter"]
