"""Device compute capability probe (§13).

Reports what the platform actually exposes. Unknown metrics stay
``None`` — never fabricate thermal or GPU data. Model routing matches
on these capabilities, never on device names.
"""

from __future__ import annotations

import os
import platform
import shutil
from typing import Any, Dict, List, Optional

from .contracts import ComputeCapability


def _thermal() -> Optional[Dict[str, Any]]:
    """Temperatures only when the OS exposes them; else None."""
    try:
        import psutil
        temps = getattr(psutil, "sensors_temperatures", None)
        if temps is None:
            return None
        reading = temps() or {}
        if not reading:
            return None
        out: Dict[str, Any] = {}
        for name, entries in reading.items():
            current = [e.current for e in entries
                       if getattr(e, "current", None) is not None]
            if current:
                out[name] = {"current_c": max(current),
                             "high_c": max(
                                 (e.high for e in entries
                                  if getattr(e, "high", None)), default=None)}
        return out or None
    except Exception:
        return None


def _battery() -> tuple:
    try:
        import psutil
        bat = psutil.sensors_battery()
        if bat is None:
            return None, None
        return ({"percent": bat.percent,
                 "secs_left": bat.secsleft if bat.secsleft >= 0 else None},
                bool(bat.power_plugged))
    except Exception:
        return None, None


def _gpus() -> List[Dict[str, Any]]:
    """Best-effort GPU inventory; empty list when undetectable."""
    gpus: List[Dict[str, Any]] = []
    try:  # NVIDIA via nvidia-smi
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                name, _, mem = line.partition(",")
                gpus.append({"name": name.strip(), "vendor": "nvidia",
                             "vram_mb": int(float(mem.strip()))
                             if mem.strip().replace(".", "").isdigit()
                             else None})
    except Exception:
        pass
    return gpus


def probe_compute(storage_path: Optional[str] = None) -> ComputeCapability:
    """Probe this machine's compute capability."""
    arch = platform.machine() or ""
    cpu_count = os.cpu_count()
    mem_total = mem_avail = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        mem_total = int(vm.total // (1024 * 1024))
        mem_avail = int(vm.available // (1024 * 1024))
    except Exception:
        pass
    try:
        usage = shutil.disk_usage(storage_path or os.getcwd())
        storage_mb = int(usage.free // (1024 * 1024))
    except Exception:
        storage_mb = None
    battery, charging = _battery()
    gpus = _gpus()
    return ComputeCapability(
        architecture=arch,
        logical_cpu_count=cpu_count,
        memory_total_mb=mem_total,
        memory_available_mb=mem_avail,
        gpu=gpus,
        accelerators=[],
        vram_mb=(gpus[0].get("vram_mb") if gpus else None),
        storage_available_mb=storage_mb,
        battery=battery,
        charging=charging,
        thermal=_thermal(),
        network={},
    )


def model_fits(manifest_resources: Dict[str, Any],
               capability: ComputeCapability) -> List[str]:
    """Return unmet resource requirements (empty = compatible).

    ``manifest_resources`` comes from a whispy-model/v2 ``resources``
    block, e.g. ``{"min_memory_mb": 1024, "requires_gpu": true,
    "accelerators": ["tpu"]}``.
    """
    unmet: List[str] = []
    if not manifest_resources:
        return unmet
    min_mem = manifest_resources.get("min_memory_mb")
    if min_mem is not None:
        if capability.memory_total_mb is None:
            unmet.append("memory_total_mb unknown")
        elif capability.memory_total_mb < int(min_mem):
            unmet.append(
                f"memory {capability.memory_total_mb}MB < {min_mem}MB")
    min_storage = manifest_resources.get("min_storage_mb")
    if min_storage is not None:
        if capability.storage_available_mb is None:
            unmet.append("storage_available_mb unknown")
        elif capability.storage_available_mb < int(min_storage):
            unmet.append(
                f"storage {capability.storage_available_mb}MB "
                f"< {min_storage}MB")
    if manifest_resources.get("requires_gpu") and not capability.gpu:
        unmet.append("gpu required but none detected")
    needed_acc = [a for a in (manifest_resources.get("accelerators") or [])]
    missing = [a for a in needed_acc if a not in capability.accelerators]
    if missing:
        unmet.append(f"accelerators missing: {missing}")
    return unmet


__all__ = ["probe_compute", "model_fits"]
