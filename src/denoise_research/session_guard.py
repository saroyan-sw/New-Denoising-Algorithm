from __future__ import annotations

import ctypes
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PowerStatus:
    battery_percent: int | None
    on_ac: bool | None
    battery_present: bool
    source: str


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_uint32),
        ("BatteryFullLifeTime", ctypes.c_uint32),
    ]


def read_power_status(simulated_percent: int | None = None) -> PowerStatus:
    """Read power state without WMI; simulation is only for watchdog tests."""
    if simulated_percent is not None:
        if not 0 <= simulated_percent <= 100:
            raise ValueError("simulated_percent must be in [0, 100]")
        return PowerStatus(simulated_percent, False, True, "simulation")
    if os.name != "nt":
        return PowerStatus(None, None, False, "unsupported_platform")
    raw = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(raw)):
        return PowerStatus(None, None, False, "GetSystemPowerStatus_failed")
    battery_present = raw.BatteryFlag not in (128, 255)
    percent = None if raw.BatteryLifePercent == 255 else int(raw.BatteryLifePercent)
    on_ac = None if raw.ACLineStatus == 255 else raw.ACLineStatus == 1
    return PowerStatus(percent, on_ac, battery_present, "GetSystemPowerStatus")


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(temp_name, target)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def guarded_checkpoint(
    checkpoint_path: str | Path,
    state: Mapping[str, Any],
    *,
    minimum_percent: int = 15,
    simulated_percent: int | None = None,
) -> PowerStatus:
    """Atomically save progress, then stop if a discharging battery is too low."""
    power = read_power_status(simulated_percent)
    payload = dict(state)
    payload["checkpoint_unix_time"] = time.time()
    payload["power"] = asdict(power)
    atomic_write_json(checkpoint_path, payload)
    if (
        power.battery_present
        and power.battery_percent is not None
        and power.battery_percent <= minimum_percent
    ):
        raise RuntimeError(
            f"Battery watchdog stop: {power.battery_percent}% <= {minimum_percent}%; "
            f"progress saved to {Path(checkpoint_path)}"
        )
    return power

