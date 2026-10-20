"""ESP32 + MPU-6050 IMU packets: schema, validation, CSV storage and summaries.

Wire format (one JSON object per line over USB serial at 921600 baud, or POSTed in batches):
  {"v":1,"dev":"esp32-wrist-01","seq":1234,"t_ms":56789,"ax":0.01,"ay":-0.98,"az":0.12,
   "gx":1.2,"gy":-0.4,"gz":0.3,"placement":"wrist","sim":false}
Units: acceleration in g, angular rate in deg/s, t_ms = device millis().
Packets with "sim": true come from the simulator and are always labelled SIMULATED downstream.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..features.common import band_power, clean, dominant_freq
from ..features.gait import accel_gait_features

FIELDS = ["v", "dev", "seq", "t_ms", "ax", "ay", "az", "gx", "gy", "gz", "placement", "sim"]
ACC_RANGE_G = 16.0
GYRO_RANGE_DPS = 2000.0


class ImuPacket(BaseModel):
    type: str = "imu"
    v: int = 1
    dev: str = Field(min_length=1, max_length=64)
    seq: int = Field(ge=0)
    t_ms: int = Field(ge=0)
    t_us: int | None = Field(default=None, ge=0)
    ax: float
    ay: float
    az: float
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    placement: str = "wrist"
    sim: bool = False

    @field_validator("placement")
    @classmethod
    def _placement(cls, v):
        return (v or "").strip().lower()

    @field_validator("ax", "ay", "az")
    @classmethod
    def _acc(cls, v):
        if not np.isfinite(v) or abs(v) > ACC_RANGE_G:
            raise ValueError("acceleration outside ±16 g")
        return v

    @field_validator("gx", "gy", "gz")
    @classmethod
    def _gyr(cls, v):
        if not np.isfinite(v) or abs(v) > GYRO_RANGE_DPS:
            raise ValueError("angular rate outside ±2000 deg/s")
        return v


def parse_line(line: str) -> ImuPacket | None:
    """Data lines only (status lines and anything malformed return None)."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
        if obj.get("type", "imu") != "imu":
            return None
        return ImuPacket(**obj)
    except Exception:
        return None


def append_csv(path: Path, packets: Iterable[ImuPacket]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    n = 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for p in packets:
            w.writerow(p.model_dump())
            n += 1
    return n


def summarize(packets: list[ImuPacket]) -> dict:
    if len(packets) < 32:
        return clean({"valid": False, "warnings": ["fewer than 32 packets"], "features": {}})
    packets = sorted(packets, key=lambda p: (p.dev, p.seq))
    seqs = np.asarray([p.seq for p in packets])
    t = np.asarray([p.t_ms for p in packets], float) / 1000.0
    gaps = int(np.sum(np.diff(seqs) > 1))
    dt = np.diff(t)
    fs = float(1.0 / np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    ax, ay, az = (np.asarray([getattr(p, k) for p in packets], float) for k in ("ax", "ay", "az"))
    gx, gy, gz = (np.asarray([getattr(p, k) for p in packets], float) for k in ("gx", "gy", "gz"))
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    f = {"packets": len(packets), "rate_hz": fs, "sequence_gaps": gaps, "duration_s": float(t[-1] - t[0]),
         "simulated": any(p.sim for p in packets), "placement": packets[0].placement}
    fs = round(fs, 2)
    if fs >= 20:
        # remove gravity/posture with a 0.5 Hz high-pass on each axis, then take the principal movement
        # axis (|acc| would fold lateral tremor to twice its frequency)
        from scipy import signal as sg
        b, a = sg.butter(2, 0.5, btype="high", fs=fs)
        acc = np.vstack([sg.filtfilt(b, a, v) for v in (ax, ay, az)]) if len(ax) > 15 else np.vstack([ax, ay, az])
        acc = acc - acc.mean(axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(acc.T, full_matrices=False)
        a0 = acc.T @ vt[0]
        tot = band_power(a0, fs, 0.3, min(20, fs / 2 - 0.5))
        f["acc_tremor_ratio_4_7hz"] = float(band_power(a0, fs, 4, 7) / tot) if tot > 0 else None
        f["acc_peak_hz"] = dominant_freq(a0, fs, 1, min(12, fs / 2 - 0.5))
        f["gyro_rms_dps"] = float(np.sqrt(np.mean(gmag ** 2)))
        f["gyro_tremor_ratio_4_7hz"] = float(band_power(gmag - gmag.mean(), fs, 4, 7) /
                                             (band_power(gmag - gmag.mean(), fs, 0.3, min(20, fs / 2 - 0.5)) or np.nan))
        if f["placement"] in {"ankle", "shank", "foot", "waist", "trunk"}:
            g = accel_gait_features(t, ax, ay, az, fs=fs)
            f.update({f"gait_{k}": v for k, v in g.get("features", {}).items()})
    warnings = []
    if fs < 50:
        warnings.append(f"sample rate {fs:.1f} Hz below the 100 Hz firmware target")
    if gaps:
        warnings.append(f"{gaps} sequence gaps (dropped packets)")
    if f["simulated"]:
        warnings.append("SIMULATED DATA — not from a real sensor")
    return clean({"valid": True, "warnings": warnings, "features": f})
