"""Simulated ESP32/MPU-6050 stream for development without hardware.

Every packet carries sim=true and the device id prefix "SIM-" so simulated data can never be
mistaken for a participant recording. The signal model is a toy: gravity + optional walking
(~1.8 Hz steps) + optional tremor (default 5 Hz) + Gaussian noise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .imu import ImuPacket


def generate(seconds: float = 10.0, fs: float = 100.0, tremor_hz: float | None = 5.0, tremor_g: float = 0.05,
             walking: bool = False, placement: str = "wrist", seed: int = 0, dev: str = "SIM-esp32-01") -> list[ImuPacket]:
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    ax = rng.normal(0, 0.01, n)
    ay = -1.0 + rng.normal(0, 0.01, n)
    az = rng.normal(0, 0.01, n)
    gx, gy, gz = (rng.normal(0, 0.5, n) for _ in range(3))
    if walking:
        ay += 0.25 * np.sin(2 * np.pi * 1.8 * t) + 0.08 * np.sin(2 * np.pi * 3.6 * t)
        gx += 40 * np.sin(2 * np.pi * 0.9 * t)
    if tremor_hz:
        ax += tremor_g * np.sin(2 * np.pi * tremor_hz * t)
        gz += 25 * tremor_g * 10 * np.sin(2 * np.pi * tremor_hz * t + 0.5)
    return [ImuPacket(dev=dev if dev.startswith("SIM-") else "SIM-" + dev, seq=i, t_ms=int(t[i] * 1000),
                      ax=float(ax[i]), ay=float(ay[i]), az=float(az[i]), gx=float(gx[i]), gy=float(gy[i]),
                      gz=float(gz[i]), placement=placement, sim=True) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description="Print simulated IMU JSON lines (sim=true) to stdout")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--fs", type=float, default=100)
    ap.add_argument("--tremor-hz", type=float, default=5.0)
    ap.add_argument("--walking", action="store_true")
    ap.add_argument("--placement", default="wrist")
    ap.add_argument("--realtime", action="store_true", help="pace output at the sample rate")
    a = ap.parse_args()
    for p in generate(a.seconds, a.fs, a.tremor_hz, walking=a.walking, placement=a.placement):
        sys.stdout.write(json.dumps(p.model_dump()) + "\n")
        if a.realtime:
            sys.stdout.flush()
            time.sleep(1 / a.fs)


if __name__ == "__main__":
    main()
