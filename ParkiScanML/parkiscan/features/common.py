"""Shared signal utilities for feature extraction."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import signal


def clean(v):
    """Convert numpy scalars / NaN / inf to JSON-safe python values (NaN -> None)."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def stats(x: Iterable[float], prefix: str) -> dict:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {f"{prefix}_mean": None, f"{prefix}_std": None, f"{prefix}_cv": None, f"{prefix}_n": 0}
    m = float(a.mean())
    s = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {f"{prefix}_mean": m, f"{prefix}_std": s, f"{prefix}_cv": (s / m) if m else None, f"{prefix}_n": int(a.size)}


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate an irregularly sampled signal onto a uniform grid at fs Hz."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    order = np.argsort(t)
    t, x = t[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]
    if t.size < 2:
        return t, x
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, x)


def band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    x = np.asarray(x, float)
    if x.size < 8:
        return float("nan")
    nper = int(min(x.size, max(16, fs * 4)))
    f, p = signal.welch(x - x.mean(), fs=fs, nperseg=nper)
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(p[m], f[m])) if m.any() else float("nan")


def dominant_freq(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    x = np.asarray(x, float)
    if x.size < 16:
        return float("nan")
    nper = int(min(x.size, max(32, fs * 4)))
    f, p = signal.welch(x - x.mean(), fs=fs, nperseg=nper)
    m = (f >= lo) & (f < hi)
    return float(f[m][np.argmax(p[m])]) if m.any() else float("nan")


def dfa_alpha(x: np.ndarray, min_box: int = 4, max_box: int | None = None) -> float:
    """Detrended fluctuation analysis scaling exponent (alpha). NaN when the series is too short (< 32)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 32:
        return float("nan")
    y = np.cumsum(x - x.mean())
    max_box = max_box or n // 4
    sizes = np.unique(np.floor(np.logspace(np.log10(min_box), np.log10(max_box), 12)).astype(int))
    fl = []
    for s in sizes:
        k = n // s
        if k < 2:
            continue
        segs = y[: k * s].reshape(k, s)
        tt = np.arange(s)
        res = []
        for seg in segs:
            c = np.polyfit(tt, seg, 1)
            res.append(np.mean((seg - np.polyval(c, tt)) ** 2))
        fl.append((s, math.sqrt(np.mean(res))))
    if len(fl) < 3:
        return float("nan")
    s, f = np.array(fl).T
    good = f > 0
    return float(np.polyfit(np.log(s[good]), np.log(f[good]), 1)[0])
