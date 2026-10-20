"""Spiral / tracing kinematics.

Input: a sequence of samples (t seconds, x, y[, pressure]). Works for the in-game spiral trace
(mouse/touch/controller, screen pixels) and for the UCI digitised-tablet spiral files.
Coordinates are normalised by the drawing's own radius so different screens/tablets are comparable.
Tremor band power needs >= ~20 Hz effective sampling; when sampling is too sparse it is reported as None.
"""
from __future__ import annotations

import numpy as np

from .common import band_power, clean, dominant_freq, resample_uniform

FS = 50.0


def spiral_features(t, x, y, pressure=None, center=None, reference=None) -> dict:
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    warnings = []
    n = t.size
    if n < 10:
        return clean({"valid": False, "warnings": ["fewer than 10 samples"], "features": {}})
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    p = None if pressure is None else np.asarray(pressure, float)[order]
    dur = float(t[-1] - t[0])
    if dur <= 0:
        return clean({"valid": False, "warnings": ["zero duration"], "features": {}})
    native_fs = (n - 1) / dur

    if p is not None and np.nanmax(p) > 0:
        drawing = p > 0
        lifts = int(np.sum((~drawing[1:]) & drawing[:-1]))
    else:
        drawing = np.ones(n, bool)
        lifts = 0

    cx, cy = center if center is not None else (np.median(x), np.median(y))
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    scale = float(np.percentile(r, 95)) or 1.0
    xn, yn, rn = dx / scale, dy / scale, r / scale
    theta = np.unwrap(np.arctan2(yn, xn))

    seg = np.hypot(np.diff(xn), np.diff(yn))
    path_len = float(seg.sum())
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    speed = seg / dt
    speed = speed[np.isfinite(speed)]

    # Archimedean fit r = a + b*theta; deviation measures smoothness of the drawn spiral
    if np.ptp(theta) > np.pi:
        A = np.vstack([np.ones_like(theta), theta]).T
        coef, *_ = np.linalg.lstsq(A, rn, rcond=None)
        resid = rn - A @ coef
        rad_rmse = float(np.sqrt(np.mean(resid ** 2)))
        loops = float(abs(np.ptp(theta)) / (2 * np.pi))
    else:
        resid = rn - rn.mean()
        rad_rmse, loops = float("nan"), float(abs(np.ptp(theta)) / (2 * np.pi))
        warnings.append("less than half a turn drawn; spiral fit skipped")

    feats = {
        "duration_s": dur, "samples": int(n), "native_rate_hz": native_fs, "path_length_norm": path_len,
        "loops": loops, "speed_mean": float(np.mean(speed)) if speed.size else None,
        "speed_cv": float(np.std(speed) / np.mean(speed)) if speed.size and np.mean(speed) > 0 else None,
        "radial_rmse_norm": rad_rmse, "pen_lifts": lifts,
    }
    if p is not None:
        pd = p[drawing]
        feats["pressure_mean"] = float(pd.mean()) if pd.size else None
        feats["pressure_cv"] = float(pd.std() / pd.mean()) if pd.size and pd.mean() > 0 else None

    # uniform resample for jerk + tremor band
    if native_fs >= 20:
        tu, xu = resample_uniform(t, xn, FS)
        _, yu = resample_uniform(t, yn, FS)
        _, ru = resample_uniform(t, resid if resid.size == t.size else rn, FS)
        if tu.size > 32:
            vx, vy = np.gradient(xu, 1 / FS), np.gradient(yu, 1 / FS)
            ax, ay = np.gradient(vx, 1 / FS), np.gradient(vy, 1 / FS)
            jx, jy = np.gradient(ax, 1 / FS), np.gradient(ay, 1 / FS)
            v_mean = np.mean(np.hypot(vx, vy)) or 1e-9
            # dimensionless squared jerk (Hogan & Sternad style), log-scaled
            jerk = np.trapezoid(jx ** 2 + jy ** 2, dx=1 / FS) * dur ** 5 / (path_len ** 2 + 1e-9)
            feats["log_dimensionless_jerk"] = float(np.log(jerk + 1e-12))
            tremor = band_power(ru, FS, 4, 8)
            total = band_power(ru, FS, 0.2, 20)
            feats["tremor_band_ratio_4_8hz"] = float(tremor / total) if total and total > 0 else None
            feats["tremor_peak_hz"] = dominant_freq(ru, FS, 3, 12)
            feats["velocity_mean_norm"] = float(v_mean)
    else:
        warnings.append(f"sampling {native_fs:.1f} Hz too low for tremor-band analysis")
        feats.update({"log_dimensionless_jerk": None, "tremor_band_ratio_4_8hz": None, "tremor_peak_hz": None})

    if reference is not None:
        ref = np.asarray(reference, float)
        if ref.ndim == 2 and ref.shape[1] == 2 and ref.shape[0] > 3:
            rx, ry = (ref[:, 0] - cx) / scale, (ref[:, 1] - cy) / scale
            d = np.sqrt((xn[:, None] - rx[None]) ** 2 + (yn[:, None] - ry[None]) ** 2).min(axis=1)
            feats["reference_deviation_mean_norm"] = float(d.mean())
            feats["reference_deviation_p95_norm"] = float(np.percentile(d, 95))
    return clean({"valid": True, "warnings": warnings, "features": feats,
                  "method_notes": ["coordinates normalised by 95th-percentile radius",
                                   "tremor band 4-8 Hz on Archimedean-fit residual, 50 Hz resample",
                                   "not diagnostic"]})


def load_tablet_file(path) -> dict[int, np.ndarray]:
    """Read a UCI spiral file (X;Y;Z;Pressure;GripAngle;Timestamp;TestID) -> {test_id: array}."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 7:
                continue
            try:
                rows.append([float(v) for v in parts[:7]])
            except ValueError:
                continue
    a = np.asarray(rows, float)
    return {int(tid): a[a[:, 6] == tid] for tid in np.unique(a[:, 6])} if a.size else {}


def tablet_features(arr: np.ndarray) -> dict:
    """Features for one test from a tablet file. Timestamp column is in milliseconds."""
    t = (arr[:, 5] - arr[0, 5]) / 1000.0
    return spiral_features(t, arr[:, 0], arr[:, 1], pressure=arr[:, 3])
