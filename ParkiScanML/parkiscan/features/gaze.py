"""Gaze / eye-movement features (webcam face-mesh iris estimates or an eye tracker).

Input samples: {"t": s, "x": float, "y": float, "valid": bool} in normalised screen coords (0-1)
or degrees (set units="deg"). Fixations use I-VT; webcam iris tracking is coarse (~2-5 deg), so
saccade metrics are indicative only.
"""
from __future__ import annotations

import numpy as np

from .common import clean


def gaze_features(samples: list[dict], screen_width_deg: float = 40.0, units: str = "norm",
                  velocity_threshold_deg_s: float = 60.0, min_fix_s: float = 0.1,
                  targets: list[dict] | None = None, target_radius: float = 0.08) -> dict:
    if len(samples) < 10:
        return clean({"valid": False, "warnings": ["fewer than 10 samples"], "features": {}})
    t = np.asarray([s["t"] for s in samples], float)
    x = np.asarray([s.get("x", np.nan) if s.get("valid", True) else np.nan for s in samples], float)
    y = np.asarray([s.get("y", np.nan) if s.get("valid", True) else np.nan for s in samples], float)
    loss = float(np.mean(~np.isfinite(x)))
    k = screen_width_deg if units == "norm" else 1.0
    xd, yd = x * k, y * k
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    vel = np.hypot(np.diff(xd), np.diff(yd)) / dt
    is_fix = np.concatenate([[False], vel < velocity_threshold_deg_s])
    fixes, start = [], None
    for i, m in enumerate(np.append(is_fix, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if t[min(i, t.size - 1)] - t[start] >= min_fix_s:
                fixes.append((start, i))
            start = None
    fix_durs = [t[min(b, t.size - 1)] - t[a] for a, b in fixes]
    sacc = vel[np.isfinite(vel) & (vel >= velocity_threshold_deg_s)]
    amps = []
    for (a1, b1), (a2, _) in zip(fixes, fixes[1:]):
        p1 = np.nanmean(np.c_[xd[a1:b1], yd[a1:b1]], 0)
        p2 = np.nanmean(np.c_[xd[a2:a2 + 3], yd[a2:a2 + 3]], 0)
        amps.append(float(np.hypot(*(p2 - p1))))
    dur = float(t[-1] - t[0])
    f = {"duration_s": dur, "data_loss_fraction": loss, "fixation_count": len(fixes),
         "fixation_rate_hz": len(fixes) / dur if dur > 0 else None,
         "fixation_duration_mean_s": float(np.mean(fix_durs)) if fix_durs else None,
         "saccade_amplitude_mean_deg": float(np.mean(amps)) if amps else None,
         "saccade_peak_velocity_mean_deg_s": float(np.mean(sacc)) if sacc.size else None}
    if targets:
        hits, lat = 0, []
        for tg in targets:
            w = (t >= tg["t_on"]) & (t <= tg.get("t_off", tg["t_on"] + 2.0))
            d = np.hypot(x[w] - tg["x"], y[w] - tg["y"])
            inside = np.where(np.isfinite(d) & (d <= target_radius))[0]
            if inside.size:
                hits += 1
                lat.append(float(t[w][inside[0]] - tg["t_on"]))
        f["target_hit_rate"] = hits / len(targets)
        f["target_latency_mean_s"] = float(np.mean(lat)) if lat else None
    warnings = ["data loss above 30%"] if loss > 0.3 else []
    return clean({"valid": True, "warnings": warnings, "features": f,
                  "method_notes": [f"I-VT {velocity_threshold_deg_s} deg/s, min fixation {min_fix_s}s",
                                   "webcam gaze is coarse; indicative only"]})
