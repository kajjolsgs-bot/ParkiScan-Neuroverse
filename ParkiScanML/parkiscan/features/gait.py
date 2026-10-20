"""Gait features from stride-interval series, accelerometer signals, or pose-derived ankle tracks.

* stride_features: stride/swing/stance series (seconds or %) -> variability (CV), asymmetry, DFA alpha.
* accel_gait_features: 3-axis acceleration -> cadence, step regularity and the Moore/Bächlin
  freeze index (power 3-8 Hz / power 0.5-3 Hz), used in the Daphnet FoG literature.
These are movement descriptors for research and engagement feedback, not diagnoses.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .common import band_power, clean, dfa_alpha, stats


def stride_features(left_stride=None, right_stride=None, stride=None, swing_pct=None, stance_pct=None,
                    double_support_pct=None) -> dict:
    feats: dict = {}
    series = {"left_stride_s": left_stride, "right_stride_s": right_stride, "stride_s": stride,
              "swing_pct": swing_pct, "stance_pct": stance_pct, "double_support_pct": double_support_pct}
    for name, s in series.items():
        if s is None:
            continue
        a = np.asarray(s, float)
        a = a[np.isfinite(a)]
        # drop physiologically impossible stride times and outliers (>3 MAD), e.g. turns
        if name.endswith("_s") and a.size:
            a = a[(a > 0.3) & (a < 3.0)]
        if a.size > 5:
            med = np.median(a)
            mad = np.median(np.abs(a - med)) or 1e-9
            a = a[np.abs(a - med) < 3 * 1.4826 * mad]
        feats.update(stats(a, name))
        if name.endswith("stride_s") and a.size >= 32:
            feats[f"{name}_dfa_alpha"] = dfa_alpha(a)
    if left_stride is not None and right_stride is not None:
        l = np.nanmean(np.asarray(left_stride, float))
        r = np.nanmean(np.asarray(right_stride, float))
        feats["stride_asymmetry_pct"] = float(abs(l - r) / ((l + r) / 2) * 100) if (l + r) else None
    return clean({"features": feats, "valid": bool(feats)})


def freeze_index(acc_vertical: np.ndarray, fs: float) -> float:
    loco = band_power(acc_vertical, fs, 0.5, 3.0)
    freeze = band_power(acc_vertical, fs, 3.0, 8.0)
    return float(freeze / loco) if loco and loco > 0 else float("nan")


def accel_window_features(ax, ay, az, fs: float) -> dict:
    """Compact descriptor of one acceleration window (used for Daphnet training and live IMU)."""
    ax, ay, az = (np.asarray(v, float) for v in (ax, ay, az))
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    f = {}
    for name, s in (("x", ax), ("y", ay), ("z", az), ("mag", mag)):
        s0 = s - s.mean()
        f[f"{name}_std"] = float(s0.std())
        f[f"{name}_p_loco"] = band_power(s0, fs, 0.5, 3.0)
        f[f"{name}_p_freeze"] = band_power(s0, fs, 3.0, 8.0)
        f[f"{name}_freeze_index"] = (f[f"{name}_p_freeze"] / f[f"{name}_p_loco"]) if f[f"{name}_p_loco"] > 0 else np.nan
    f["mag_total_power"] = band_power(mag - mag.mean(), fs, 0.5, 8.0)
    return f


def accel_gait_features(t, ax, ay, az, fs: float | None = None) -> dict:
    t = np.asarray(t, float)
    ax, ay, az = (np.asarray(v, float) for v in (ax, ay, az))
    if t.size < 32:
        return clean({"valid": False, "warnings": ["fewer than 32 samples"], "features": {}})
    fs = fs or (t.size - 1) / (t[-1] - t[0])
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    m0 = mag - mag.mean()
    b, a = signal.butter(2, [0.5, min(8.0, fs / 2 - 0.5)], btype="band", fs=fs)
    mf = signal.filtfilt(b, a, m0) if m0.size > 15 else m0
    ac = np.correlate(mf, mf, "full")[mf.size - 1:]
    ac = ac / (ac[0] or 1)
    lo, hi = int(0.3 * fs), int(2.0 * fs)
    feats = {"duration_s": float(t[-1] - t[0]), "rate_hz": float(fs)}
    if hi < ac.size:
        pk, _ = signal.find_peaks(ac[lo:hi])
        if pk.size:
            step_lag = (pk[0] + lo) / fs
            feats["cadence_steps_per_min"] = float(60.0 / step_lag)
            feats["step_regularity"] = float(ac[pk[0] + lo])
            k2 = 2 * (pk[0] + lo)
            feats["stride_regularity"] = float(ac[k2]) if k2 < ac.size else None
    feats["freeze_index"] = freeze_index(m0, fs)
    feats["activity_rms_g"] = float(np.sqrt(np.mean(m0 ** 2)))
    return clean({"valid": True, "features": feats, "warnings": [],
                  "method_notes": ["autocorrelation cadence on 0.5-8 Hz band-passed magnitude",
                                   "freeze index = P(3-8 Hz)/P(0.5-3 Hz) (Moore 2008; Bächlin 2010)"]})


def pose_gait_features(t, left_ankle_y, right_ankle_y) -> dict:
    """Step timing from pose-estimated ankle heights (normalised image y, smaller = higher)."""
    t = np.asarray(t, float)
    la, ra = np.asarray(left_ankle_y, float), np.asarray(right_ankle_y, float)
    ok = np.isfinite(la) & np.isfinite(ra)
    if ok.sum() < 30:
        return clean({"valid": False, "warnings": ["not enough visible ankle frames"], "features": {}})
    t, la, ra = t[ok], la[ok], ra[ok]
    fs = (t.size - 1) / (t[-1] - t[0])
    diff = la - ra
    b, a = signal.butter(2, min(3.0, fs / 2 - 0.1), btype="low", fs=fs)
    d = signal.filtfilt(b, a, diff - diff.mean()) if diff.size > 15 else diff
    pk, _ = signal.find_peaks(d, distance=max(1, int(0.35 * fs)), prominence=np.std(d) * 0.5)
    tr, _ = signal.find_peaks(-d, distance=max(1, int(0.35 * fs)), prominence=np.std(d) * 0.5)
    left_strides = np.diff(t[pk])
    right_strides = np.diff(t[tr])
    res = stride_features(left_stride=left_strides, right_stride=right_strides)
    res["features"]["step_count"] = int(pk.size + tr.size)
    res["features"]["cadence_steps_per_min"] = float((pk.size + tr.size) / (t[-1] - t[0]) * 60) if t[-1] > t[0] else None
    res["method_notes"] = ["steps from alternating left-right ankle height difference (webcam pose)",
                           "camera framing strongly affects accuracy"]
    return clean(res)
