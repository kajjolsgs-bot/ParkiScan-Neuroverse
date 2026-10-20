"""Upper-body movement features from BlazePose/MediaPipe landmarks streamed by Unity.

Input frames: {"t": seconds, "landmarks": [[x, y, z, visibility], ...33]} in normalised image coords.
Indices follow MediaPipe Pose: 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 27/28 ankles.
"""
from __future__ import annotations

import numpy as np

from .common import band_power, clean, dominant_freq, resample_uniform
from .gait import pose_gait_features

L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 15, 16, 23, 24, 27, 28
MIN_VIS = 0.5


def _series(frames, idx):
    t, xs, ys = [], [], []
    for fr in frames:
        lm = fr.get("landmarks") or []
        if len(lm) > idx and (len(lm[idx]) < 4 or (lm[idx][3] or 0) >= MIN_VIS):
            t.append(fr["t"]); xs.append(lm[idx][0]); ys.append(lm[idx][1])
        else:
            t.append(fr["t"]); xs.append(np.nan); ys.append(np.nan)
    return np.asarray(t, float), np.asarray(xs, float), np.asarray(ys, float)


def pose_features(frames: list[dict]) -> dict:
    if len(frames) < 20:
        return clean({"valid": False, "warnings": ["fewer than 20 frames"], "features": {}})
    t, _, _ = _series(frames, 0)
    dur = float(t[-1] - t[0])
    fps = (len(t) - 1) / dur if dur > 0 else 0
    f: dict = {"frames": len(frames), "duration_s": dur, "fps": fps}
    warnings = []
    _, lsx, lsy = _series(frames, L_SH)
    _, rsx, rsy = _series(frames, R_SH)
    shoulder_w = np.nanmedian(np.abs(lsx - rsx)) or 0.1
    f["visibility_upper_body"] = float(np.mean(np.isfinite(lsx) & np.isfinite(rsx)))
    for side, wi, sx, sy in (("left", L_WR, lsx, lsy), ("right", R_WR, rsx, rsy)):
        _, wx, wy = _series(frames, wi)
        rel_y = (wy - sy) / shoulder_w
        rel_x = (wx - sx) / shoulder_w
        ok = np.isfinite(rel_y)
        f[f"{side}_wrist_visible"] = float(ok.mean())
        if ok.sum() > 10:
            f[f"{side}_arm_swing_range"] = float(np.nanpercentile(rel_x, 95) - np.nanpercentile(rel_x, 5))
            if fps >= 15:
                tu, yu = resample_uniform(t[ok], rel_y[ok], 30.0)
                if yu.size > 64:
                    tb = band_power(yu, 30.0, 4, 7)
                    tot = band_power(yu, 30.0, 0.3, 12)
                    f[f"{side}_wrist_tremor_ratio_4_7hz"] = float(tb / tot) if tot > 0 else None
                    f[f"{side}_wrist_peak_hz"] = dominant_freq(yu, 30.0, 3, 12)
    if f.get("left_arm_swing_range") and f.get("right_arm_swing_range"):
        l, r = f["left_arm_swing_range"], f["right_arm_swing_range"]
        f["arm_swing_asymmetry_pct"] = float(abs(l - r) / ((l + r) / 2) * 100) if (l + r) else None
    _, lhx, lhy = _series(frames, L_HIP)
    _, rhx, rhy = _series(frames, R_HIP)
    mid_sh = np.stack([(lsx + rsx) / 2, (lsy + rsy) / 2], 1)
    mid_hip = np.stack([(lhx + rhx) / 2, (lhy + rhy) / 2], 1)
    v = mid_sh - mid_hip
    ok = np.all(np.isfinite(v), 1)
    if ok.sum() > 10:
        lean = np.degrees(np.arctan2(v[ok, 0], -v[ok, 1]))
        f["trunk_lean_deg_mean"] = float(np.mean(lean))
        f["trunk_sway_deg_std"] = float(np.std(lean))
    if fps < 15:
        warnings.append(f"pose frame rate {fps:.1f} fps is below 15; tremor bands skipped")
    _, _, lay = _series(frames, L_ANK)
    _, _, ray = _series(frames, R_ANK)
    gait = pose_gait_features(t, lay, ray)
    if gait.get("valid"):
        f.update({f"gait_{k}": v for k, v in gait["features"].items()})
    else:
        warnings.append("ankles not visible enough for step timing")
    return clean({"valid": True, "warnings": warnings, "features": f,
                  "method_notes": ["distances normalised by shoulder width", "visibility threshold 0.5",
                                   "webcam pose estimates are approximate and framing dependent"]})
