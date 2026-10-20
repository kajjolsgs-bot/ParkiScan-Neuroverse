"""Simulated consumer-wearable streams (Apple Health / Google Fit style) for a twin, and — for the synthetic cohort
only — simulated ground-truth freezing episodes from a documented generative hazard model.

Streams on a 15-minute grid for `days` days:
  steps (per 15 min), heart rate (bpm), motion-derived gait-instability index (from an ankle/wrist IMU FoG detector,
  0–1 per 15 min), and one record per night: total sleep, deep / REM / light minutes, awakenings, nightly RMSSD (HRV).

The hazard model (ASSUMPTIONS, used to create synthetic labels; stated in the model card):
  log λ(t) = b0 + 1.1·fog_trait_z + 0.25·(H&Y − 2.5) + 2.2·OFF(t) + 0.35·sleep_debt_h + 0.25·low_HRV_z
             + 0.6·walking(t) + u_i                       (λ per 15 min, u_i ~ N(0, 0.35²) individual effect)
OFF(t) = 1 − E(t) from the levodopa model. FoG being more frequent in OFF and triggered while walking is documented
(Conde et al., Front. Neurol. 2023); the sleep and HRV terms are hypotheses included to show multi-stream fusion.
"""
from __future__ import annotations

import math

import numpy as np

from . import pkpd

STEP_MIN = 15
PER_DAY = 24 * 60 // STEP_MIN

HAZ = {"b0": -6.1, "trait": 1.1, "hy": 0.25, "off": 2.2, "sleep_debt": 0.35, "low_hrv": 0.25, "walk": 0.6, "u_sd": 0.35}


def _circadian(h: np.ndarray) -> np.ndarray:
    """Relative activity by hour of day (0 at night, peaks late morning and early evening)."""
    awake = ((h >= 6.5) & (h < 22.5)).astype(float)
    shape = 0.55 + 0.45 * np.exp(-((h - 10.5) / 2.5) ** 2) + 0.35 * np.exp(-((h - 17.5) / 2.0) ** 2)
    return awake * shape


def simulate(summary: dict, days: int = 7, seed: int = 0, dose_shift_h: float = 0.0, extra_doses=None,
             dose_mg: float | None = None, sleep_delta_h: float = 0.0, with_events: bool = True, sleep_from_day: int = 0) -> dict:
    # independent random streams, so a what-if scenario changes ONLY what it modifies (same nights, noise, person)
    ss = np.random.SeedSequence(seed)
    r_night, r_steps, r_hr, r_u, r_ev, r_det = (np.random.default_rng(x) for x in ss.spawn(6))
    reg = summary["regimen"]
    mg = dose_mg if dose_mg else reg["levodopa_mg"]
    n = days * PER_DAY
    t = np.arange(n) * STEP_MIN / 60.0                      # hours since day 0, 00:00
    hod = t % 24
    doses = pkpd.schedule(reg["times_h"], mg, days + 1, start_day=-1, shift_h=dose_shift_h, extra=extra_doses)
    c, e = pkpd.effect(t, doses, summary["pd_ec50"], summary.get("pd_ldr", 0.0), ce0=0.0)
    off = 1 - e
    # ---- nights (sleep ends the morning of day d)
    nights = []
    for d in range(days):
        z = r_night.standard_normal(4); wu = r_night.random()
        tst = float(np.clip(summary["sleep_baseline_h"] + (sleep_delta_h if d >= sleep_from_day else 0.0) + 0.55 * z[0], 3.0, 9.5))
        deep = max(10.0, (0.14 + 0.03 * z[1]) * tst * 60)
        rem = max(10.0, (0.18 - (0.03 if summary.get("rem_sleep_behaviour_disorder") else 0) + 0.03 * z[2]) * tst * 60)
        lam_w = 2.5 + 0.08 * summary["disease_duration_y"]
        wake = int(np.clip(round(lam_w + np.sqrt(lam_w) * (wu * 2 - 1) * 1.7), 0, 15))
        rmssd = float(np.clip(summary["hrv_baseline_ms"] - 1.8 * max(0, summary["sleep_need_h"] - tst) + 4.5 * z[3], 6, 90))
        nights.append({"day": d, "total_sleep_h": round(tst, 2), "deep_min": round(deep), "rem_min": round(rem),
                       "light_min": round(max(0, tst * 60 - deep - rem)), "awakenings": wake, "rmssd_ms": round(rmssd, 1),
                       "sleep_debt_h": round(max(0.0, summary["sleep_need_h"] - tst), 2)})
    day_idx = (t // 24).astype(int)
    debt = np.array([nights[d]["sleep_debt_h"] for d in day_idx])
    hrv_z = np.array([(summary["hrv_baseline_ms"] - nights[d]["rmssd_ms"]) / 6.0 for d in day_idx])
    # ---- activity: steps fall when OFF and with poor sleep
    act = _circadian(hod) * (1 - 0.45 * off) * (1 - 0.05 * debt)
    lam_steps = summary["steps_baseline"] / (PER_DAY * 0.52) * act
    noise = r_steps.lognormal(0, 0.35, n); unif = r_steps.random(n)
    steps = np.floor(np.maximum(0, lam_steps * noise) + unif)            # monotone in lam: same noise, same person
    walking = np.clip(steps / 400.0, 0, 1.5)
    hr = 64 + 0.022 * steps + 4 * off + r_hr.normal(0, 2.5, n)
    hr[_circadian(hod) == 0] -= 6
    out = {"t_h": t, "hour": hod, "plasma": c, "effect": e, "off": off, "steps": steps, "hr": np.round(hr, 1),
           "walking": walking, "sleep_debt": debt, "hrv_z": hrv_z, "nights": nights, "doses": doses}
    # ---- hazard and (synthetic) freezing episodes
    trait_z = (summary["fog_trait"] - 0.35) / 0.2
    u = r_u.normal(0, HAZ["u_sd"])
    loglam = (HAZ["b0"] + HAZ["trait"] * trait_z + HAZ["hy"] * (summary["hoehn_yahr"] - 2.5) + HAZ["off"] * off
              + HAZ["sleep_debt"] * debt + HAZ["low_hrv"] * np.clip(hrv_z, -2, 3) + HAZ["walk"] * walking + u)
    lam = np.exp(loglam) * (walking > 0.02)
    out["hazard"] = lam
    if with_events:
        fog = r_ev.poisson(lam)
        out["fog_events"] = fog.astype(int)
        # wearable FoG detector (Daphnet-trained, ~0.84 sens / 0.77 spec per window, aggregated) → instability index
        det = np.clip(0.55 * np.minimum(fog, 2) + r_det.beta(1.2, 12, n) * walking, 0, 1)
        out["instability"] = np.round(det, 3)
    return out


def hours_label(t_h: np.ndarray, events: np.ndarray, horizon_h: float = 2.0) -> np.ndarray:
    """1 if at least one FoG episode occurs in (t, t + horizon]."""
    k = int(round(horizon_h * 60 / STEP_MIN))
    cs = np.concatenate([[0], np.cumsum(events)])
    idx = np.arange(len(events))
    hi = np.minimum(idx + 1 + k, len(events))
    return ((cs[hi] - cs[idx + 1]) > 0).astype(int)
