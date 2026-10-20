"""Two-hour-ahead freezing-of-gait risk: feature builder shared by training and the live twin.

Inputs at time t (all available at t):
  STATIC (EHR)    age, disease duration, H&Y, MDS-UPDRS III, NFOG-Q, falls, MoCA, LEDD, comorbidities, genotype flags
  PK (EHR × time) dopaminergic effect now and its forecast minimum over the next 2 h (from the prescribed schedule),
                  minutes since the last / until the next levodopa dose
  WEARABLE        steps in the last 1 h / 2 h, heart rate, last night's sleep (total, debt, deep, REM, awakenings),
                  nightly HRV (RMSSD) relative to baseline, gait-instability index (IMU FoG detector) in the last 2 h / 24 h
Target: ≥ 1 freezing episode in (t, t + 2 h].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pkpd
from .sim import PER_DAY, STEP_MIN

STATIC = ["age", "disease_duration_y", "hoehn_yahr", "updrs3_off", "nfogq", "falls_12m", "moca", "ledd_mg",
          "orthostatic_hypotension", "depression", "rem_sleep_behaviour_disorder", "gba_carrier"]
PK = ["pk_effect_now", "pk_effect_min_2h", "pk_off_frac_2h", "min_since_dose", "min_to_dose"]
WEAR = ["hour_sin", "hour_cos", "steps_1h", "steps_2h", "hr_1h", "sleep_total_h", "sleep_debt_h", "deep_min", "rem_min",
        "awakenings", "rmssd_rel", "instab_2h", "instab_24h"]
GROUPS = {"ehr_only": STATIC, "wearable_only": WEAR, "ehr+wearable": STATIC + WEAR, "digital_twin (ehr+wearable+pk)": STATIC + WEAR + PK}


def _roll_sum(x: np.ndarray, k: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x)) + 1
    return cs[i] - cs[np.maximum(0, i - k)]


def features(summary: dict, sim: dict) -> pd.DataFrame:
    t = sim["t_h"]
    n = len(t)
    k2 = int(120 / STEP_MIN)
    e = sim["effect"]
    # forecast over the next 2 h from the (known) schedule — mechanistic, no future sensor data used
    from numpy.lib.stride_tricks import sliding_window_view
    ep = np.concatenate([e, np.full(k2, e[-1])])
    fut_min = sliding_window_view(ep, k2 + 1).min(axis=1)[:n]
    offp = np.concatenate([(e < pkpd.OFF_THRESHOLD).astype(float), np.zeros(k2)])
    fut_off = sliding_window_view(offp[1:], k2).mean(axis=1)[:n] if n > 1 else np.zeros(n)
    if len(fut_off) < n:
        fut_off = np.concatenate([fut_off, np.zeros(n - len(fut_off))])
    dts = np.sort(np.array([d for d, _ in sim["doses"]], float))
    j = np.searchsorted(dts, t, side="right")
    since = np.where(j > 0, (t - dts[np.maximum(j - 1, 0)]) * 60, 600.0)
    to = np.where(j < len(dts), (dts[np.minimum(j, len(dts) - 1)] - t) * 60, 600.0)
    day = (t // 24).astype(int)
    N = sim["nights"]
    hod = sim["hour"]
    inst = sim.get("instability", np.zeros(n))
    f = {
        "pk_effect_now": e, "pk_effect_min_2h": fut_min, "pk_off_frac_2h": fut_off,
        "min_since_dose": np.minimum(since, 600), "min_to_dose": np.minimum(to, 600),
        "hour_sin": np.sin(2 * np.pi * hod / 24), "hour_cos": np.cos(2 * np.pi * hod / 24),
        "steps_1h": _roll_sum(sim["steps"], 4), "steps_2h": _roll_sum(sim["steps"], 8),
        "hr_1h": _roll_sum(sim["hr"], 4) / 4,
        "sleep_total_h": np.array([N[d]["total_sleep_h"] for d in day]), "sleep_debt_h": np.array([N[d]["sleep_debt_h"] for d in day]),
        "deep_min": np.array([N[d]["deep_min"] for d in day]), "rem_min": np.array([N[d]["rem_min"] for d in day]),
        "awakenings": np.array([N[d]["awakenings"] for d in day]),
        "rmssd_rel": np.array([N[d]["rmssd_ms"] for d in day]) / max(1.0, summary["hrv_baseline_ms"]),
        "instab_2h": _roll_sum(inst, 8), "instab_24h": _roll_sum(inst, PER_DAY),
    }
    df = pd.DataFrame(f)
    for c in STATIC:
        v = summary[c]
        df[c] = float(v) if not isinstance(v, bool) else float(v)
    df["t_h"] = t
    df["hour"] = hod
    return df
