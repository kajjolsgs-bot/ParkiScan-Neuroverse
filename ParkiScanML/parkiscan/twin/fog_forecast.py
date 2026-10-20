"""Freezing-of-gait FORECASTING on the real Daphnet recordings (seconds ahead of onset).

Labels (per 4 s window, 0.5 s hop, aligned to the window END = "now"):
  * fog       – at least half of the window is annotated FoG (detection target, used by the twin's "now" state)
  * pre_fog   – the window contains no FoG and the next FoG onset is ≤ H s after "now"   (forecast positive)
  * walk      – no FoG in the window and no onset within H s                            (forecast negative)
Windows that contain any FoG sample are excluded from forecast training/scoring (they are detection, not prediction).

Features are computed causally (only samples up to "now"): spectral band powers, Moore–Bächlin freeze index,
cadence (dominant locomotor frequency), step regularity (autocorrelation), jerk, plus trend features over the
previous 2–6 s, because the literature describes gait deterioration (rising cadence, shrinking steps, rising
freeze-band energy) shortly before a freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .daphnet_meta import DAPHNET_TABLE_I, ehr_features

FS = 64.0
WIN = 256          # 4 s
HOP = 32           # 0.5 s
SENSORS = {"ankle": (1, 2, 3), "thigh": (4, 5, 6), "trunk": (7, 8, 9)}
AXES = ("fwd", "vert", "lat")


@dataclass
class Run:
    subject: str
    run: str
    acc: dict          # sensor -> (n, 3) array in g
    ann: np.ndarray    # 0 outside protocol, 1 no freeze, 2 freeze


def find_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("S??R0?.txt"))


def load_runs(root: Path) -> list[Run]:
    runs = []
    for f in find_files(root):
        a = np.loadtxt(f)
        acc = {s: a[:, c[0]:c[2] + 1] / 1000.0 for s, c in SENSORS.items()}
        runs.append(Run(f.stem[:3], f.stem, acc, a[:, 10].astype(int)))
    return runs


def episodes(ann: np.ndarray) -> list[tuple[int, int]]:
    """(onset, end) sample indices of contiguous FoG (==2) runs."""
    fog = (ann == 2).astype(np.int8)
    d = np.diff(np.concatenate([[0], fog, [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


# ------------------------------------------------------------------ vectorised window features
_FREQ = np.fft.rfftfreq(WIN, 1.0 / FS)
_HANN = np.hanning(WIN)


def _band(P, lo, hi):
    m = (_FREQ >= lo) & (_FREQ < hi)
    return P[:, m].sum(axis=1)


def _signal_feats(x: np.ndarray, prefix: str) -> dict:
    """x: (N, WIN) windows of one signal → feature columns."""
    x0 = x - x.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(x0 * _HANN, axis=1)) ** 2 / WIN
    loco, frz, tot = _band(P, 0.5, 3.0), _band(P, 3.0, 8.0), _band(P, 0.5, 8.0)
    m = (_FREQ >= 0.5) & (_FREQ < 3.0)
    dom = _FREQ[m][np.argmax(P[:, m], axis=1)]
    Pn = P[:, (_FREQ >= 0.5) & (_FREQ < 8.0)]
    Pn = Pn / (Pn.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(Pn * np.log(Pn + 1e-12)).sum(axis=1)
    # step regularity: max normalised autocorrelation at lags 0.35–1.4 s
    F = np.fft.rfft(x0, n=2 * WIN, axis=1)
    ac = np.fft.irfft(F * np.conj(F), axis=1)[:, :WIN]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(0.35 * FS), int(1.4 * FS)
    reg = ac[:, lo:hi].max(axis=1)
    jerk = np.sqrt((np.diff(x, axis=1) * FS) ** 2).mean(axis=1)
    return {
        f"{prefix}_std": x0.std(axis=1),
        f"{prefix}_p_loco": np.log1p(loco),
        f"{prefix}_p_freeze": np.log1p(frz),
        f"{prefix}_p_total": np.log1p(tot),
        f"{prefix}_fi": np.log1p(frz) - np.log1p(loco),     # log freeze index (stable)
        f"{prefix}_dom_f": dom,
        f"{prefix}_spec_ent": ent,
        f"{prefix}_step_reg": reg,
        f"{prefix}_jerk": np.log1p(jerk),
    }


def _windows(sig: np.ndarray, starts: np.ndarray) -> np.ndarray:
    idx = starts[:, None] + np.arange(WIN)[None, :]
    return sig[idx]


TREND_KEYS = ("_fi", "_dom_f", "_p_loco", "_step_reg", "_std")


def run_features(run: Run, sensors=("ankle", "thigh", "trunk")) -> pd.DataFrame:
    n = len(run.ann)
    starts = np.arange(0, n - WIN + 1, HOP)
    ends = starts + WIN                         # exclusive; "now" = ends / FS
    cols: dict = {}
    for s in sensors:
        a = run.acc[s]
        for k, ax in enumerate(AXES):
            cols.update(_signal_feats(_windows(a[:, k], starts), f"{s}_{ax}"))
        mag = np.sqrt((a ** 2).sum(axis=1))
        cols.update(_signal_feats(_windows(mag, starts), f"{s}_mag"))
    df = pd.DataFrame(cols)
    # causal trend features (per run, ordered in time): change vs 2 s / 6 s ago and 6 s rolling variability
    trend = {}
    for c in [c for c in df.columns if c.endswith(TREND_KEYS) and ("_vert" in c or "_mag" in c)]:
        trend[c + "_d2s"] = df[c] - df[c].shift(4)
        trend[c + "_d6s"] = df[c] - df[c].shift(12)
        trend[c + "_sd6s"] = df[c].rolling(12, min_periods=4).std()
    df = pd.concat([df, pd.DataFrame(trend)], axis=1)
    # labels
    ann = run.ann
    win_ann = _windows(ann, starts)
    in_protocol = (win_ann != 0).all(axis=1)
    fog_frac = (win_ann == 2).mean(axis=1)
    any_fog = (win_ann == 2).any(axis=1)
    onsets = np.array([o for o, _ in episodes(ann)], dtype=float)
    now = ends.astype(float)
    if len(onsets):
        j = np.searchsorted(onsets, now, side="left")
        nxt = np.where(j < len(onsets), onsets[np.minimum(j, len(onsets) - 1)], np.inf)
    else:
        nxt = np.full(len(now), np.inf)
    df["t_now"] = now / FS
    df["tto_s"] = (nxt - now) / FS                # seconds until next onset (inf if none)
    df["fog"] = (fog_frac >= 0.5).astype(int)
    df["any_fog"] = any_fog.astype(int)
    df["in_protocol"] = in_protocol.astype(int)
    df["subject"] = run.subject
    df["run"] = run.run
    ehr = pd.DataFrame({k: np.full(len(df), v) for k, v in ehr_features(run.subject).items()})
    return pd.concat([df, ehr], axis=1)


def build_table(root: Path, sensors=("ankle", "thigh", "trunk")) -> pd.DataFrame:
    parts = [run_features(r, sensors) for r in load_runs(root)]
    df = pd.concat(parts, ignore_index=True)
    return df[df["in_protocol"] == 1].reset_index(drop=True)


def feature_columns(df: pd.DataFrame, sensors=("ankle",), with_ehr: bool = False) -> list[str]:
    meta = {"t_now", "tto_s", "fog", "any_fog", "in_protocol", "subject", "run"}
    cols = [c for c in df.columns if c not in meta and not c.startswith("ehr_") and c.split("_")[0] in sensors]
    if with_ehr:
        cols += [c for c in df.columns if c.startswith("ehr_")]
    return cols


def forecast_labels(df: pd.DataFrame, horizon_s: float) -> pd.Series:
    """1 = onset within horizon (and no FoG in the window), 0 = no onset within horizon; NaN = excluded (FoG present)."""
    y = pd.Series(np.where(df["tto_s"] <= horizon_s, 1.0, 0.0), index=df.index)
    y[df["any_fog"] == 1] = np.nan
    return y


__all__ = ["DAPHNET_TABLE_I", "build_table", "feature_columns", "forecast_labels", "load_runs", "episodes", "FS", "HOP", "WIN"]
