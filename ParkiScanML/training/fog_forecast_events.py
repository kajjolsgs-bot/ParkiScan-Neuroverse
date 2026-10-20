"""Event-level operating curve for the deployed freezing-of-gait forecaster.

Post-processes the leave-one-subject-out predictions written by `training.fog_forecast_train`
(reports/fog_forecast_oof.csv.gz) — no retraining, every score is out-of-fold.

Adds to reports/fog_forecast_report.json:
  * "cue_rule"        the pre-specified per-fold thresholds with a 10 s cue refractory period: a wearable cue lasts
                      several seconds, so re-triggering during an active cue is not a new (false) alarm;
  * "operating_curve" one pooled threshold per specificity level → % of freezes warned, false cues / h, lead time.
    This is descriptive (like an ROC curve): it shows the trade-off a clinician would choose from, not a tuned result.

Then rewrites reports/fog_forecast_report.md.

    python -m training.fog_forecast_events
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from parkiscan.twin.fog_forecast import FS, episodes, load_runs
from training.fog_forecast_train import CONSEC, ROOT, WARN_WINDOW_S, write_md

REFRACTORY_S = 10.0
SPEC_LEVELS = (0.85, 0.90, 0.95, 0.97, 0.99)


def evaluate(oof: pd.DataFrame, runs: dict, thr_of, consec: int = CONSEC, refractory_s: float = REFRACTORY_S) -> dict:
    warned = total = false_cues = 0
    nonfog_s = 0.0
    leads = []
    for (subject, run), g in oof.groupby(["subject", "run"]):
        g = g.sort_values("t_now")
        t, p = g["t_now"].to_numpy(), g["p"].to_numpy()
        a = p >= thr_of(subject)
        al = a.copy()
        for k in range(1, consec):
            al[k:] &= a[:-k]
            al[:k] = False
        cues, last = [], -1e9
        for i in np.nonzero(al)[0]:
            if t[i] - last >= refractory_s:
                cues.append(t[i])
                last = t[i]
        eps = [(o / FS, e / FS) for o, e in episodes(runs[run].ann)]
        for o, e in eps:
            if not ((t >= o - WARN_WINDOW_S) & (t <= o)).any():
                continue
            total += 1
            hit = np.nonzero(al & (t >= o - WARN_WINDOW_S) & (t < o))[0]
            if len(hit):
                warned += 1
                leads.append(o - t[hit[0]])
        false_cues += sum(1 for c in cues if not any(o - WARN_WINDOW_S <= c <= e + 2.0 for o, e in eps))
        infog = np.zeros(len(t), bool)
        for o, e in eps:
            infog |= (t >= o) & (t <= e)
        nonfog_s += (~infog).sum() * 0.5
    hours = nonfog_s / 3600.0
    return {"episodes": total, "warned_before_onset": warned, "early_warning_rate": warned / max(1, total),
            "false_cues_per_hour": false_cues / max(1e-9, hours), "median_lead_s": float(np.median(leads)) if leads else None,
            "non_fog_hours": hours}


def main():
    rp = ROOT / "reports" / "fog_forecast_report.json"
    rep = json.loads(rp.read_text())
    oof = pd.read_csv(ROOT / "reports" / "fog_forecast_oof.csv.gz")
    runs = {r.run: r for r in load_runs(ROOT / "data" / "raw" / "daphnet")}
    fold_thr = {s: v["threshold"] for s, v in rep["deployed"]["per_subject"].items()}

    cue = evaluate(oof, runs, lambda s: fold_thr[s])
    cue["rule"] = (f"per-fold threshold (training-subject specificity 0.85), {CONSEC} consecutive windows, "
                   f"{REFRACTORY_S:.0f} s cue refractory period")
    rep["deployed"]["cue_rule"] = cue
    neg = oof.loc[(oof["any_fog"] == 0) & (oof["tto_s"] > rep["deployed"]["horizon_s"]), "p"].to_numpy()
    curve = []
    for q in SPEC_LEVELS:
        th = float(np.quantile(neg, q))
        r = evaluate(oof, runs, lambda s, th=th: th)
        curve.append({"window_specificity": q, "threshold": th, **r})
        print(f"spec {q:.2f}  thr {th:.3f}  warned {r['early_warning_rate']:.0%}  false cues/h {r['false_cues_per_hour']:.1f}"
              f"  lead {r['median_lead_s']:.1f} s")
    rep["deployed"]["operating_curve"] = curve
    rp.write_text(json.dumps(rep, indent=1))
    mp = ROOT / "models" / "fog_forecast.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
        meta["metrics"].update({"false_cues_per_hour": cue["false_cues_per_hour"], "cue_refractory_s": REFRACTORY_S})
        meta["cue_refractory_s"] = REFRACTORY_S
        mp.write_text(json.dumps(meta, indent=1))
    write_md(rep)
    print(json.dumps(cue, indent=1))


if __name__ == "__main__":
    main()
