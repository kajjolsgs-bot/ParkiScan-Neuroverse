"""Train the digital twin's 2-hour freezing-risk model on the SYNTHETIC cohort (patient-grouped 5-fold CV).

    python -m training.twin_risk_train            # ~3–6 min

Writes models/twin_risk.joblib + .json and reports/twin_risk_report.json / .md.
Because labels come from the documented simulator (parkiscan/twin/sim.py), these metrics show that the fusion
pipeline can recover risk signals from EHR + wearables + pharmacokinetics; they are NOT clinical accuracy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from parkiscan.twin.ehr import synthetic_cohort
from parkiscan.twin.risk import GROUPS, features
from parkiscan.twin.sim import HAZ, hours_label, simulate

ROOT = Path(__file__).resolve().parents[1]
N_PATIENTS, DAYS, SUB = 400, 7, 2       # 400 synthetic patients × 7 days, every 2nd 15-min step (30 min)


def build():
    rows = []
    for i, rec in enumerate(synthetic_cohort(N_PATIENTS)):
        s = rec["summary"]
        sim = simulate(s, days=DAYS, seed=1000 + i)
        df = features(s, sim)
        df["y"] = hours_label(sim["t_h"], sim["fog_events"], 2.0)
        df["pid"] = s["id"]
        awake = (df["hour"] >= 6.5) & (df["hour"] < 22.0)
        df = df[awake & (df["t_h"] >= 24)].iloc[::SUB]          # day 0 = burn-in (24 h history needed)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
                                          l2_regularization=1.0, random_state=0)


def cv(df, cols, k=5):
    pids = np.array(sorted(df["pid"].unique()))
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(pids), k)
    oof = np.zeros(len(df))
    for f in folds:
        te = df["pid"].isin(f).to_numpy()
        m = model().fit(df.loc[~te, cols], df.loc[~te, "y"])
        oof[te] = m.predict_proba(df.loc[te, cols])[:, 1]
    return oof


def calib(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for a, b in zip(q[:-1], q[1:]):
        m = (p >= a) & (p <= b)
        if m.sum():
            out.append({"pred": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    return out


def run():
    t0 = time.time()
    df = build()
    y = df["y"].to_numpy()
    print(f"rows {len(df)}, patients {df['pid'].nunique()}, prevalence {y.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    rep = {"data": f"synthetic cohort: {N_PATIENTS} patients × {DAYS} days, 30-min samples while awake, day 0 burn-in",
           "target": "≥1 freezing episode in the next 2 h (simulated ground truth)", "validation": "patient-grouped 5-fold CV",
           "hazard_parameters": HAZ, "n_rows": int(len(df)), "prevalence": float(y.mean()), "results": {}}
    for name, cols in GROUPS.items():
        p = cv(df, cols)
        rep["results"][name] = {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                                "brier": float(brier_score_loss(y, p)), "n_features": len(cols)}
        if name.startswith("digital_twin"):
            rep["calibration"] = calib(y, p)
            # top-decile alerting: what share of 2-h windows with a freeze fall in the highest-risk 10 %?
            thr = np.quantile(p, 0.9)
            rep["top_decile"] = {"threshold": float(thr), "capture": float(y[p >= thr].sum() / max(1, y.sum())),
                                 "precision": float(y[p >= thr].mean()), "lift": float(y[p >= thr].mean() / y.mean())}
        print(f"{name:32s} AUC {rep['results'][name]['auc']:.3f}  PR-AUC {rep['results'][name]['pr_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    cols = GROUPS["digital_twin (ehr+wearable+pk)"]
    final = model().fit(df[cols], df["y"])
    joblib.dump(final, ROOT / "models" / "twin_risk.joblib")
    meta = {"dataset": "synthetic_twin_cohort", "features": cols, "horizon_h": 2.0, "metrics": rep["results"]["digital_twin (ehr+wearable+pk)"],
            "top_decile": rep["top_decile"], "evidence": "simulated labels — demonstrates the fusion pipeline, not clinical accuracy",
            "usable_for_research_scores": False}
    (ROOT / "models" / "twin_risk.json").write_text(json.dumps(meta, indent=1))
    rep["runtime_s"] = time.time() - t0
    (ROOT / "reports" / "twin_risk_report.json").write_text(json.dumps(rep, indent=1))
    L = ["# Digital twin — 2-hour freezing risk (synthetic cohort)", "", rep["data"] + ".", "", "Validation: " + rep["validation"] +
         f". Prevalence of a freeze within 2 h: {rep['prevalence']:.1%}.", "",
         "| inputs | ROC AUC | PR AUC | Brier |", "|---|---|---|---|"]
    for k, v in rep["results"].items():
        L.append(f"| {k} | {v['auc']:.3f} | {v['pr_auc']:.3f} | {v['brier']:.4f} |")
    td = rep["top_decile"]
    L += ["", f"Top-decile alerts capture {td['capture']:.0%} of the 2-h windows that contain a freeze (precision {td['precision']:.0%}, "
          f"lift {td['lift']:.1f}×).", "", "**Read honestly:** labels are generated by the simulator in `parkiscan/twin/sim.py` "
          "(hazard parameters listed in the JSON report). The comparison shows that fusing static records, wearables and the "
          "levodopa model recovers more of the simulated risk than any single stream; real-world accuracy is unknown."]
    (ROOT / "reports" / "twin_risk_report.md").write_text("\n".join(L) + "\n")
    print(json.dumps(rep["top_decile"]))


if __name__ == "__main__":
    run()
