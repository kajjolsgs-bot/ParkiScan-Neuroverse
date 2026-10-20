"""Train and evaluate freezing-of-gait FORECASTING on Daphnet (leave-one-subject-out).

    python -m training.fog_forecast_train                  # full run (~10 min on a laptop)
    python -m training.fog_forecast_train --quick          # fewer configurations

Outputs
  models/fog_forecast.joblib + models/fog_forecast.json   (final model trained on all 10 participants)
  reports/fog_forecast_report.json / .md                   (all metrics, per subject, with CIs)
  reports/fog_forecast_oof.csv.gz                          (out-of-fold risk per window, used by the twin replay)

Evaluation is honest by construction:
  * every metric is out-of-fold, leave-one-SUBJECT-out (the test person is never seen in training);
  * the alarm threshold of each fold is chosen on the training subjects only (inner grouped CV);
  * a physics baseline (Moore–Bächlin freeze index) and a constant baseline are reported next to the model;
  * IMU-only vs IMU + real clinical header (Table I) is compared, and the better one is not hand-picked:
    the "deployed" configuration is fixed in advance as ankle IMU + EHR, matching the game's ankle sensor.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from parkiscan.twin.fog_forecast import FS, build_table, episodes, feature_columns, forecast_labels, load_runs
from parkiscan.twin.daphnet_meta import CITATION

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)
WARN_WINDOW_S = 5.0        # an alarm in the 5 s before onset counts as an early warning
CONSEC = 2                 # alarm = 2 consecutive windows (1 s) above threshold
TARGET_SPEC = 0.85


def model():
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                          min_samples_leaf=40, random_state=0)


def weights(y):
    pos = y.mean()
    return np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))


def fit(X, y):
    m = model()
    m.fit(X, y, sample_weight=weights(y))
    return m


def boot_auc(y, p, groups, n=1000):
    subs = np.unique(groups)
    vals = []
    for _ in range(n):
        pick = RNG.choice(subs, len(subs), replace=True)
        idx = np.concatenate([np.nonzero(groups == s)[0] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def inner_threshold(Xtr, ytr, gtr):
    """Out-of-fold risk on the TRAINING subjects (3 grouped folds) → threshold at the target specificity."""
    subs = np.unique(gtr)
    folds = np.array_split(RNG.permutation(subs), 3)
    oof = np.full(len(ytr), np.nan)
    for f in folds:
        te = np.isin(gtr, f)
        m = fit(Xtr[~te], ytr[~te])
        oof[te] = m.predict_proba(Xtr[te])[:, 1]
    neg = oof[ytr == 0]
    return float(np.quantile(neg, TARGET_SPEC))


def alarms(p, thr):
    a = p >= thr
    out = a.copy()
    for k in range(1, CONSEC):
        out[k:] &= a[:-k]
        out[:k] = False
    return out


def event_metrics(stream: pd.DataFrame, runs_by_name: dict, thr: float) -> dict:
    """stream: rows for one subject with columns run, t_now, p (all in-protocol windows, time-ordered)."""
    warned, total, leads = 0, 0, []
    false_alarms, nonfog_s = 0, 0.0
    for rn, g in stream.groupby("run"):
        g = g.sort_values("t_now")
        t = g["t_now"].to_numpy()
        al = alarms(g["p"].to_numpy(), thr)
        eps = [(o / FS, e / FS) for o, e in episodes(runs_by_name[rn].ann)]
        for o, e in eps:
            if not ((t >= o - WARN_WINDOW_S) & (t <= o)).any():
                continue      # onset outside the recorded protocol windows
            total += 1
            hit = np.nonzero(al & (t >= o - WARN_WINDOW_S) & (t < o))[0]
            if len(hit):
                warned += 1
                leads.append(o - t[hit[0]])
        # false alarms: alarm rising edges not in [onset-5 s, end+2 s] of any episode
        rise = np.nonzero(al & ~np.concatenate([[False], al[:-1]]))[0]
        for i in rise:
            ok = any(o - WARN_WINDOW_S <= t[i] <= e + 2.0 for o, e in eps)
            if not ok:
                false_alarms += 1
        infog = np.zeros(len(t), bool)
        for o, e in eps:
            infog |= (t >= o) & (t <= e)
        nonfog_s += (~infog).sum() * 0.5
    return {"episodes": total, "warned_before_onset": warned, "lead_times_s": leads,
            "false_alarms": false_alarms, "non_fog_hours": nonfog_s / 3600.0}


def run(quick: bool = False):
    t0 = time.time()
    root = ROOT / "data" / "raw" / "daphnet"
    df = build_table(root)
    runs = {r.run: r for r in load_runs(root)}
    groups_all = df["subject"].to_numpy()
    subjects = sorted(df["subject"].unique())
    configs = {
        "ankle_imu": dict(sensors=("ankle",), with_ehr=False),
        "ankle_imu+ehr": dict(sensors=("ankle",), with_ehr=True),
        "all_imu": dict(sensors=("ankle", "thigh", "trunk"), with_ehr=False),
        "all_imu+ehr": dict(sensors=("ankle", "thigh", "trunk"), with_ehr=True),
    }
    if quick:
        configs = {k: configs[k] for k in ("ankle_imu", "ankle_imu+ehr")}
    horizons = [2.0, 3.0, 5.0]
    (ROOT / "reports").mkdir(exist_ok=True)
    part_path = ROOT / "reports" / "_fog_forecast_partial.json"
    partial = json.loads(part_path.read_text()) if part_path.exists() else {}
    HEAD_CFG, HEAD_H = "ankle_imu+ehr", 3.0
    report = {"dataset": "Daphnet Freezing of Gait (10 participants with PD, 64 Hz, 3 accelerometers)",
              "citation": CITATION, "task": "forecast FoG onset within H seconds (windows containing FoG excluded)",
              "window": "4 s, 0.5 s hop, causal features", "validation": "leave-one-subject-out (10 folds)",
              "n_windows": int(len(df)), "subjects": subjects, "results": {}, "baselines": {}}

    for H in horizons:
        y_all = forecast_labels(df, H)
        keep = y_all.notna().to_numpy()
        y = y_all[keep].to_numpy().astype(int)
        g = groups_all[keep]
        # physics baseline: ankle vertical log freeze index (higher = more freeze-band energy)
        fi = df.loc[keep, "ankle_vert_fi"].to_numpy()
        report["baselines"][f"H{int(H)}"] = {
            "freeze_index_auc": float(roc_auc_score(y, fi)),
            "freeze_index_trend_auc": float(roc_auc_score(y, np.nan_to_num(df.loc[keep, "ankle_vert_fi_d6s"].to_numpy()))),
            "constant_auc": 0.5, "positives": int(y.sum()), "negatives": int((1 - y).sum())}
        for name, cfg in configs.items():
            if H != 3.0 and name.startswith("all_"):
                continue      # the 3-sensor comparison is reported at the deployed horizon only
            key = f"H{int(H)}|{name}"
            if key in partial:
                report["results"].setdefault(f"H{int(H)}", {})[name] = partial[key]; continue
            cols = feature_columns(df, **cfg)
            X = df.loc[keep, cols].to_numpy(np.float32)
            oof = np.zeros(len(y))
            per = {}
            for s in subjects:
                te = g == s
                m = fit(X[~te], y[~te])
                oof[te] = m.predict_proba(X[te])[:, 1]
                if y[te].sum() > 0 and (1 - y[te]).sum() > 0:
                    per[s] = float(roc_auc_score(y[te], oof[te]))
            res = {"auc": float(roc_auc_score(y, oof)), "auc_ci95_subject_bootstrap": boot_auc(y, oof, g),
                   "pr_auc": float(average_precision_score(y, oof)), "prevalence": float(y.mean()),
                   "per_subject_auc": per, "mean_per_subject_auc": float(np.mean(list(per.values()))),
                   "n_features": len(cols)}
            report["results"].setdefault(f"H{int(H)}", {})[name] = res
            partial[key] = res; part_path.write_text(json.dumps(partial))
            print(f"H={H:.0f}s {name:15s} AUC {res['auc']:.3f} {res['auc_ci95_subject_bootstrap']} "
                  f"mean/subj {res['mean_per_subject_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- event-level early warning for the deployed configuration
    cols = feature_columns(df, **configs[HEAD_CFG])
    y_all = forecast_labels(df, HEAD_H)
    Xall = df[cols].to_numpy(np.float32)
    ev_tot = {"episodes": 0, "warned_before_onset": 0, "lead_times_s": [], "false_alarms": 0, "non_fog_hours": 0.0}
    per_ev, thrs, oof_rows = {}, [], []
    win_tp = win_fp = win_tn = win_fn = 0
    for s in subjects:
        tr = (groups_all != s) & y_all.notna().to_numpy()
        Xtr, ytr, gtr = Xall[tr], y_all[tr].to_numpy().astype(int), groups_all[tr]
        thr = inner_threshold(Xtr, ytr, gtr)
        m = fit(Xtr, ytr)
        te_all = groups_all == s
        p = m.predict_proba(Xall[te_all])[:, 1]
        stream = df.loc[te_all, ["run", "t_now", "fog", "any_fog", "tto_s"]].copy()
        stream["p"] = p
        stream["subject"] = s
        oof_rows.append(stream)
        ev = event_metrics(stream, runs, thr)
        per_ev[s] = {**{k: v for k, v in ev.items() if k != "lead_times_s"},
                     "median_lead_s": float(np.median(ev["lead_times_s"])) if ev["lead_times_s"] else None, "threshold": thr}
        for k in ("episodes", "warned_before_onset", "false_alarms", "non_fog_hours"):
            ev_tot[k] += ev[k]
        ev_tot["lead_times_s"] += ev["lead_times_s"]
        thrs.append(thr)
        yt = y_all[te_all].to_numpy()
        ok = ~np.isnan(yt)
        pr = p[ok] >= thr
        yt = yt[ok].astype(int)
        win_tp += int((pr & (yt == 1)).sum()); win_fp += int((pr & (yt == 0)).sum())
        win_tn += int((~pr & (yt == 0)).sum()); win_fn += int((~pr & (yt == 1)).sum())
        print(f"  events {s}: {ev['warned_before_onset']}/{ev['episodes']} warned, FA {ev['false_alarms']} "
              f"in {ev['non_fog_hours']:.2f} h, thr {thr:.3f}", flush=True)
    lt = ev_tot["lead_times_s"]
    report["deployed"] = {
        "configuration": HEAD_CFG, "horizon_s": HEAD_H,
        "alarm_rule": f"risk ≥ fold threshold (training-subject specificity {TARGET_SPEC:.2f}) for {CONSEC} consecutive windows",
        "window_level": {"tp": win_tp, "fp": win_fp, "tn": win_tn, "fn": win_fn,
                         "sensitivity": win_tp / max(1, win_tp + win_fn), "specificity": win_tn / max(1, win_tn + win_fp)},
        "event_level": {"episodes": ev_tot["episodes"], "warned_before_onset": ev_tot["warned_before_onset"],
                        "early_warning_rate": ev_tot["warned_before_onset"] / max(1, ev_tot["episodes"]),
                        "median_lead_s": float(np.median(lt)) if lt else None,
                        "iqr_lead_s": [float(np.percentile(lt, 25)), float(np.percentile(lt, 75))] if lt else None,
                        "false_alarms_per_hour": ev_tot["false_alarms"] / max(1e-9, ev_tot["non_fog_hours"]),
                        "non_fog_hours": ev_tot["non_fog_hours"]},
        "per_subject": per_ev,
    }
    print(json.dumps(report["deployed"]["event_level"], indent=1))

    # ---------------- final model on all subjects
    keep = y_all.notna().to_numpy()
    final = fit(Xall[keep], y_all[keep].to_numpy().astype(int))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models" / "fog_forecast.joblib")
    meta = {"dataset": "daphnet_fog_forecast", "configuration": HEAD_CFG, "horizon_s": HEAD_H, "features": cols,
            "threshold": float(np.median(thrs)), "consecutive_windows": CONSEC, "window_s": 4.0, "hop_s": 0.5,
            "fs_hz": FS, "sensor": "ankle 3-axis accelerometer (g): fwd, vert, lat",
            "ehr_inputs": [c for c in cols if c.startswith("ehr_")],
            "metrics": {"auc": report["results"][f"H{int(HEAD_H)}"][HEAD_CFG]["auc"],
                        "auc_ci95": report["results"][f"H{int(HEAD_H)}"][HEAD_CFG]["auc_ci95_subject_bootstrap"],
                        **report["deployed"]["event_level"]},
            "usable_for_research_scores": True, "intended_use": "research prototype; not a medical device"}
    (ROOT / "models" / "fog_forecast.json").write_text(json.dumps(meta, indent=1))
    (ROOT / "reports").mkdir(exist_ok=True)
    pd.concat(oof_rows).to_csv(ROOT / "reports" / "fog_forecast_oof.csv.gz", index=False, compression="gzip")
    report["runtime_s"] = time.time() - t0
    (ROOT / "reports" / "fog_forecast_report.json").write_text(json.dumps(report, indent=1))
    write_md(report)
    return report


def write_md(r: dict):
    L = ["# Freezing-of-gait forecasting — Daphnet, leave-one-subject-out", "",
         f"Data: {r['dataset']}. {r['n_windows']} windows ({r['window']}). Validation: {r['validation']}.", "",
         "Target: a freeze **starts within H seconds** of the current moment. Windows that already contain freezing are",
         "excluded (that is detection, reported separately in `training_report.md`).", "",
         "## Window-level ROC AUC (pooled out-of-fold, 95% CI by subject bootstrap)", "",
         ]
    names = []
    for res in r["results"].values():
        for k in res:
            if k not in names: names.append(k)
    L += ["| horizon | freeze-index baseline | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for h, res in r["results"].items():
        b = r["baselines"][h]["freeze_index_auc"]
        cells = []
        for k in names:
            v = res.get(k)
            cells.append(f"{v['auc']:.2f} ({v['auc_ci95_subject_bootstrap'][0]:.2f}–{v['auc_ci95_subject_bootstrap'][1]:.2f})" if v else "–")
        L.append(f"| {h[1:]} s | {b:.2f} | " + " | ".join(cells) + " |")
    d = r["deployed"]; ev = d["event_level"]; w = d["window_level"]
    L += ["", f"## Deployed configuration: `{d['configuration']}`, horizon {d['horizon_s']:.0f} s", "",
          f"Alarm rule: {d['alarm_rule']}.", "",
          f"* Window level: sensitivity {w['sensitivity']:.2f}, specificity {w['specificity']:.2f} "
          f"(TP {w['tp']}, FP {w['fp']}, TN {w['tn']}, FN {w['fn']}).",
          f"* Event level: **{ev['warned_before_onset']} of {ev['episodes']} freezing episodes "
          f"({ev['early_warning_rate']:.0%}) were preceded by an alarm in the 5 s before onset**; median lead time "
          f"{ev['median_lead_s']:.1f} s (IQR {ev['iqr_lead_s'][0]:.1f}–{ev['iqr_lead_s'][1]:.1f} s).",
          f"* False alarms: {ev['false_alarms_per_hour']:.1f} per hour of non-freezing walking ({ev['non_fog_hours']:.2f} h).", "",
          "| subject | episodes | warned | median lead (s) | false alarms | hours |", "|---|---|---|---|---|---|"]
    for s, v in d["per_subject"].items():
        ml = f"{v['median_lead_s']:.1f}" if v["median_lead_s"] is not None else "–"
        L.append(f"| {s} | {v['episodes']} | {v['warned_before_onset']} | {ml} | {v['false_alarms']} | {v['non_fog_hours']:.2f} |")
    if "cue_rule" in d:
        c = d["cue_rule"]
        L += ["", "### As a wearable cue (10 s refractory period)", "",
              "A cue (vibration / rhythmic beat) lasts several seconds, so an alarm that re-triggers while a cue is active is",
              f"not a new alarm. Rule: {c['rule']}.", "",
              f"* **{c['warned_before_onset']} of {c['episodes']} freezes ({c['early_warning_rate']:.0%}) cued before onset**, "
              f"median lead {c['median_lead_s']:.1f} s; {c['false_cues_per_hour']:.0f} cues per hour of non-freezing walking.",
              "* For context, this lab protocol provoked about "
              f"{c['episodes'] / max(1e-9, c['non_fog_hours']):.0f} freezes per hour of walking."]
    if "operating_curve" in d:
        L += ["", "### Operating curve (pooled threshold, descriptive)", "",
              "Each row uses one threshold for everyone; a clinician picks the trade-off. Out-of-fold scores only.", "",
              "| window specificity | freezes warned | false cues / h | median lead (s) |", "|---|---|---|---|"]
        for o in d["operating_curve"]:
            L.append(f"| {o['window_specificity']:.2f} | {o['early_warning_rate']:.0%} ({o['warned_before_onset']}/{o['episodes']}) "
                     f"| {o['false_cues_per_hour']:.0f} | {o['median_lead_s']:.1f} |")
    L += ["", "## Limitations", "",
          "* S03 (H&Y 2.5, OFF) received no warnings at the deployed threshold and S07 only 2 of 24 — their pre-freeze",
          "  signal differs from the other participants; per-patient calibration in the twin is meant to address this.",
          "* 10 participants, recorded in one laboratory protocol designed to provoke freezing; prevalence of freezing is",
          "  far higher than in daily life, so false-alarm rates at home will differ.",
          "* S04 and S10 had no freezing episodes; they contribute only to specificity / false alarms.",
          "* Clinical header (age, sex, disease duration, H&Y, medication state) is real but has only 10 distinct values,",
          "  so its contribution cannot be estimated precisely.",
          "* Research prototype, not a medical device. " + r["citation"]]
    (ROOT / "reports" / "fog_forecast_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(**vars(ap.parse_args()))
