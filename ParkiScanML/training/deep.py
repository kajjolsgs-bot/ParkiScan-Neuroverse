"""Deep-learning and differential (multi-class) experiments.

* Daphnet FoG: a 1-D CNN (parkiscan.models.cnn1d, NumPy) trained on RAW ankle-accelerometer windows, evaluated
  leave-one-subject-out (each subject is an independent test set; threshold chosen on the training subjects only).
* PhysioNet gait: 4-class differential (control / Parkinson's / Huntington's / ALS) with grouped CV — a research
  sub-stratification of gait patterns, never a diagnosis.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from parkiscan.models.cnn1d import TinyCNN1D
from training.heldout import wilson, youden_threshold


def daphnet_cnn_loso(X, y, g, epochs=8, seed=0, save_path=None, log=print):
    t0 = time.time()
    P = np.full(len(y), np.nan)
    T = np.full(len(y), np.nan)
    subjects = np.unique(g)
    per = {}
    for s in subjects:
        te = g == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        # threshold from a validation split of the TRAINING subjects (one training subject held out)
        tr_subj = np.unique(g[tr])
        val_s = tr_subj[np.argmax([np.sum((g == t) & (y == 1)) for t in tr_subj])]   # a subject with freezes
        fit_idx = tr & (g != val_s)
        val_idx = g == val_s
        m = TinyCNN1D(seed=seed).fit(X[fit_idx], y[fit_idx], epochs=epochs)
        thr, _ = youden_threshold(y[val_idx], m.predict_proba(X[val_idx])[:, 1])
        m = TinyCNN1D(seed=seed).fit(X[tr], y[tr], epochs=epochs)
        P[te] = m.predict_proba(X[te])[:, 1]
        T[te] = thr
        if len(np.unique(y[te])) == 2:
            per[str(s)] = float(roc_auc_score(y[te], P[te]))
        log(f"[cnn] {s}: done ({time.time() - t0:.0f}s)")
    ok = np.isfinite(P)
    yh = (P[ok] >= T[ok]).astype(int)
    yy = y[ok]
    tn, fp, fn, tp = confusion_matrix(yy, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    prec = tp / (tp + fp) if tp + fp else None
    if save_path is not None:
        TinyCNN1D(seed=seed).fit(X, y, epochs=epochs).save(save_path)
    return {"model": "1-D CNN (NumPy): Conv(3→16,k9)-ReLU-MaxPool4-Conv(16→32,k7)-ReLU-GAP-Dense(2)",
            "input": "raw ankle acceleration, 3 × 256 samples (4 s @ 64 Hz), per-window mean removed",
            "protocol": "leave-one-subject-out; decision threshold tuned on a held-out TRAINING subject",
            "test": {"n_windows": int(ok.sum()), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                     "sensitivity_recall": sens, "sensitivity_95ci": wilson(tp, tp + fn), "specificity": spec,
                     "specificity_95ci": wilson(tn, tn + fp), "precision_ppv": prec,
                     "f1": 2 * prec * sens / (prec + sens) if prec and sens else 0.0,
                     "roc_auc": float(roc_auc_score(yy, P[ok])),
                     "note": "window-level; overlapping windows make intervals optimistic"},
            "per_subject_auc": per, "seconds": round(time.time() - t0, 1)}


def gait_differential(X, y, classes, models: dict, seed=0):
    """One recording per participant, so stratified K-fold on rows is participant-grouped by construction."""
    y = np.asarray(y).astype(int)
    out = {}
    skf = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=seed)
    for name, mdl in models.items():
        pred = np.full(len(y), -1)
        proba = np.zeros((len(y), len(classes)))
        for tr, te in skf.split(X, y):
            f = clone(mdl).fit(X.iloc[tr], y[tr])
            pr = f.predict_proba(X.iloc[te])
            full = np.zeros((len(te), len(classes))); full[:, f.classes_] = pr
            proba[te] = full; pred[te] = full.argmax(1)
        cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
        rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
        try:
            auc_ovr = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        except ValueError:
            auc_ovr = None
        out[name] = {"macro_f1": float(f1_score(y, pred, average="macro")), "balanced_accuracy": float(rec.mean()),
                     "per_class_recall": {c: float(r) for c, r in zip(classes, rec)},
                     "confusion_matrix": cm.tolist(), "macro_roc_auc_ovr": auc_ovr}
    return out
