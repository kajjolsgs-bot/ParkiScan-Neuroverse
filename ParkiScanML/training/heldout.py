"""Independent held-out evaluation, decision thresholds, abstention, and severity sub-stratification.

Held-out protocol (classification datasets):
1. ~25 % of PARTICIPANTS are set aside as an independent test set (stratified, grouped; fixed seed). They are
   never used for model selection, threshold tuning or fitting.
2. On the remaining participants, every candidate model gets participant-grouped out-of-fold predictions; the model
   with the best OOF ROC AUC is chosen, and the decision threshold that maximises Youden's J (sensitivity +
   specificity − 1) on those OOF predictions is fixed. A ±margin band around the threshold is "indeterminate"
   (the pipeline abstains rather than forcing a yes/no).
3. The chosen model is refitted on all training participants and evaluated ONCE on the held-out participants:
   confusion matrix, sensitivity (recall), specificity, precision, F1, balanced accuracy, ROC AUC, Brier score,
   Wilson 95 % intervals for sensitivity/specificity, and metrics with/without the abstention band.
Small public datasets give wide intervals; numbers are research benchmarks, never diagnostic performance.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, confusion_matrix, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold

ABSTAIN_MARGIN = 0.10


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def _agg(y, p, g):
    """Participant-level: mean probability per participant (labels are constant per participant)."""
    df = pd.DataFrame({"g": g, "y": y, "p": p}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return df["y"].to_numpy(), df["p"].to_numpy(), df.index.to_numpy()


def youden_threshold(y, p):
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.round(p, 4)):
        yh = (p >= t).astype(int)
        sens = np.mean(yh[y == 1] == 1) if np.any(y == 1) else 0
        spec = np.mean(yh[y == 0] == 0) if np.any(y == 0) else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, float(best_j)


def binary_report(y, p, thr, margin=ABSTAIN_MARGIN):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = 2 * prec * sens / (prec + sens) if prec and sens else 0.0
    rep = {
        "n": int(len(y)), "threshold": thr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity_recall": sens, "sensitivity_95ci": wilson(tp, tp + fn),
        "specificity": spec, "specificity_95ci": wilson(tn, tn + fp),
        "precision_ppv": prec, "f1": f1,
        "balanced_accuracy": (sens + spec) / 2 if sens is not None and spec is not None else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }
    # abstention: predictions too close to the threshold are reported as "indeterminate"
    decided = np.abs(p - thr) >= margin
    rep["abstention_margin"] = margin
    rep["indeterminate_fraction"] = float(1 - decided.mean())
    if decided.sum() >= 2 and len(np.unique(y[decided])) == 2:
        yd, hd = y[decided], yh[decided]
        tn2, fp2, fn2, tp2 = confusion_matrix(yd, hd, labels=[0, 1]).ravel()
        rep["decided_only"] = {"n": int(decided.sum()), "confusion_matrix": {"tn": int(tn2), "fp": int(fp2), "fn": int(fn2), "tp": int(tp2)},
                               "sensitivity": tp2 / (tp2 + fn2) if tp2 + fn2 else None, "specificity": tn2 / (tn2 + fp2) if tn2 + fp2 else None}
    return rep


def heldout_classification(X, y, g, models: dict, seed: int = 42, test_frac_folds: int = 4):
    y = pd.Series(y).reset_index(drop=True).astype(int)
    g = pd.Series(g).reset_index(drop=True)
    X = X.reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=test_frac_folds, shuffle=True, random_state=seed)
    tr, te = next(outer.split(X, y, g))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr].to_numpy(), g.iloc[tr].to_numpy()
    Xte, yte, gte = X.iloc[te], y.iloc[te].to_numpy(), g.iloc[te].to_numpy()
    n_min = int(min(np.bincount(pd.Series(ytr).groupby(gtr).first().astype(int))))
    inner = StratifiedGroupKFold(n_splits=max(2, min(5, n_min)), shuffle=True, random_state=seed)
    selection = {}
    oofs = {}
    for name, mdl in models.items():
        if name.startswith("dummy"):
            continue
        oof = np.full(len(ytr), np.nan)
        for a, b in inner.split(Xtr, ytr, gtr):
            if len(np.unique(ytr[a])) < 2:
                continue
            oof[b] = clone(mdl).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
        ok = np.isfinite(oof)
        ya, pa, _ = _agg(ytr[ok], oof[ok], gtr[ok])
        selection[name] = float(roc_auc_score(ya, pa)) if len(np.unique(ya)) == 2 else 0.0
        oofs[name] = (ya, pa)
    best = max(selection, key=selection.get)
    thr, j = youden_threshold(*oofs[best])
    final = clone(models[best]).fit(Xtr, ytr)
    pte = final.predict_proba(Xte)[:, 1]
    ya, pa, _ = _agg(yte, pte, gte)
    rep = binary_report(ya, pa, thr)
    return {
        "protocol": f"independent held-out participants (1/{test_frac_folds}, stratified & grouped, seed {seed}); "
                    "model + Youden threshold chosen on training participants only",
        "train_participants": int(len(np.unique(gtr))), "test_participants": int(len(np.unique(gte))),
        "model_selected": best, "inner_cv_auc_by_model": selection, "youden_j_train_oof": j,
        "test": rep,
    }


def loso_event_detection(X, y, g, model, inner_folds: int = 3):
    """Leave-one-subject-out with a threshold tuned on the training subjects only (event detection, e.g. FoG)."""
    y = np.asarray(y).astype(int)
    g = np.asarray(g)
    P = np.full(len(y), np.nan)
    T = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        gi = g[tr]
        oof = np.full(len(tr), np.nan)
        for a, b in GroupKFold(n_splits=min(inner_folds, len(np.unique(gi)))).split(X.iloc[tr], y[tr], gi):
            if len(np.unique(y[tr][a])) < 2:
                continue
            oof[b] = clone(model).fit(X.iloc[tr].iloc[a], y[tr][a]).predict_proba(X.iloc[tr].iloc[b])[:, 1]
        ok = np.isfinite(oof)
        thr, _ = youden_threshold(y[tr][ok], oof[ok])
        P[te] = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        T[te] = thr
    ok = np.isfinite(P)
    yh = (P[ok] >= T[ok]).astype(int)
    yy = y[ok]
    tn, fp, fn, tp = confusion_matrix(yy, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    prec = tp / (tp + fp) if tp + fp else None
    return {"protocol": "leave-one-subject-out; threshold tuned per fold on the other subjects (inner grouped CV)",
            "test": {"n_windows": int(ok.sum()), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                     "sensitivity_recall": sens, "sensitivity_95ci": wilson(tp, tp + fn), "specificity": spec,
                     "specificity_95ci": wilson(tn, tn + fp), "precision_ppv": prec,
                     "f1": 2 * prec * sens / (prec + sens) if prec and sens else 0.0,
                     "roc_auc": float(roc_auc_score(yy, P[ok])),
                     "note": "window-level; consecutive windows overlap so CIs are optimistic"}}


# ---------------------------------------------------------------- severity sub-stratification
def severity_regression(X, sev, groups, models: dict, loo: bool = False):
    sev = np.asarray(sev, float)
    g = np.asarray(groups)
    out = {}
    split = LeaveOneOut() if loo else GroupKFold(n_splits=min(5, len(np.unique(g))))
    for name, mdl in models.items():
        p = np.full(len(sev), np.nan)
        for tr, te in (split.split(X) if loo else split.split(X, sev, g)):
            p[te] = clone(mdl).fit(X.iloc[tr], sev[tr]).predict(X.iloc[te])
        rho = spearmanr(sev, p).correlation if np.nanstd(p) > 0 else float("nan")
        out[name] = {"mae": float(mean_absolute_error(sev, p)), "spearman_rho": float(rho) if rho == rho else None}
    return out


def severity_binary(X, y, g, models: dict, loo: bool = False):
    y = np.asarray(y).astype(int)
    g = np.asarray(g)
    out = {}
    for name, mdl in models.items():
        p = np.full(len(y), np.nan)
        split = LeaveOneOut().split(X) if loo else StratifiedGroupKFold(n_splits=min(5, int(min(np.bincount(pd.Series(y).groupby(g).first().astype(int))))), shuffle=True, random_state=0).split(X, y, g)
        for tr, te in split:
            if len(np.unique(y[tr])) < 2:
                continue
            p[te] = clone(mdl).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        ok = np.isfinite(p)
        ya, pa, _ = _agg(y[ok], p[ok], g[ok])
        thr, _ = youden_threshold(ya, pa)
        out[name] = {"roc_auc": float(roc_auc_score(ya, pa)) if len(np.unique(ya)) == 2 else None,
                     "at_youden_threshold_optimistic": binary_report(ya, pa, thr, margin=0.0)}
    return out
