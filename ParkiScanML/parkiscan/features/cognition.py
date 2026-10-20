"""Cognitive task summaries (memory nodes, Memory Core sequences, check-in responses, trail tasks).

Input: a list of trials, each {"correct": bool|None, "rt_ms": float|None, "target": any, "response": any,
"omitted": bool, "span": int}. Unknown fields are ignored.
"""
from __future__ import annotations

import numpy as np

from .common import clean


def cognition_features(trials: list[dict], completion_time_s: float | None = None) -> dict:
    n = len(trials)
    if n == 0:
        return clean({"valid": False, "warnings": ["no trials"], "features": {}})
    correct = [t.get("correct") for t in trials]
    answered = [c for c in correct if c is not None]
    omitted = sum(1 for t in trials if t.get("omitted") or t.get("correct") is None)
    rts = np.asarray([t["rt_ms"] for t in trials if isinstance(t.get("rt_ms"), (int, float)) and t["rt_ms"] > 0], float)
    rts_correct = np.asarray([t["rt_ms"] for t in trials if t.get("correct") and isinstance(t.get("rt_ms"), (int, float))
                              and t["rt_ms"] > 0], float)
    # perseveration: same wrong response repeated on consecutive error trials
    persev = 0
    for a, b in zip(trials, trials[1:]):
        if a.get("correct") is False and b.get("correct") is False and a.get("response") is not None \
                and a.get("response") == b.get("response"):
            persev += 1
    spans = [t["span"] for t in trials if isinstance(t.get("span"), (int, float)) and t.get("correct")]
    f = {
        "trials": n, "answered": len(answered), "omissions": omitted,
        "accuracy": float(np.mean(answered)) if answered else None,
        "errors": int(sum(1 for c in answered if c is False)),
        "rt_median_ms": float(np.median(rts)) if rts.size else None,
        "rt_mean_ms": float(rts.mean()) if rts.size else None,
        "rt_cv": float(rts.std() / rts.mean()) if rts.size > 1 and rts.mean() > 0 else None,
        "rt_correct_median_ms": float(np.median(rts_correct)) if rts_correct.size else None,
        "perseverative_errors": persev,
        "max_span_correct": int(max(spans)) if spans else None,
        "completion_time_s": completion_time_s,
    }
    # learning trend: RT slope across trials (negative = speeding up)
    if rts.size >= 4:
        f["rt_slope_ms_per_trial"] = float(np.polyfit(np.arange(rts.size), rts, 1)[0])
    return clean({"valid": True, "warnings": [], "features": f,
                  "method_notes": ["RTs <= 0 ignored", "not a validated neuropsychological score"]})
