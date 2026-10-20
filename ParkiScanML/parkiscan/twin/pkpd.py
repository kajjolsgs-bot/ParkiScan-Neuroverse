"""Levodopa pharmacokinetics → dopaminergic effect → motor ON/OFF state (the twin's mechanistic core).

One-compartment model with first-order oral absorption and an effect compartment:

    C(t)  = Σ_doses  D·F/V · ka/(ka−ke) · (e^{−ke·τ} − e^{−ka·τ}),       τ = t − t_dose ≥ 0
    dCe/dt = ke0 · (C − Ce)                                              (effect-site lag)
    E_s(t) = Ce^γ / (EC50^γ + Ce^γ)                                       (short-duration response)
    E(t)  = LDR + (1 − LDR) · E_s(t)                                      (0 = fully OFF, 1 = fully ON)

Parameters: elimination half-life 1.5 h (ke = ln2/1.5) and Tmax ≈ 1 h (ka = 2.0 h⁻¹) — in line with reported
immediate-release levodopa/carbidopa values (t½ ≈ 1.5 h; Tmax 0.5–2 h; Hayashi et al., Front. Pharmacol. 2025).
V·F is folded into a dose scale so that a 100 mg dose peaks near C = 1 (arbitrary units). EC50 is individual: it
rises with disease duration, which narrows the therapeutic window and produces wearing-off — the clinical pattern
the twin is meant to anticipate. LDR is the long-duration response: large early in the disease (doses wear off
unnoticed) and fading with progression, when motor state follows plasma levodopa closely. Illustrative model for simulation; not for dosing decisions.
"""
from __future__ import annotations

import math

import numpy as np

KE = math.log(2) / 1.5       # h⁻¹
KA = 2.0                     # h⁻¹  (Tmax ≈ 0.95 h)
KE0 = math.log(2) / 0.5      # h⁻¹  effect-site equilibration (t½ 30 min)
GAMMA = 2.5
DOSE_SCALE = 1.0 / 100.0 / (KA / (KA - KE) * ((KE / KA) ** (KE / (KA - KE)) - (KE / KA) ** (KA / (KA - KE))))
OFF_THRESHOLD = 0.35


def tmax_h() -> float:
    return math.log(KA / KE) / (KA - KE)


def plasma(t_h: np.ndarray, doses: list[tuple[float, float]]) -> np.ndarray:
    """doses: [(time_h, mg)] on the same clock as t_h (hours; may span several days)."""
    t = np.asarray(t_h, float)
    c = np.zeros_like(t)
    for td, mg in doses:
        tau = t - td
        m = tau >= 0
        c[m] += mg * DOSE_SCALE * KA / (KA - KE) * (np.exp(-KE * tau[m]) - np.exp(-KA * tau[m]))
    return c


def effect(t_h: np.ndarray, doses, ec50: float, ldr: float = 0.0, ce0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (plasma C, effect E in 0..1) on the grid t_h (must be uniformly spaced)."""
    t = np.asarray(t_h, float)
    c = plasma(t, doses)
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.25
    ce = np.empty_like(c)
    x = ce0
    a = 1 - math.exp(-KE0 * dt)
    for i, ci in enumerate(c):
        x += a * (ci - x)
        ce[i] = x
    es = ce ** GAMMA / (ec50 ** GAMMA + ce ** GAMMA)
    return c, ldr + (1 - ldr) * es


def schedule(times_h: list[float], mg: float, days: int, start_day: int = 0, shift_h: float = 0.0,
             extra: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    out = []
    for d in range(start_day, start_day + days):
        for t in times_h:
            out.append((d * 24 + t + shift_h, mg))
    for e in extra or []:
        out.append(e)
    return sorted(out)
