"""Synthetic electronic health records (Synthea-style), emitted as a FHIR R4-shaped bundle.

Nothing here describes a real person. Two kinds of record are produced:

* **Daphnet-anchored records** — the real clinical header of the ten Daphnet participants (age, sex, disease
  duration, Hoehn & Yahr, medication state; Bächlin et al. 2010, Table I) extended with SYNTHETIC history,
  medications, labs and genetics so that the twin has a complete record to fuse with the real sensor recordings.
* **Fully synthetic records** — a simulated cohort with demographics and comorbidities chosen for an Indian
  out-patient movement-disorder clinic setting. Prevalences below are simulation parameters, not epidemiological
  estimates.

Every resource carries the tag `SYNTHETIC` (meta.tag) so it can never be confused with patient data.
"""
from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta

import numpy as np

from .daphnet_meta import DAPHNET_TABLE_I

TAG = [{"system": "urn:parkiscan:provenance", "code": "SYNTHETIC", "display": "Synthetic record — not a real patient"}]
TODAY = date(2026, 10, 14)

FIRST_F = ["Lakshmi", "Meena", "Kamala", "Saroja", "Anjali", "Fatima", "Gurpreet", "Leela", "Rekha", "Shanthi", "Mary", "Asha"]
FIRST_M = ["Ramesh", "Suresh", "Abdul", "Venkat", "Harish", "Joseph", "Mohan", "Rajiv", "Iqbal", "Prakash", "Gopal", "Arjun"]
LAST = ["Iyer", "Nair", "Reddy", "Sharma", "Khan", "Patel", "Menon", "Rao", "Singh", "D'Souza", "Pillai", "Gupta", "Das", "Kulkarni"]
CITIES = [("Bengaluru", "Karnataka"), ("Chennai", "Tamil Nadu"), ("Hyderabad", "Telangana"), ("Kochi", "Kerala"),
          ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Delhi", "Delhi"), ("Kolkata", "West Bengal")]
LANG = {"Karnataka": "kn", "Tamil Nadu": "ta", "Telangana": "te", "Kerala": "ml", "Maharashtra": "mr",
        "Delhi": "hi", "West Bengal": "bn"}

# simulation parameters (documented in the model card; NOT epidemiological estimates)
P_HTN, P_T2D, P_DEP, P_OH, P_RBD = 0.35, 0.20, 0.25, 0.20, 0.30
P_GBA, P_LRRK2 = 0.06, 0.01


def _rid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


def _cond(pid, code, system, display, onset: date, clinical="active"):
    return {"resourceType": "Condition", "id": _rid(pid, code), "meta": {"tag": TAG},
            "subject": {"reference": f"Patient/{pid}"},
            "code": {"coding": [{"system": system, "code": code, "display": display}], "text": display},
            "clinicalStatus": {"text": clinical}, "onsetDateTime": onset.isoformat()}


def _obs(pid, code, display, value, unit, when: date, system="urn:parkiscan:obs", interp=None):
    o = {"resourceType": "Observation", "id": _rid(pid, code, when), "meta": {"tag": TAG}, "status": "final",
         "subject": {"reference": f"Patient/{pid}"},
         "code": {"coding": [{"system": system, "code": code, "display": display}], "text": display},
         "effectiveDateTime": when.isoformat()}
    if isinstance(value, str):
        o["valueString"] = value
    else:
        o["valueQuantity"] = {"value": round(float(value), 2), "unit": unit}
    if interp:
        o["interpretation"] = [{"text": interp}]
    return o


SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"


def _regimen(rng, hy: float, dur: float):
    """Levodopa/carbidopa IR schedule. More frequent, larger doses with longer disease duration."""
    n = 3 if dur < 5 else 4 if dur < 12 else 5
    mg = 100 if dur < 8 else 150 if dur < 16 else 200
    first = 7.0 + float(rng.choice([0, 0.5, 1.0]))
    gap = (14.5 if n <= 3 else 13.5) / (n - 1) if n > 1 else 0
    times = [round(first + i * gap, 2) for i in range(n)]
    extra = []
    if dur > 6 and rng.random() < 0.55:
        extra.append({"drug": "Pramipexole ER", "dose": "1.5 mg once daily", "ledd_mg": 150})
    if rng.random() < 0.35:
        extra.append({"drug": "Rasagiline", "dose": "1 mg once daily", "ledd_mg": 100})
    if dur > 10 and rng.random() < 0.4:
        extra.append({"drug": "Entacapone", "dose": "200 mg with each levodopa dose", "ledd_mg": int(0.33 * mg * n)})
    ledd = mg * n + sum(e["ledd_mg"] for e in extra)
    return {"levodopa_mg": mg, "times_h": times, "doses_per_day": n, "adjuncts": extra, "ledd_mg": int(ledd)}


def build_record(pid: str, rng: np.random.Generator, header: dict | None = None, source: str = "synthetic") -> dict:
    """Return {'summary': flat dict used by the models/dashboard, 'fhir': FHIR R4-shaped Bundle}."""
    if header:
        sex, age, dur, hy = header["sex"], int(header["age"]), float(header["disease_duration_y"]), float(header["hoehn_yahr_on"])
        med_state = header.get("tested", "OFF")
    else:
        sex = "F" if rng.random() < 0.38 else "M"
        age = int(np.clip(rng.normal(66, 8), 45, 88))
        dur = float(np.clip(rng.gamma(2.2, 3.6), 0.5, 30))
        hy = float(np.clip(np.round((1.4 + 0.075 * dur + rng.normal(0, 0.5)) * 2) / 2, 1, 5))
        med_state = "ON"
    first = (FIRST_F if sex == "F" else FIRST_M)[int(rng.integers(0, 12))]
    last = LAST[int(rng.integers(0, len(LAST)))]
    city, state = CITIES[int(rng.integers(0, len(CITIES)))]
    birth = date(TODAY.year - age, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
    onset = TODAY - timedelta(days=int(dur * 365.25))
    updrs3 = float(np.clip(8 + 9.5 * hy + rng.normal(0, 5), 4, 90))         # MDS-UPDRS III (OFF), simulated
    fog_trait = 1 / (1 + math.exp(-(-2.2 + 0.55 * (hy - 2) + 0.09 * dur + rng.normal(0, 0.7))))
    if source == "daphnet":
        fog_trait = max(fog_trait, 0.7)        # Daphnet recruited freezers (8 of 10 froze in the lab)
    nfogq = float(np.clip(np.round(24 * fog_trait + rng.normal(0, 2)), 0, 24))
    falls = int(rng.poisson(0.2 + 1.6 * fog_trait))
    moca = float(np.clip(np.round(rng.normal(26.5 - 0.12 * dur - 0.04 * max(0, age - 65), 2.2)), 12, 30))
    gba = rng.random() < P_GBA
    lrrk2 = rng.random() < P_LRRK2
    htn, t2d, dep, oh, rbd = (rng.random() < p for p in (P_HTN, P_T2D, P_DEP, P_OH + 0.01 * dur, P_RBD))
    reg = _regimen(rng, hy, dur)
    sys_bp = float(np.clip(rng.normal(138 if htn else 124, 12), 95, 190))
    bmi = float(np.clip(rng.normal(24.0, 3.5), 16, 38))
    hba1c = float(np.clip(rng.normal(7.4 if t2d else 5.5, 0.6), 4.6, 11))
    vitd = float(np.clip(rng.normal(19, 7), 5, 60))       # ng/mL; low vitamin D is common in simulated cohort
    b12 = float(np.clip(rng.normal(320, 120), 90, 900))    # pg/mL; long-term levodopa may lower B12
    hrv_base = float(np.clip(rng.normal(30 - 0.25 * (age - 60) - (5 if oh else 0), 7), 8, 70))   # nightly RMSSD, ms
    activity = float(np.clip(rng.normal(5200 - 700 * (hy - 2), 1100), 900, 11000))           # steps/day baseline
    sleep_need = float(np.clip(rng.normal(7.0, 0.4), 6, 8.5))
    sleep_base = float(np.clip(rng.normal(6.3 - 0.03 * dur - (0.4 if rbd else 0), 0.6), 3.8, 8.2))
    # individual pharmacodynamics: the therapeutic window narrows with disease duration (wearing-off)
    ec50 = float(np.clip(rng.normal(0.40 + 0.05 * dur, 0.08), 0.25, 2.2))
    ldr = float(np.clip(0.85 * math.exp(-dur / 5.0) + rng.normal(0, 0.05), 0.0, 0.9))
    summary = {
        "id": pid, "source": source, "name": f"{first} {last}", "sex": sex, "age": age, "birth_date": birth.isoformat(),
        "city": city, "state": state, "language": LANG.get(state, "en"),
        "synthetic_id": "SYN-" + pid.upper(), "diagnosis": "Parkinson's disease", "onset_date": onset.isoformat(),
        "disease_duration_y": round(dur, 1), "hoehn_yahr": hy, "updrs3_off": round(updrs3, 0), "nfogq": nfogq,
        "falls_12m": falls, "moca": moca, "gba_carrier": bool(gba), "lrrk2_g2019s": bool(lrrk2),
        "hypertension": bool(htn), "type2_diabetes": bool(t2d), "depression": bool(dep),
        "orthostatic_hypotension": bool(oh), "rem_sleep_behaviour_disorder": bool(rbd),
        "systolic_bp": round(sys_bp), "bmi": round(bmi, 1), "hba1c": round(hba1c, 1), "vitamin_d": round(vitd, 1),
        "b12": round(b12), "regimen": reg, "ledd_mg": reg["ledd_mg"], "recorded_med_state": med_state,
        "hrv_baseline_ms": round(hrv_base, 1), "steps_baseline": round(activity), "sleep_baseline_h": round(sleep_base, 2),
        "sleep_need_h": round(sleep_need, 2), "pd_ec50": round(ec50, 3), "pd_ldr": round(ldr, 3), "fog_trait": round(fog_trait, 3),
        "real_fields": ["sex", "age", "disease_duration_y", "hoehn_yahr", "recorded_med_state"] if source == "daphnet" else [],
    }
    fhir = {"resourceType": "Bundle", "type": "collection", "meta": {"tag": TAG}, "entry": []}
    E = fhir["entry"]
    E.append({"resource": {"resourceType": "Patient", "id": pid, "meta": {"tag": TAG},
                           "identifier": [{"system": "urn:parkiscan:synthetic-id", "value": summary["synthetic_id"]}],
                           "name": [{"text": summary["name"]}], "gender": "female" if sex == "F" else "male",
                           "birthDate": birth.isoformat(), "address": [{"city": city, "state": state, "country": "IN"}],
                           "communication": [{"language": {"text": summary["language"]}}]}})
    E.append({"resource": _cond(pid, "49049000", SNOMED, "Parkinson's disease", onset)})
    for flag, code, disp in ((htn, "38341003", "Hypertensive disorder"), (t2d, "44054006", "Type 2 diabetes mellitus"),
                             (dep, "35489007", "Depressive disorder"), (oh, "28651003", "Orthostatic hypotension"),
                             (rbd, "RBD", "REM sleep behaviour disorder")):
        if flag:
            E.append({"resource": _cond(pid, code, SNOMED if code.isdigit() else "urn:parkiscan:cond", disp,
                                        onset - timedelta(days=int(rng.integers(-900, 2400))))})
    last_visit = TODAY - timedelta(days=int(rng.integers(20, 120)))
    for code, disp, val, unit, system in (
        ("HY", "Hoehn and Yahr stage (ON)", hy, "stage", "urn:parkiscan:obs"),
        ("UPDRS3", "MDS-UPDRS Part III motor score (OFF)", updrs3, "points", "urn:parkiscan:obs"),
        ("NFOGQ", "New Freezing of Gait Questionnaire", nfogq, "points", "urn:parkiscan:obs"),
        ("FALLS12", "Falls in last 12 months", falls, "count", "urn:parkiscan:obs"),
        ("MOCA", "Montreal Cognitive Assessment", moca, "points", "urn:parkiscan:obs"),
        ("8480-6", "Systolic blood pressure", sys_bp, "mm[Hg]", LOINC),
        ("39156-5", "Body mass index", bmi, "kg/m2", LOINC),
        ("4548-4", "Hemoglobin A1c", hba1c, "%", LOINC),
        ("1989-3", "25-hydroxyvitamin D", vitd, "ng/mL", LOINC),
        ("2132-9", "Cobalamin (vitamin B12)", b12, "pg/mL", LOINC),
    ):
        E.append({"resource": _obs(pid, code, disp, val, unit, last_visit, system)})
    E.append({"resource": _obs(pid, "GBA", "GBA variant carrier (simulated genotype)", "positive" if gba else "negative", "", last_visit)})
    E.append({"resource": _obs(pid, "LRRK2", "LRRK2 G2019S (simulated genotype)", "positive" if lrrk2 else "negative", "", last_visit)})
    for i, t in enumerate(reg["times_h"]):
        E.append({"resource": {"resourceType": "MedicationRequest", "id": _rid(pid, "ldopa", i), "meta": {"tag": TAG},
                               "status": "active", "intent": "order", "subject": {"reference": f"Patient/{pid}"},
                               "medicationCodeableConcept": {"text": f"Levodopa/carbidopa IR {reg['levodopa_mg']}/25 mg"},
                               "dosageInstruction": [{"text": f"1 tablet at {int(t):02d}:{int(round((t % 1) * 60)):02d}"}]}})
    for a in reg["adjuncts"]:
        E.append({"resource": {"resourceType": "MedicationRequest", "id": _rid(pid, a["drug"]), "meta": {"tag": TAG},
                               "status": "active", "intent": "order", "subject": {"reference": f"Patient/{pid}"},
                               "medicationCodeableConcept": {"text": a["drug"]}, "dosageInstruction": [{"text": a["dose"]}]}})
    return {"summary": summary, "fhir": fhir}


def daphnet_records(seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [build_record(f"dph-{s.lower()}", rng, header=h, source="daphnet") | {"daphnet_subject": s}
            for s, h in DAPHNET_TABLE_I.items()]


def synthetic_cohort(n: int, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [build_record(f"syn-{i:04d}", rng) for i in range(n)]
