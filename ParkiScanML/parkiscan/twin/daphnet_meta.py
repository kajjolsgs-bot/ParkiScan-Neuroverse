"""Real clinical header of the ten Daphnet participants.

Transcribed from Table I of: Bächlin M, Plotnik M, Roggen D, Maidan I, Hausdorff JM, Giladi N, Tröster G.
"Wearable Assistant for Parkinson's Disease Patients With the Freezing of Gait Symptom." IEEE Transactions on
Information Technology in Biomedicine 14(2):436-446, 2010 (preprint shipped in the dataset's doc/ folder).

"tested" is the medication state during recording: OFF = >12 h after the last anti-parkinsonian dose; ON = the two
patients with frequent FoG in the ON state were not asked to withhold medication.
"""
from __future__ import annotations

DAPHNET_TABLE_I = {
    "S01": {"sex": "M", "age": 66, "disease_duration_y": 16, "hoehn_yahr_on": 3.0, "tested": "OFF"},
    "S02": {"sex": "M", "age": 67, "disease_duration_y": 7, "hoehn_yahr_on": 2.0, "tested": "ON"},
    "S03": {"sex": "M", "age": 59, "disease_duration_y": 30, "hoehn_yahr_on": 2.5, "tested": "OFF"},
    "S04": {"sex": "M", "age": 62, "disease_duration_y": 3, "hoehn_yahr_on": 3.0, "tested": "OFF"},
    "S05": {"sex": "M", "age": 75, "disease_duration_y": 6, "hoehn_yahr_on": 2.0, "tested": "OFF"},
    "S06": {"sex": "F", "age": 63, "disease_duration_y": 22, "hoehn_yahr_on": 2.0, "tested": "OFF"},
    "S07": {"sex": "M", "age": 66, "disease_duration_y": 2, "hoehn_yahr_on": 2.5, "tested": "OFF"},
    "S08": {"sex": "F", "age": 68, "disease_duration_y": 18, "hoehn_yahr_on": 4.0, "tested": "ON"},
    "S09": {"sex": "M", "age": 73, "disease_duration_y": 9, "hoehn_yahr_on": 2.0, "tested": "OFF"},
    "S10": {"sex": "F", "age": 65, "disease_duration_y": 24, "hoehn_yahr_on": 3.0, "tested": "OFF"},
}

CITATION = ("Bächlin M, Plotnik M, Roggen D, et al. Wearable Assistant for Parkinson's Disease Patients With the "
            "Freezing of Gait Symptom. IEEE Trans Inf Technol Biomed. 2010;14(2):436-446. Dataset: UCI Machine "
            "Learning Repository, Daphnet Freezing of Gait (CC BY 4.0).")


def ehr_features(subject: str) -> dict:
    """Numeric static features for model fusion (real values from Table I)."""
    m = DAPHNET_TABLE_I[subject]
    return {
        "ehr_age": float(m["age"]),
        "ehr_female": 1.0 if m["sex"] == "F" else 0.0,
        "ehr_disease_duration_y": float(m["disease_duration_y"]),
        "ehr_hoehn_yahr": float(m["hoehn_yahr_on"]),
        "ehr_med_on": 1.0 if m["tested"] == "ON" else 0.0,
    }
