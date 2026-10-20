"""Transparent multimodal assessment of one ParkiScan session.

Every game task maps to a domain, its extracted features, the model (if any) that may legitimately use them,
and an output. Nothing is classified unless the evidence supports it:

* A domain is VALID only if its task completed, was not a QA/test run, was not flagged invalid, and the key
  features exist. Otherwise it is INSUFFICIENT / INVALID / NOT_COLLECTED, with the reason.
* Research model outputs are only computed when the SAME feature extractor produced the model's training
  features (spiral traces; ankle-IMU windows) and the model passed its quality gates. They carry the model's
  independent-test sensitivity/specificity and an abstention band ("indeterminate" near the threshold).
* The overall PD / not-PD classification is WITHHELD: no model has been validated on ParkiScan game sessions
  with clinical labels, so any yes/no would be unsupported. The output says exactly what would be needed.
* Sub-stratification (motor phenotype / severity) is likewise withheld; the domain profile is shown instead.
"""
from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------------- task → domain → features
TASK_MAP: dict[str, dict[str, Any]] = {
    "L1_T1_FOLLOW_LANTERN": {
        "domain": "visual_pursuit_attention", "label": "Visual pursuit & attention (webcam gaze estimate, or pointer fallback)",
        # "a|b": either feature satisfies the key (gaze sessions vs pointer-fallback sessions)
        "key": ["gazeErrorMean|pointerErrorMean", "gazeOnTargetPct|pointerOnTargetPct", "checkpointRtMeanSec|checkpointClickRtMeanSec"],
        "features": ["gazeErrorMean", "gazeErrorMedian", "gazeErrorSd", "gazeErrorMeanDeg", "gazeOnTargetPct", "gazeCalibrationRmse",
                     "gazeCalibrationRmseDeg", "gazeTrackingLossFraction", "pursuitGainX", "pursuitGainY", "pursuitLagSec",
                     "gazeJumpCount", "checkpointGazeLatencyMeanSec", "checkpointRtMeanSec",
                     "pointerErrorMean", "pointerErrorSd", "pointerOnTargetPct", "reacquisitionLatencyMeanSec",
                     "checkpointClickRtMeanSec", "checkpointsHit", "checkpointsMissed", "targetLossCount"],
        "model": None, "why_no_model": "no public dataset with this task; webcam gaze is a calibrated head/eye-region estimate, "
                                       "not an infrared eye tracker — tracked against the participant's own repeat sessions",
    },
    "L1_T2_MEMORY_CORE": {
        "domain": "speech_voice_rhythm", "label": "Speech, voice & rhythm (sustained vowel, pa-ta-ka, clapping, sentence repetition)",
        "key": ["voicedDurationSec"],
        "features": ["voicedDurationSec", "pitchSdSemitones", "pauseCount", "speechRateSyllablesPerSec", "snrDb", "clapIoiCv",
                     "clapMeanAbsAsynchronySec", "svc_jitter_local_approx", "svc_shimmer_local_approx", "svc_hnr_db_mean",
                     "svc_f0_std_semitones", "asr_wer", "asr_cer"],
        "model": None, "why_no_model": "public voice datasets ship Praat/MDVP features this pipeline does not reproduce "
                                       "(different extractor, microphones and protocol) — applying those models would be invalid",
    },
    "L1_T3_RECOVER_MEMORY": {
        "domain": "memory", "label": "Visual memory (encode, recall, order)",
        "key": ["recallAccuracy"],
        "features": ["recallAccuracy", "recallHits", "recallOmissions", "falseRecognitions", "recallOrderMatchesDiscovery", "recallDurationSec"],
        "model": None, "why_no_model": "no labelled dataset for this task",
    },
    "L2_T1_TRACE_NEURAL_CIRCUIT": {
        "domain": "fine_motor_drawing", "label": "Fine motor control (spiral tracing)",
        "key": ["radialDeviationRms"],
        "features": ["radialDeviationMean", "radialDeviationRms", "withinToleranceFraction", "tremorAmplitudeProxyRms", "smoothnessLDLJ",
                     "velocityMean", "penLifts", "completionTimeSec"],
        "model": "spiral", "why_no_model": None,
    },
    "L2_T2_CONNECT_MEMORY_NODES": {
        "domain": "processing_speed_sequencing", "label": "Processing speed & sequencing (trail-making style)",
        "key": ["completionTimeSec"],
        "features": ["completionTimeSec", "sequencingAccuracy", "wrongConnections", "interSelectionTimeMeanSec", "interSelectionTimeSdSec", "connectionPrecisionMean"],
        "model": None, "why_no_model": "no labelled dataset for this task",
    },
    "L2_T3_REPAIR_MEMORY_COMPONENT": {
        "domain": "visuospatial_motor_planning", "label": "Visuospatial planning & manipulation",
        "key": ["componentsFitted"],
        "features": ["componentsFitted", "placementErrorMMean", "rotationAccuracyDegMean", "misalignedPlacements", "dragSpeedMean", "completionTimeSec"],
        "model": None, "why_no_model": "no labelled dataset for this task",
    },
    "L2_T4_REACTOR_SIGNAL": {
        "domain": "reaction_inhibition", "label": "Reaction time & response control",
        "key": ["reactionTimeMeanSec"],
        "features": ["reactionTimeMeanSec", "reactionTimeMedianSec", "reactionTimeSdSec", "responseAccuracy", "falseResponses", "missedResponses"],
        "model": None, "why_no_model": "no labelled dataset for this task",
    },
    "L3_T1_MIRROR_MEMORY_AVATAR": {
        "domain": "upper_limb_movement", "label": "Upper-limb movement (webcam pose mirroring)",
        "key": ["movementsCompleted"],
        "features": ["movementsCompleted", "movementsTotal", "trackingConfidenceMean", "missingFrameFraction", "poseRateHz"],
        "model": None, "why_no_model": "webcam pose features are not comparable to any available labelled dataset",
    },
    "L3_T2_GLOWING_MEMORY_PATH_STANDING": {
        "domain": "gait_stepping", "label": "Stepping in place & stopping at signals (webcam)",
        "key": ["steps"],
        "features": ["steps", "cadencePerMin", "stepIntervalSdSec", "alternationRatio", "redLightStopAccuracy", "signalResponseAccuracy", "trackingConfidenceMean"],
        "model": None, "why_no_model": "PhysioNet gait models use force-insole stride intervals over 5 min of walking; webcam stepping in place is not comparable",
    },
    "L3_T2_NEURAL_MEMORY_VEHICLE_SEATED": {
        "domain": "hand_reaction_signals", "label": "Hand-raise reactions to traffic signals (seated, webcam)",
        "key": ["rightHandRaises"],
        "features": ["rightHandRaises", "leftHandRaises", "redLightStopAccuracy", "signalResponseAccuracy", "ambiguousBothHandsEvents", "trackingConfidenceMean"],
        "model": None, "why_no_model": "no labelled dataset for this task",
    },
    "L3_CHECKPOINT_PATIENT_REPORT": {
        "domain": "self_report", "label": "Self-reported symptoms (6 check-in questions; not a validated scale)",
        "key": [], "features": [], "model": None,
        "why_no_model": "the questions are not a validated instrument; answers are summarised, never scored as a diagnosis",
    },
}

SENSOR_DOMAINS = {
    "ankle": {"domain": "freezing_of_gait_imu", "model": "daphnet", "label": "Freezing-of-gait windows (ankle IMU)"},
    "wrist": {"domain": "tremor_imu", "model": None, "label": "Wrist tremor (IMU)"},
}

MIN_VALID_DOMAINS_FOR_ANY_CLASSIFICATION = 6   # documented policy; still not sufficient without a validated model


def _metrics(record: dict) -> dict[str, float]:
    out = {}
    for m in record.get("metrics", []) or []:
        v = m.get("value")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out[m.get("name")] = float(v)
    return out


def _status(record: dict, key: list[str], metrics: dict) -> tuple[str, list[str]]:
    reasons = []
    flags = record.get("qualityFlags", []) or []
    comp = (record.get("completion") or "").lower()
    q = (record.get("quality") or "").lower()
    if any("QA_TEST_RUN" in f for f in flags):
        return "invalid", ["automated QA test run — not participant data"]
    if comp.startswith("skipped") or q == "not_collected":
        return "not_collected", ["task skipped by the participant"]
    if comp in ("complete_no_audio",) or q in ("unavailable",):
        return "insufficient", ["required sensor unavailable (" + (q or comp) + ")"]
    if q == "simulated":
        return "invalid", ["keyboard/simulated input instead of the camera"]
    missing = [k for k in key if not any(a in metrics for a in k.split("|"))]
    if missing:
        reasons.append("missing key features: " + ", ".join(missing))
    if q == "degraded":
        reasons.append("degraded quality: " + ", ".join(f for f in flags if not f.startswith(("language=", "sentence=", "recording="))) [:200])
    if not comp.startswith("complete"):
        reasons.append("task not completed (" + comp + ")")
    if missing or not comp.startswith("complete"):
        return "insufficient", reasons
    return ("valid_with_caveats" if q == "degraded" else "valid"), reasons


def _self_report(rows: list[dict]) -> dict:
    freq = {"Never": 0, "Occasionally": 1, "Sometimes": 2, "Often": 3}
    sev = {"No difficulty": 0, "A little difficulty": 1, "Some difficulty": 2, "A lot of difficulty": 3}
    per = {}
    for r in rows:
        resp = r.get("response", "")
        lvl = freq.get(resp, sev.get(resp))
        per[r.get("domain") or r.get("questionId")] = {
            "response": resp or r.get("status"), "level_0_3": lvl, "language_shown": r.get("shownLanguage"),
        }
    answered = [v["level_0_3"] for v in per.values() if v["level_0_3"] is not None]
    return {"items": per, "answered": len(answered),
            "summary": "participant-reported; unvalidated questions; shown as given, not scored"}


def build_assessment(session: dict, tasks: list[dict], feature_results: list[dict], sensors: dict | None,
                     research_models: dict, research_enabled: bool) -> dict:
    domains: dict[str, dict] = {}
    for t in tasks:
        rec = t.get("record", {})
        tid = rec.get("taskId")
        spec = TASK_MAP.get(tid)
        if spec is None:
            continue
        m = _metrics(rec)
        an = t.get("server_analysis") or {}
        for k, v in (an.get("features") or {}).items():
            if isinstance(v, (int, float)) and v == v:
                m["server_" + k] = float(v)
        st, reasons = _status(rec, spec["key"], m)
        d = domains.setdefault(spec["domain"], {"label": spec["label"], "tasks": [], "status": "not_collected"})
        entry = {"task_id": tid, "subtask_id": rec.get("subtaskId"), "status": st, "reasons": reasons,
                 "features": {k: m[k] for k in spec["features"] if k in m} | {k: v for k, v in m.items() if k.startswith(("svc_", "asr_", "server_")) and k not in spec["features"]},
                 "input_source": rec.get("inputSource"), "duration_s": rec.get("durationSec")}
        if spec["domain"] == "self_report":
            entry["self_report"] = _self_report(an.get("rows", []))
            entry["status"] = "valid" if rec.get("completion", "").startswith("complete") else "insufficient"
        # research model: only where the same extractor produced the training features
        if spec["model"] and an.get("research_score"):
            entry["research_output"] = an["research_score"]
        elif spec["model"]:
            entry["research_output"] = {"status": "not computed",
                                        "reason": "research scores disabled (PARKISCAN_ENABLE_RESEARCH_SCORES=false)" if not research_enabled
                                        else "model unavailable or features missing"}
        else:
            entry["research_output"] = {"status": "no applicable model", "reason": spec["why_no_model"]}
        d["tasks"].append(entry)
    order = {"valid": 4, "valid_with_caveats": 3, "insufficient": 2, "invalid": 1, "not_collected": 0}
    for d in domains.values():
        d["status"] = max((e["status"] for e in d["tasks"]), key=lambda s: order.get(s, 0)) if d["tasks"] else "not_collected"
    for spec in TASK_MAP.values():
        domains.setdefault(spec["domain"], {"label": spec["label"], "tasks": [], "status": "not_collected"})
    # wearable sensors (never invented)
    for place, sd in SENSOR_DOMAINS.items():
        devs = {k: v for k, v in (sensors or {}).items() if (v.get("features") or {}).get("placement") == place}
        if not devs:
            domains[sd["domain"]] = {"label": sd["label"], "status": "not_collected", "tasks": [],
                                     "reasons": ["no " + place + " sensor connected (hardware not yet integrated)"]}
        else:
            sim = any((v.get("features") or {}).get("simulated") for v in devs.values())
            domains[sd["domain"]] = {"label": sd["label"], "status": "invalid" if sim else "valid", "devices": devs,
                                     "reasons": ["SIMULATED sensor data — excluded from assessment"] if sim else []}
    valid = [k for k, d in domains.items() if d["status"] in ("valid", "valid_with_caveats") and k != "self_report"]
    missing = [k for k, d in domains.items() if d["status"] not in ("valid", "valid_with_caveats")]
    research = []
    for d in domains.values():
        for e in d.get("tasks", []):
            ro = e.get("research_output") or {}
            if "value" in ro:
                research.append({"domain_task": e["task_id"], **ro})
    overall = {
        "classification": "WITHHELD",
        "reason": ("No model has been trained and validated on ParkiScan game sessions with clinically confirmed labels. "
                   "A Parkinson's / not-Parkinson's output would therefore be unsupported and could produce false positives "
                   "or false negatives."),
        "valid_domains": valid, "domains_missing_or_invalid": missing,
        "policy": {"min_valid_domains": MIN_VALID_DOMAINS_FOR_ANY_CLASSIFICATION,
                   "also_required": ["a multimodal model trained on ParkiScan sessions from participants with clinician-confirmed "
                                     "diagnoses (and controls)", "independent external validation with reported sensitivity, "
                                     "specificity and calibration", "ethics approval and clinician review of every result"]},
        "research_indicators": research,
        "research_indicator_caution": "experimental, trained on small public datasets recorded with different hardware; "
                                      "never shown to participants and never a diagnosis",
    }
    strat = {
        "motor_phenotype": "WITHHELD — tremor-dominant vs postural-instability/gait subtypes need clinical labels (MDS-UPDRS) to validate",
        "severity": "WITHHELD — severity models from public data (H&Y from insole gait, UPDRS from voice) are not transferable to game tasks; see training report",
        "domain_profile": {k: d["status"] for k, d in domains.items()},
    }
    return {"schema": "parkiscan.assessment.v1", "session_id": session.get("session_id"), "domains": domains,
            "overall": overall, "sub_stratification": strat,
            "data_quality": {"valid_domain_count": len(valid), "total_domains": len(domains)},
            "notice": "Research prototype. Not a medical device. Results describe task performance and data quality only."}
