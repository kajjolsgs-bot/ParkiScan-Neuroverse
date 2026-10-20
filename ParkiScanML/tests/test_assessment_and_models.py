import numpy as np

from parkiscan.assessment import build_assessment
from parkiscan.features.asr import error_rates, normalize
from parkiscan.models.cnn1d import TinyCNN1D, _softmax
from training.heldout import binary_report, wilson, youden_threshold


def _task(tid, metrics, completion="complete", quality="ok", flags=None, analysis=None, sub=None):
    return {"record": {"taskId": tid, "subtaskId": sub, "completion": completion, "quality": quality,
                       "qualityFlags": flags or [], "metrics": [{"name": k, "value": v} for k, v in metrics.items()]},
            "server_analysis": analysis or {}}


def test_assessment_withholds_classification_and_reports_status():
    tasks = [
        _task("L2_T1_TRACE_NEURAL_CIRCUIT", {"radialDeviationRms": 0.04, "penLifts": 1}),
        _task("L1_T3_RECOVER_MEMORY", {"recallAccuracy": 0.67}),
        _task("L1_T2_MEMORY_CORE", {"voicedDurationSec": 4.2}, quality="degraded", flags=["lowSNR"], sub="A_sustained_vowel"),
        _task("L2_T4_REACTOR_SIGNAL", {}, completion="skipped", quality="not_collected"),
        _task("L1_T1_FOLLOW_LANTERN", {"pointerErrorMean": 0.1}, flags=["QA_TEST_RUN_NOT_PARTICIPANT_DATA"]),
        _task("L3_CHECKPOINT_PATIENT_REPORT", {}, analysis={"rows": [{"questionId": "CHK2_TREMOR", "domain": "Tremor", "response": "Sometimes", "status": "answered"}]}),
    ]
    a = build_assessment({"session_id": "s1"}, tasks, [], None, {}, False)
    d = a["domains"]
    assert a["overall"]["classification"] == "WITHHELD"
    assert d["fine_motor_drawing"]["status"] == "valid"
    assert d["speech_voice_rhythm"]["status"] == "valid_with_caveats"
    assert d["reaction_inhibition"]["status"] == "not_collected"
    assert d["visual_pursuit_attention"]["status"] == "invalid"
    assert d["freezing_of_gait_imu"]["status"] == "not_collected"            # no sensor: never invented
    sr = d["self_report"]["tasks"][0]["self_report"]["items"]["Tremor"]
    assert sr["level_0_3"] == 2
    assert "WITHHELD" in a["sub_stratification"]["motor_phenotype"]
    # the spiral domain names its applicable model but does not compute it while research scores are disabled
    ro = d["fine_motor_drawing"]["tasks"][0]["research_output"]
    assert ro["status"] == "not computed"


def test_simulated_sensor_is_excluded():
    sensors = {"SIM-esp32-01": {"features": {"placement": "ankle", "simulated": True}}}
    a = build_assessment({"session_id": "s"}, [], [], sensors, {}, False)
    assert a["domains"]["freezing_of_gait_imu"]["status"] == "invalid"


def test_error_rates_and_indic_normalisation():
    r = error_rates("The sun is shining.", "the sun is shining")
    assert r["wer"] == 0 and r["cer"] == 0
    r = error_rates("the sun is shining", "the moon is shining")
    assert abs(r["wer"] - 0.25) < 1e-9
    assert normalize("नमस्ते।") == "नमस्ते"


def test_threshold_and_wilson():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([0.1, 0.2, 0.6, 0.55, 0.8, 0.9])
    t, j = youden_threshold(y, p)
    assert 0.55 <= t <= 0.6 and j > 0.5
    rep = binary_report(y, p, t)
    cm = rep["confusion_matrix"]
    assert cm["tp"] + cm["fn"] == 3 and cm["tn"] + cm["fp"] == 3
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi <= 1.0


def test_cnn_gradients_and_learning():
    rng = np.random.default_rng(0)
    m = TinyCNN1D(in_ch=2, c1=3, k1=3, c2=4, k2=3, pool=2)
    for k in m.p:
        m.p[k] = m.p[k].astype(np.float64)
    x = rng.normal(size=(4, 2, 20)); y = np.array([0, 1, 1, 0])

    def loss():
        pr = _softmax(m._forward(x)); return -np.mean(np.log(pr[np.arange(4), y]))
    lg, c = m._forward(x, keep=True)
    g = m._backward(c, (_softmax(lg) - np.eye(2)[y]) / 4)
    for k in ("W1", "W2", "W3"):
        i = np.unravel_index(np.argmax(np.abs(g[k])), g[k].shape); e = 1e-5
        old = m.p[k][i]; m.p[k][i] = old + e; l1 = loss(); m.p[k][i] = old - e; l2 = loss(); m.p[k][i] = old
        assert abs(g[k][i] - (l1 - l2) / (2 * e)) < 1e-6
    fs = 64; t = np.arange(256) / fs
    X = rng.normal(size=(700, 3, 256)).astype(np.float32) * 0.3
    yy = rng.integers(0, 2, 700)
    X[yy == 1, 0] += np.sin(2 * np.pi * 6 * t).astype(np.float32)
    net = TinyCNN1D().fit(X[:600], yy[:600], epochs=8)
    acc = ((net.predict_proba(X[600:])[:, 1] > 0.5) == yy[600:]).mean()
    assert acc > 0.9
