import numpy as np
import pytest

from parkiscan.features.cognition import cognition_features
from parkiscan.features.common import clean, dfa_alpha
from parkiscan.features.gait import accel_gait_features, freeze_index, stride_features
from parkiscan.features.gaze import gaze_features
from parkiscan.features.pose import pose_features
from parkiscan.features.spiral import spiral_features
from parkiscan.features.voice import SR, detect_claps, voice_features


def _vowel(f0=150.0, seconds=2.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    y = sum(0.3 / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6))
    return (y + rng.normal(0, noise, t.size)).astype(np.float32)


def test_voice_f0_recovered():
    r = voice_features(_vowel(150.0), SR)
    assert r["quality"]["valid"]
    assert abs(r["features"]["f0_mean_hz"] - 150) < 5
    assert r["features"]["voiced_fraction"] > 0.8
    assert r["features"]["hnr_db_mean"] > 10


def test_voice_rejects_short_and_silent():
    assert not voice_features(np.zeros(SR // 2, np.float32), SR)["quality"]["valid"]


def test_pauses_counted():
    v = _vowel(seconds=1.0)
    y = np.concatenate([v, np.zeros(int(0.6 * SR), np.float32), v, np.zeros(int(0.6 * SR), np.float32), v])
    r = voice_features(y, SR)
    assert r["features"]["pause_count"] == 2


def test_claps_detected_with_tempo():
    y = np.zeros(SR * 4, np.float32)
    rng = np.random.default_rng(1)
    for k in range(6):
        i = int((0.3 + k * 0.6) * SR)
        y[i:i + 400] += rng.normal(0, 0.5, 400) * np.exp(-np.arange(400) / 60)
    r = detect_claps(y, SR, target_bpm=100)
    assert r["clap_count"] == 6
    assert abs(r["ioi_s_mean"] - 0.6) < 0.03


def _spiral(n=600, noise=0.0, tremor=0.0, fs=100.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    th = np.linspace(0, 6 * np.pi, n)
    r = 0.1 + 0.05 * th + tremor * np.sin(2 * np.pi * 6 * t)
    return t, r * np.cos(th) + rng.normal(0, noise, n), r * np.sin(th) + rng.normal(0, noise, n)


def test_spiral_smooth_vs_tremor():
    a = spiral_features(*_spiral())
    b = spiral_features(*_spiral(tremor=0.03))
    assert a["valid"] and b["valid"]
    assert abs(a["features"]["loops"] - 3) < 0.1
    assert b["features"]["radial_rmse_norm"] > a["features"]["radial_rmse_norm"]
    assert b["features"]["tremor_band_ratio_4_8hz"] > a["features"]["tremor_band_ratio_4_8hz"]
    assert 5 <= b["features"]["tremor_peak_hz"] <= 7


def test_spiral_too_few_points():
    assert not spiral_features([0, 1], [0, 1], [0, 1])["valid"]


def test_stride_cv_and_asymmetry():
    rng = np.random.default_rng(0)
    l = 1.1 + rng.normal(0, 0.02, 200)
    r = 1.2 + rng.normal(0, 0.02, 200)
    f = stride_features(left_stride=l, right_stride=r)["features"]
    assert 0.01 < f["left_stride_s_cv"] < 0.03
    assert 7 < f["stride_asymmetry_pct"] < 10
    assert "left_stride_s_dfa_alpha" in f


def test_dfa_white_noise_near_half():
    x = np.random.default_rng(3).normal(size=2048)
    assert 0.35 < dfa_alpha(x) < 0.65


def test_freeze_index_higher_for_trembling():
    fs = 64.0
    t = np.arange(int(fs * 8)) / fs
    walk = np.sin(2 * np.pi * 1.8 * t)
    freeze = 0.3 * np.sin(2 * np.pi * 1.8 * t) + np.sin(2 * np.pi * 6 * t)
    assert freeze_index(freeze, fs) > 5 * freeze_index(walk, fs)


def test_accel_cadence():
    fs = 100.0
    t = np.arange(int(fs * 20)) / fs
    ay = -1 + 0.3 * np.sin(2 * np.pi * 1.8 * t)
    f = accel_gait_features(t, np.zeros_like(t), ay, np.zeros_like(t))["features"]
    assert abs(f["cadence_steps_per_min"] - 108) < 6


def test_cognition_summary():
    trials = [{"correct": True, "rt_ms": 800}, {"correct": False, "rt_ms": 1200, "response": "A"},
              {"correct": False, "rt_ms": 1100, "response": "A"}, {"correct": None, "omitted": True},
              {"correct": True, "rt_ms": 700, "span": 4}]
    f = cognition_features(trials)["features"]
    assert f["accuracy"] == 0.5
    assert f["omissions"] == 1
    assert f["perseverative_errors"] == 1
    assert f["max_span_correct"] == 4


def test_gaze_fixations():
    samples, t = [], 0.0
    for px in (0.2, 0.5, 0.8):
        for _ in range(30):
            samples.append({"t": t, "x": px, "y": 0.5, "valid": True})
            t += 1 / 60
    f = gaze_features(samples, targets=[{"t_on": 0.0, "x": 0.5, "y": 0.5}])["features"]
    assert f["fixation_count"] == 3
    assert f["target_hit_rate"] == 1.0


def test_pose_arm_swing_asymmetry():
    frames = []
    for i in range(300):
        t = i / 30
        lm = [[0.5, 0.5, 0, 1.0] for _ in range(33)]
        lm[11] = [0.45, 0.3, 0, 1]; lm[12] = [0.55, 0.3, 0, 1]
        lm[23] = [0.46, 0.6, 0, 1]; lm[24] = [0.54, 0.6, 0, 1]
        lm[15] = [0.45 + 0.04 * np.sin(2 * np.pi * 0.9 * t), 0.5, 0, 1]
        lm[16] = [0.55 + 0.01 * np.sin(2 * np.pi * 0.9 * t), 0.5, 0, 1]
        lm[27] = [0.47, 0.9 + 0.02 * np.sin(2 * np.pi * 0.9 * t), 0, 1]
        lm[28] = [0.53, 0.9 - 0.02 * np.sin(2 * np.pi * 0.9 * t), 0, 1]
        frames.append({"t": t, "landmarks": lm})
    f = pose_features(frames)["features"]
    assert f["left_arm_swing_range"] > 3 * f["right_arm_swing_range"]
    assert f["arm_swing_asymmetry_pct"] > 80


def test_clean_handles_nan():
    assert clean({"a": float("nan"), "b": np.float32(1.5), "c": [np.inf]}) == {"a": None, "b": 1.5, "c": [None]}
