"""Acoustic voice features and clap-rhythm detection, computed locally with librosa.

Method notes (reported with every result so nobody mistakes these for Praat/MDVP values):
* F0 via probabilistic YIN (librosa.pyin), 65–400 Hz, 10 ms hop.
* Jitter / shimmer are FRAME-LEVEL APPROXIMATIONS (relative mean absolute difference of consecutive
  pyin periods / frame peak amplitudes). They are not cycle-to-cycle Praat measures and must not be
  compared numerically with MDVP:Jitter(%) or Shimmer columns in public datasets.
* HNR from the normalised autocorrelation peak per voiced frame (Boersma-style r/(1-r) in dB).
* Pauses: RMS-energy gaps ≥ 250 ms below an adaptive threshold.
Nothing here is diagnostic.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .common import clean, stats

SR = 16000
HOP = 160  # 10 ms
FRAME = 640  # 40 ms

METHOD_NOTES = [
    "F0: librosa.pyin 65-400 Hz, 10 ms hop",
    "jitter/shimmer: frame-level approximations, NOT Praat cycle-to-cycle measures",
    "HNR: autocorrelation peak per voiced frame",
    "pauses: RMS gaps >= 250 ms below adaptive threshold",
    "research/engagement metrics only; not diagnostic",
]


def load_audio(src) -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    if isinstance(src, (bytes, bytearray)):
        y, sr = sf.read(io.BytesIO(src), always_2d=False)
    else:
        y, sr = sf.read(str(Path(src)), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    return y, sr


def _segments(mask: np.ndarray, hop_s: float, min_len_s: float) -> list[tuple[float, float]]:
    segs, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) * hop_s >= min_len_s:
                segs.append((start * hop_s, i * hop_s))
            start = None
    return segs


def quality(y: np.ndarray, sr: int) -> dict:
    import librosa

    dur = y.size / sr
    clip = float(np.mean(np.abs(y) > 0.99)) if y.size else 1.0
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0] if y.size >= FRAME else np.array([0.0])
    noise = np.percentile(rms, 10) + 1e-9
    sig = np.percentile(rms, 90) + 1e-9
    snr = 20 * np.log10(sig / noise)
    warnings = []
    if dur < 1.0:
        warnings.append("recording shorter than 1 s")
    if clip > 0.01:
        warnings.append("clipping above 1% of samples")
    if snr < 10:
        warnings.append("low signal-to-noise estimate (<10 dB)")
    if sig < 0.005:
        warnings.append("very quiet recording")
    return {"duration_s": dur, "clipping_fraction": clip, "snr_estimate_db": float(snr), "warnings": warnings,
            "valid": dur >= 1.0 and sig >= 0.005}


def voice_features(y: np.ndarray, sr: int = SR) -> dict:
    import librosa

    q = quality(y, sr)
    out: dict = {"quality": q, "method_notes": METHOD_NOTES}
    if not q["valid"]:
        out["features"] = {}
        return clean(out)

    f0, vflag, vprob = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    voiced = np.isfinite(f0) & vflag
    f0v = f0[voiced]
    feats: dict = {"voiced_fraction": float(voiced.mean()) if voiced.size else 0.0}
    if f0v.size >= 3:
        st = 12 * np.log2(f0v / 100.0)
        feats.update({
            "f0_mean_hz": float(f0v.mean()), "f0_std_hz": float(f0v.std()), "f0_min_hz": float(f0v.min()),
            "f0_max_hz": float(f0v.max()), "f0_range_semitones": float(st.max() - st.min()),
            "f0_std_semitones": float(st.std()),
        })
        # frame-level jitter approximation from consecutive voiced periods
        per = 1.0 / f0
        dif = []
        for i in range(1, per.size):
            if voiced[i] and voiced[i - 1]:
                dif.append(abs(per[i] - per[i - 1]))
        feats["jitter_local_approx"] = float(np.mean(dif) / np.mean(1.0 / f0v)) if dif else None
    else:
        feats.update({k: None for k in ["f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_range_semitones",
                                         "f0_std_semitones", "jitter_local_approx"]})

    frames = librosa.util.frame(np.pad(y, (0, max(0, FRAME - y.size))), frame_length=FRAME, hop_length=HOP).T
    n = min(len(frames), voiced.size)
    frames, vmask = frames[:n], voiced[:n]
    peaks = np.abs(frames).max(axis=1)
    pv = peaks[vmask]
    if pv.size >= 3:
        d = [abs(peaks[i] - peaks[i - 1]) for i in range(1, n) if vmask[i] and vmask[i - 1]]
        feats["shimmer_local_approx"] = float(np.mean(d) / (pv.mean() + 1e-9)) if d else None
    else:
        feats["shimmer_local_approx"] = None

    # HNR per voiced frame
    hnrs = []
    for fr, fv in zip(frames[vmask], f0[:n][vmask]):
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, mode="full")[FRAME - 1:]
        if ac[0] <= 0:
            continue
        # unbiased normalisation: the raw autocorrelation shrinks by (N-k)/N with lag k
        ac = ac / ac[0] * FRAME / (FRAME - np.arange(FRAME))
        lo, hi = int(sr / 400), int(sr / 65)
        r = float(ac[lo:hi].max()) if hi < ac.size else float("nan")
        r = min(r, 0.9999)
        if 0 < r:
            hnrs.append(10 * np.log10(r / (1 - r)))
    feats["hnr_db_mean"] = float(np.mean(hnrs)) if hnrs else None

    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-9)
    feats["intensity_db_mean"] = float(db.mean())
    feats["intensity_db_std"] = float(db.std())
    thr = max(np.percentile(db, 20) + 6, db.max() - 35)
    silent = db < thr
    pauses = _segments(silent, HOP / sr, 0.25)
    # ignore leading/trailing silence
    total = y.size / sr
    inner = [p for p in pauses if p[0] > 0.05 and p[1] < total - 0.05]
    feats["pause_count"] = len(inner)
    feats["pause_total_s"] = float(sum(b - a for a, b in inner))
    feats["pause_mean_s"] = float(np.mean([b - a for a, b in inner])) if inner else 0.0
    speech_s = float((~silent).sum() * HOP / sr)
    feats["speech_time_s"] = speech_s
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    feats["onset_rate_per_speech_s"] = float(len(onsets) / speech_s) if speech_s > 0.5 else None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    for i in range(13):
        feats[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i + 1}_std"] = float(mfcc[i].std())
    out["features"] = feats
    return clean(out)


def detect_claps(y: np.ndarray, sr: int = SR, target_bpm: float | None = None) -> dict:
    """Detect hand claps (sharp broadband transients) and summarise rhythm regularity."""
    import librosa
    from scipy import signal as sg

    q = quality(y, sr)
    if y.size < sr // 2:
        return clean({"quality": q, "clap_count": 0, "clap_times_s": []})
    # emphasise the 1-5 kHz band where clap energy concentrates
    b, a = sg.butter(4, [1000, min(5000, sr / 2 - 100)], btype="band", fs=sr)
    yb = sg.filtfilt(b, a, y)
    env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=HOP)
    if env.max() <= 0:
        return clean({"quality": q, "clap_count": 0, "clap_times_s": []})
    env = env / env.max()
    peaks, _ = sg.find_peaks(env, height=0.3, distance=int(0.15 * sr / HOP), prominence=0.2)
    times = peaks * HOP / sr
    ioi = np.diff(times)
    res = {"quality": q, "clap_count": int(times.size), "clap_times_s": times.tolist()}
    res.update(stats(ioi, "ioi_s"))
    if target_bpm and ioi.size:
        period = 60.0 / target_bpm
        res["target_period_s"] = period
        res["tempo_error_pct"] = float((ioi.mean() - period) / period * 100)
        res["mean_abs_period_deviation_s"] = float(np.mean(np.abs(ioi - period)))
    res["method_notes"] = ["1-5 kHz band-pass, onset-strength peak picking (>=150 ms apart)"]
    return clean(res)
