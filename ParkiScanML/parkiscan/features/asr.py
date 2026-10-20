"""Optional, fully local speech recognition for the sentence-repetition task.

Uses faster-whisper if it is installed (``./tools/download_models.sh --asr``); otherwise reports itself as
unavailable — no transcript is ever guessed. Audio never leaves the computer.
Accuracy = word error rate (WER) for space-delimited scripts and character error rate (CER) on Unicode
characters (better suited to Tamil, Hindi, Malayalam and Telugu agglutination), against the sentence NIA asked
the participant to repeat. ASR errors inflate both, so these are indicative only.
"""
from __future__ import annotations

import os
import re
import threading
import unicodedata
from pathlib import Path

_MODEL = None
_STREAM = None
_STREAM_NAME = None
_STATUS = None
_LOCK = threading.Lock()
SR = 16000
ROOT = Path(__file__).resolve().parents[2]


def available() -> tuple[bool, str]:
    global _STATUS
    if _STATUS is not None:
        return _STATUS
    try:
        import faster_whisper  # noqa: F401
        _STATUS = (True, "faster-whisper installed")
    except Exception:
        _STATUS = (False, "faster-whisper not installed (run ./tools/download_models.sh --asr)")
    return _STATUS


def _model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        name = os.getenv("PARKISCAN_ASR_MODEL", "small")
        _MODEL = WhisperModel(name, device="cpu", compute_type="int8", download_root=str(ROOT / "asr_models"))
    return _MODEL


def _stream_model():
    """Small, fast model for live word follow (default 'base'); falls back to the main model if it is missing."""
    global _STREAM, _STREAM_NAME
    if _STREAM is None:
        from faster_whisper import WhisperModel
        name = os.getenv("PARKISCAN_ASR_STREAM_MODEL", "base")
        try:
            _STREAM = WhisperModel(name, device="cpu", compute_type="int8", download_root=str(ROOT / "asr_models"))
            _STREAM_NAME = name
        except Exception:
            _STREAM = _model()
            _STREAM_NAME = os.getenv("PARKISCAN_ASR_MODEL", "small")
    return _STREAM


def warm_up() -> None:
    """Load the live model in a background thread (the first request is then fast)."""
    if not available()[0] or _STREAM is not None:
        return

    def _load():
        try:
            with _LOCK:
                _stream_model()
        except Exception:
            pass
    threading.Thread(target=_load, daemon=True).start()


def stream_status() -> dict:
    ok, why = available()
    return {"available": ok, "detail": why, "model": _STREAM_NAME or os.getenv("PARKISCAN_ASR_STREAM_MODEL", "base"),
            "loaded": _STREAM is not None, "scope": "local only — audio never leaves this computer"}


def transcribe_pcm(pcm, language: str = "en") -> dict:
    """Live transcription of 16 kHz mono float32 samples (the audio recorded so far). Greedy decoding, no VAD,
    no context carry-over: one short, fast pass that is repeated as the participant keeps speaking."""
    import numpy as np

    ok, why = available()
    if not ok:
        return {"available": False, "reason": why}
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size < SR // 3 or float(np.sqrt(np.mean(pcm ** 2))) < 1e-4:   # < 0.33 s or digital silence
        return {"available": True, "text": "", "words": []}
    try:
        with _LOCK:
            segs, info = _stream_model().transcribe(
                pcm, language=language if language in {"en", "ta", "hi", "ml", "te"} else None,
                beam_size=1, best_of=1, temperature=0.0, condition_on_previous_text=False,
                without_timestamps=True, vad_filter=False, no_speech_threshold=0.6)
            parts = [s.text.strip() for s in segs if getattr(s, "no_speech_prob", 0.0) < 0.8]
        text = " ".join(p for p in parts if p)
        return {"available": True, "text": text, "words": normalize(text).split(), "model": _STREAM_NAME}
    except Exception as e:
        return {"available": False, "reason": f"ASR failed: {type(e).__name__}"}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"[।॥.,!?;:\"'“”‘’()\-–—…]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _edit(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def error_rates(reference: str, hypothesis: str) -> dict:
    r, h = normalize(reference), normalize(hypothesis)
    rw, hw = r.split(), h.split()
    rc, hc = list(r.replace(" ", "")), list(h.replace(" ", ""))
    return {"wer": _edit(rw, hw) / max(1, len(rw)), "cer": _edit(rc, hc) / max(1, len(rc)),
            "ref_words": len(rw), "hyp_words": len(hw)}


def transcribe(wav_path: str, language: str = "en") -> dict:
    ok, why = available()
    if not ok:
        return {"available": False, "reason": why}
    try:
        segs, info = _model().transcribe(wav_path, language=language if language in {"en", "ta", "hi", "ml", "te"} else None,
                                         beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs)
        return {"available": True, "text": text, "language_detected": info.language,
                "language_probability": float(info.language_probability)}
    except Exception as e:  # model download missing, corrupt audio …
        return {"available": False, "reason": f"ASR failed: {type(e).__name__}"}
