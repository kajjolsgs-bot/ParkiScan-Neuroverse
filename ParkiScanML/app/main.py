"""ParkiScanML — local analysis service for ParkiScan Neuroverse.

Run:  ./run_local.sh            (or: uvicorn app.main:app --host 127.0.0.1 --port 8765)
Docs: http://127.0.0.1:8765/docs

Binds to localhost by default. Nothing is uploaded anywhere; all session data is written under
PARKISCAN_SESSION_DIR. All outputs are engagement / research metrics, never diagnoses.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from parkiscan.datasets.loaders import available
from parkiscan.features.cognition import cognition_features
from parkiscan.features.common import clean
from parkiscan.features.gait import accel_gait_features, accel_window_features, pose_gait_features, stride_features
from parkiscan.features.gaze import gaze_features
from parkiscan.features.pose import pose_features
from parkiscan.features.spiral import spiral_features
from parkiscan.assessment import build_assessment
from parkiscan.features import asr
from parkiscan.features.voice import detect_claps, load_audio, voice_features
from parkiscan.sensors.imu import ImuPacket, append_csv, summarize

from . import schemas as S
from .config import SCHEMA_VERSION, VERSION, get_settings
from .sessions import SessionStore, now, safe_id
from .tts import TtsError, synthesize

log = logging.getLogger("parkiscanml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = get_settings()
store = SessionStore(settings.session_dir)
app = FastAPI(title="ParkiScanML", version=VERSION,
              description="Local feature extraction for ParkiScan Neuroverse. Research/engagement metrics only — "
                          "not a medical device and not diagnostic.")

NOTICE = ("ParkiScan metrics describe how tasks were performed in a game. They are not a diagnosis, are not "
          "clinically validated, and must not be used to make medical decisions.")

# ------------------------------------------------------------------ research models (opt-in)
_models: dict[str, Any] = {}


def _load_models():
    import joblib

    for meta in settings.models_dir.glob("*.json"):
        try:
            m = json.loads(meta.read_text())
            if not m.get("usable_for_research_scores"):
                continue
            _models[m["dataset"]] = {"meta": m, "model": joblib.load(meta.with_suffix(".joblib"))}
        except Exception as e:  # a broken model file must never stop the service
            log.warning("could not load model %s: %s", meta.name, e)


if settings.enable_research_scores:
    _load_models()


def _research_score(dataset: str, rows: list[dict]) -> dict | None:
    """Apply a dataset-trained model to feature rows computed by the SAME extractor. Opt-in only."""
    if not settings.enable_research_scores or dataset not in _models:
        return None
    m = _models[dataset]
    import pandas as pd

    X = pd.DataFrame(rows).reindex(columns=m["meta"]["features"])
    if X.isna().all(axis=None):
        return None
    p = m["model"].predict_proba(X)[:, 1]
    return clean({"dataset": dataset, "model": m["meta"]["model"], "value": float(np.mean(p)),
                  "label": m["meta"]["label"], "quality_flags": m["meta"].get("quality_flags", []),
                  "caution": "Experimental research output from a model trained on a small public dataset recorded "
                             "with different hardware. Not a diagnosis. Do not show to participants as a result."})


# ------------------------------------------------------------------ helpers
def _record(session_id: str | None, kind: str, name: str, result: dict, inputs_meta: dict | None = None) -> dict:
    payload = {"schema": SCHEMA_VERSION, "kind": kind, "name": name, "created_utc": now(), "notice": NOTICE,
               "inputs": inputs_meta or {}, "result": result}
    if session_id:
        try:
            store.save_result(safe_id(session_id), "features", f"{kind}_{name}", payload)
        except ValueError as e:
            raise HTTPException(422, str(e))
    return payload


def _allowed_raw(path: str) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser()
    try:
        rp = p.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    for root in settings.allowed_raw_roots:
        try:
            rp.relative_to(root.expanduser().resolve())
            return rp if rp.suffix.lower() == ".csv" and rp.stat().st_size < 200 * 1024 * 1024 else None
        except (ValueError, FileNotFoundError, OSError):
            continue
    return None


def _read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def analyse_raw(task_id: str, rows: list[dict]) -> dict:
    """Server-side analysis of Unity raw CSVs for the tasks that have a matching extractor."""
    tid = (task_id or "").upper()
    if not rows:
        return {"analysed": False, "reason": "empty raw file"}
    cols = set(rows[0].keys())
    if {"xNorm", "yNorm", "down"} <= cols:  # Level 2 spiral trace
        last = max((_f(r.get("attempt")) or 0) for r in rows)
        pts = [r for r in rows if (_f(r.get("attempt")) or 0) == last and (_f(r.get("down")) or 0) > 0]
        if len(pts) < 10:
            return {"analysed": False, "reason": "fewer than 10 pen-down samples"}
        t = [_f(r["t"]) for r in pts]
        res = spiral_features(t, [_f(r["xNorm"]) for r in pts], [_f(r["yNorm"]) for r in pts])
        sc = _research_score("spiral", [res.get("features", {})])
        return {"analysed": True, "extractor": "spiral", "attempt": int(last), **res,
                **({"research_score": sc} if sc else {})}
    if {"correct", "rtSec"} <= cols or {"correct", "timeSinceLastSec"} <= cols:
        trials = []
        for r in rows:
            c = r.get("correct", "")
            if c == "":
                continue
            rt = _f(r.get("rtSec")) if "rtSec" in cols else _f(r.get("timeSinceLastSec"))
            trials.append({"correct": c.strip().lower() in {"1", "true", "yes"}, "rt_ms": rt * 1000 if rt else None,
                           "response": r.get("nodeSelected") or r.get("actualAction")})
        return {"analysed": True, "extractor": "cognition", **cognition_features(trials)}
    if {"result", "rtSec", "expectedAction"} <= cols:
        trials = [{"correct": r.get("result", "").lower() in {"correct", "hit", "ok"}, "rt_ms": (_f(r.get("rtSec")) or 0) * 1000,
                   "response": r.get("actualAction")} for r in rows if r.get("result")]
        return {"analysed": True, "extractor": "cognition", **cognition_features(trials)}
    if {"questionId", "response"} <= cols:
        # questionnaire answers are stored verbatim; they are NOT scored by this service
        answered = sum(1 for r in rows if (r.get("status") or "").lower() == "answered")
        keep = ("questionId", "domain", "response", "responseIndex", "responseTimeSec", "status", "helpRequested", "shownLanguage", "voicePlayed")
        return {"analysed": True, "extractor": "checkin_responses", "items": len(rows), "answered": answered,
                "rows": [{k: r.get(k) for k in keep} for r in rows],
                "note": "check-in responses are recorded and summarised per domain, not scored as a diagnosis"}
    if {"db", "voiced", "f0Hz"} <= cols:
        f0 = [_f(r["f0Hz"]) for r in rows if (_f(r.get("voiced")) or 0) > 0 and (_f(r.get("f0Hz")) or 0) > 0]
        db = [_f(r["db"]) for r in rows if _f(r.get("db")) is not None]
        st = 12 * np.log2(np.asarray(f0) / 100.0) if f0 else np.array([])
        return clean({"analysed": True, "extractor": "voice_frames",
                      "features": {"frames": len(rows), "voiced_fraction": len(f0) / max(1, len(rows)),
                                   "f0_mean_hz": float(np.mean(f0)) if f0 else None,
                                   "f0_std_semitones": float(st.std()) if st.size else None,
                                   "intensity_db_mean": float(np.mean(db)) if db else None},
                      "note": "frame summaries computed in Unity; no audio was transmitted"})
    return {"analysed": False, "reason": f"no server-side extractor for task {task_id} (Unity metrics kept as-is)"}


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "service": "ParkiScanML", "version": VERSION, "time_utc": now()}


@app.get("/capabilities")
def capabilities():
    return {
        "version": VERSION, "schema": SCHEMA_VERSION, "notice": NOTICE,
        "features": {
            "voice": "upload WAV/FLAC/OGG; F0, approx jitter/shimmer, HNR, intensity, pauses, MFCC; clap rhythm",
            "spiral": "trace kinematics, radial deviation, tremor band, jerk",
            "cognition": "accuracy, RT, omissions, perseveration, span",
            "gait": "stride variability/DFA, accelerometer cadence & freeze index, pose step timing",
            "pose": "arm swing amplitude/asymmetry, wrist tremor band, trunk sway",
            "gaze": "I-VT fixations, saccades, data loss, target hits",
            "imu": "ESP32/MPU-6050 packets (JSON lines); tremor band, cadence, freeze index; simulator flagged",
        },
        "datasets_on_disk": available(settings.data_dir),
        "research_scores_enabled": settings.enable_research_scores,
        "research_models_loaded": sorted(_models),
        "speech_recognition": {"available": asr.available()[0], "detail": asr.available()[1], "scope": "local only",
                               "live_word_follow": "POST /asr/stream (raw PCM16 16 kHz mono), GET /asr/status"},
        "assessment": "GET /session/{id}/assessment — per-domain status, research outputs, overall classification withheld",
        "tts": {"provider": "elevenlabs", "configured": settings.tts_configured,
                "endpoint_enabled": settings.enable_tts_endpoint, "scope": "NIA scripted lines only"},
        "privacy": {"binds": settings.host, "uploads": "none", "participant_audio_to_third_parties": False},
    }


@app.post("/session/start")
def session_start(req: S.SessionStart):
    try:
        data = store.start(safe_id(req.session_id), req.model_dump(exclude={"session_id"}))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"ok": True, "session": data}


@app.post("/session/stop")
def session_stop(req: S.SessionStop):
    try:
        return {"ok": True, "session": store.stop(safe_id(req.session_id))}
    except ValueError as e:
        raise HTTPException(422, str(e))
    except KeyError:
        raise HTTPException(404, "unknown session")


@app.post("/tasks")
def submit_task(req: S.TaskSubmit):
    rec = req.record
    sid = rec.get("sessionId") or ""
    try:
        sid = safe_id(sid)
    except ValueError as e:
        raise HTTPException(422, f"record.sessionId invalid: {e}")
    raw = _allowed_raw(req.raw_csv_path or "")
    analysis = analyse_raw(rec.get("taskId", ""), _read_csv(raw)) if raw else {
        "analysed": False, "reason": "raw CSV not provided, missing, or outside allowed folders"}
    payload = {"schema": SCHEMA_VERSION, "kind": "task", "received_utc": now(), "language": req.language,
               "record": rec, "server_analysis": analysis, "notice": NOTICE}
    store.save_result(sid, "tasks", f"{rec.get('levelId', 'L')}_{rec.get('taskId', 'task')}", payload)
    return {"ok": True, "session_id": sid, "task_id": rec.get("taskId"), "server_analysis": analysis}


@app.post("/features/voice")
async def features_voice(file: UploadFile = File(...), session_id: str | None = Form(None),
                         mode: str = Form("speech"), target_bpm: float | None = Form(None),
                         task_id: str | None = Form(None), language: str | None = Form("en"),
                         expected_text: str | None = Form(None)):
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "file too large")
    try:
        y, sr = load_audio(data)
    except Exception:
        raise HTTPException(415, "unsupported or corrupt audio (use WAV, FLAC or OGG)")
    res = detect_claps(y, sr, target_bpm) if mode == "clap" else voice_features(y, sr)
    if mode == "clap":
        # flatten clap metrics into "features" so every response has the same shape
        res = {**res, "features": {k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}
    if expected_text and mode != "clap":
        # optional local ASR: transcript accuracy against the sentence NIA asked for (never guessed when ASR is absent)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            tf.write(data); tf.flush()
            tr = asr.transcribe(tf.name, language or "en")
        if tr.get("available"):
            tr.update(asr.error_rates(expected_text, tr["text"]))
        res["asr"] = clean(tr)
    # the audio itself is not stored by this endpoint (Unity keeps the WAV in its own local session folder)
    return _record(session_id, "voice" if mode != "clap" else "clap", task_id or file.filename or "audio", res,
                   {"filename": file.filename, "bytes": len(data), "mode": mode, "task_id": task_id, "language": language,
                    "audio_stored": False})


@app.get("/asr/status")
def asr_status(warm: int = 0):
    """Live speech recognition state (Memory Core word follow). ``warm=1`` starts loading the model in the background."""
    if warm:
        asr.warm_up()
    return asr.stream_status()


@app.post("/asr/stream")
async def asr_stream(request: Request, language: str = "en"):
    """Live word follow: body = the audio recorded so far, raw 16-bit little-endian PCM, 16 kHz mono (≤ 30 s).
    Returns the transcript of that audio. Nothing is stored; audio never leaves this computer."""
    import time

    data = await request.body()
    if len(data) > SR_STREAM * 2 * 30:
        raise HTTPException(413, "audio too long (max 30 s)")
    if len(data) % 2:
        data = data[:-1]
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    t0 = time.perf_counter()
    res = await run_in_threadpool(asr.transcribe_pcm, pcm, language or "en")
    res["latency_s"] = round(time.perf_counter() - t0, 3)
    res["audio_s"] = round(pcm.size / SR_STREAM, 2)
    return clean(res)


SR_STREAM = 16000


@app.post("/features/spiral")
def features_spiral(req: S.SpiralRequest):
    p = req.points
    res = spiral_features([q.t for q in p], [q.x for q in p], [q.y for q in p],
                          pressure=[q.pressure or 0 for q in p] if any(q.pressure for q in p) else None,
                          center=req.center, reference=req.reference)
    sc = _research_score("spiral", [res.get("features", {})])
    if sc:
        res["research_score"] = sc
    return _record(req.session_id, "spiral", "trace", res, {"points": len(p)})


@app.post("/features/cognition")
def features_cognition(req: S.CognitionRequest):
    res = cognition_features([t.model_dump() for t in req.trials], req.completion_time_s)
    return _record(req.session_id, "cognition", req.task, res, {"trials": len(req.trials)})


@app.post("/features/gait")
def features_gait(req: S.GaitRequest):
    if req.ax and req.ay and req.az and req.t:
        res = accel_gait_features(req.t, req.ax, req.ay, req.az)
        src = "accelerometer"
    elif req.left_ankle_y and req.right_ankle_y and req.t:
        res = pose_gait_features(req.t, req.left_ankle_y, req.right_ankle_y)
        src = "pose"
    elif req.left_stride_s or req.right_stride_s or req.stride_s:
        res = stride_features(req.left_stride_s, req.right_stride_s, req.stride_s)
        src = "strides"
    else:
        raise HTTPException(422, "provide stride series, t+ax/ay/az, or t+left/right_ankle_y")
    return _record(req.session_id, "gait", src, res, {"source": src})


@app.post("/features/pose")
def features_pose(req: S.PoseRequest):
    res = pose_features([f.model_dump() for f in req.frames])
    return _record(req.session_id, "pose", "frames", res, {"frames": len(req.frames)})


@app.post("/features/gaze")
def features_gaze(req: S.GazeRequest):
    res = gaze_features([s.model_dump() for s in req.samples], req.screen_width_deg, req.units,
                        targets=[t.model_dump() for t in req.targets] if req.targets else None)
    return _record(req.session_id, "gaze", "samples", res, {"samples": len(req.samples)})


@app.post("/sensors/imu")
def sensors_imu(req: S.ImuBatch):
    try:
        sid = safe_id(req.session_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    good, bad = [], 0
    for p in req.packets:
        try:
            good.append(ImuPacket(**p))
        except Exception:
            bad += 1
    store.ensure(sid)
    n = append_csv(store.dir(sid) / "imu.csv", good)
    return {"ok": True, "accepted": n, "rejected": bad, "simulated": any(p.sim for p in good)}


def _imu_summary(sid: str) -> dict | None:
    p = store.dir(sid) / "imu.csv"
    if not p.exists():
        return None
    pk = []
    for r in _read_csv(p):
        try:
            pk.append(ImuPacket(**{**r, "sim": r.get("sim") in {"True", "true", "1"}}))
        except Exception:
            continue
    res = {}
    for dev in sorted({q.dev for q in pk}):
        dp = [q for q in pk if q.dev == dev]
        res[dev] = summarize(dp)
        if dp and dp[0].placement in {"ankle", "shank"} and len(dp) > 256:
            fs = res[dev]["features"].get("rate_hz") or 0
            if fs >= 50:
                # 4 s windows, 1 s hop, same features as the Daphnet model
                ax = np.array([q.ax for q in dp]); ay = np.array([q.ay for q in dp]); az = np.array([q.az for q in dp])
                w, h = int(4 * fs), int(fs)
                rows = [accel_window_features(ax[s:s + w], ay[s:s + w], az[s:s + w], fs) for s in range(0, len(dp) - w, h)]
                sc = _research_score("daphnet", rows)
                if sc:
                    res[dev]["research_score_fog_window_fraction"] = sc
    return clean(res)


@app.get("/session/{session_id}/summary")
def session_summary(session_id: str):
    try:
        sid = safe_id(session_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not store.exists(sid):
        raise HTTPException(404, "unknown session")
    sess = store.read(sid)
    tasks = store.results(sid, "tasks")
    feats = store.results(sid, "features")
    by_level: dict[str, list] = {}
    for t in tasks:
        r = t["record"]
        by_level.setdefault(r.get("levelId", "?"), []).append({
            "task_id": r.get("taskId"), "subtask_id": r.get("subtaskId"), "completion": r.get("completion"),
            "quality": r.get("quality"), "quality_flags": r.get("qualityFlags", []),
            "duration_s": r.get("durationSec"), "input_source": r.get("inputSource"),
            "modalities": r.get("modalities", []), "unity_metrics": r.get("metrics", []),
            "server_analysis": t.get("server_analysis")})
    out = {
        "schema": SCHEMA_VERSION, "session": sess, "generated_utc": now(), "notice": NOTICE,
        "counts": {"tasks": len(tasks), "feature_requests": len(feats)},
        "levels": by_level,
        "feature_results": [{"kind": f["kind"], "name": f["name"], "created_utc": f["created_utc"],
                             "valid": f["result"].get("valid", True), "warnings": f["result"].get("warnings", []),
                             "features": f["result"].get("features", f["result"])} for f in feats],
        "sensors": _imu_summary(sid),
        "research_scores_enabled": settings.enable_research_scores,
    }
    return clean(out)


@app.get("/session/{session_id}/assessment")
def session_assessment(session_id: str):
    """Transparent per-domain assessment with data-quality status; overall classification is withheld (see module doc)."""
    summ = session_summary(session_id)
    sid = safe_id(session_id)
    return clean(build_assessment(summ["session"], store.results(sid, "tasks"), store.results(sid, "features"),
                                  summ.get("sensors"), _models, settings.enable_research_scores))


@app.post("/tts")
def tts(req: S.TtsRequest):
    """Synthesise one scripted NIA line (disabled unless PARKISCAN_ENABLE_TTS=true). Never send participant audio."""
    if not settings.enable_tts_endpoint:
        raise HTTPException(403, "TTS endpoint disabled (set PARKISCAN_ENABLE_TTS=true); use tools/generate_nia_voice.py")
    lang = req.language if req.language in {"en", "ta", "hi", "ml", "te"} else "en"
    try:
        path = synthesize(settings, req.text, settings.session_dir.parent / "tts_cache" / lang)
    except TtsError as e:
        raise HTTPException(503, str(e))
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


# ------------------------------------------------------------------ digital twin + clinician dashboard
from .twin_api import mount as _mount_twin  # noqa: E402

_mount_twin(app)
