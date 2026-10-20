"""Digital Twin API + doctor dashboard (mounted by app/main.py).

    GET  /dashboard                    clinician dashboard (static web app)
    GET  /twin/patients                cohort list with current risk and alerts
    GET  /twin/{id}                    full virtual patient: EHR (FHIR-shaped), wearables, PK state, risk forecast, alerts
    GET  /twin/{id}/fhir               the synthetic FHIR R4-shaped bundle
    POST /twin/{id}/whatif             simulate a change (dose timing / extra dose / dose size / sleep) over the next 24 h
    GET  /twin/{id}/replay             real Daphnet accelerometer replay with seconds-ahead freeze risk (Daphnet twins)
    POST /twin/ingest_game             a finished game session from Unity → observations on the player's twin
    GET  /twin/model_card              evidence for both models (metrics, data, limits)
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from parkiscan.twin.engine import ROOT, TwinStore, replay

router = APIRouter()
_store: TwinStore | None = None
_lock = threading.Lock()
DASH = Path(__file__).resolve().parent / "dashboard"


def store() -> TwinStore:
    global _store
    with _lock:
        if _store is None:
            _store = TwinStore()
    return _store


class WhatIf(BaseModel):
    dose_shift_min: float = Field(0.0, ge=-180, le=180, description="shift every future levodopa dose by N minutes")
    extra_dose_clock_h: float | None = Field(None, ge=0, lt=24, description="add one extra dose at this clock time (h)")
    dose_mg: float | None = Field(None, ge=50, le=300, description="change the levodopa dose size (mg)")
    sleep_delta_h: float = Field(0.0, ge=-3, le=3, description="change tonight's sleep by N hours")


class GameSession(BaseModel):
    session_id: str
    finished_at: str | None = None
    avatar: str | None = "grandma"
    display_name: str | None = None
    language: str | None = "en"
    metrics: dict = {}
    tasks: list = []
    flags: list = []


@router.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(DASH / "index.html")


@router.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/dashboard")


@router.get("/twin/patients")
def patients():
    return {"patients": store().list(), "notice": "Research prototype. Synthetic / anonymised data only. Not a medical device."}


@router.get("/twin/model_card")
def model_card():
    out = {}
    for name in ("fog_forecast_report.json", "twin_risk_report.json", "training_report.json"):
        p = ROOT / "reports" / name
        if p.exists():
            try:
                out[name.replace(".json", "")] = json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
    return out


@router.get("/twin/{pid}")
def twin(pid: str):
    t = store().get(pid)
    if t is None:
        raise HTTPException(404, "unknown twin")
    return t.detail()


@router.get("/twin/{pid}/fhir")
def twin_fhir(pid: str):
    t = store().get(pid)
    if t is None:
        raise HTTPException(404, "unknown twin")
    return t.rec["fhir"]


@router.post("/twin/{pid}/whatif")
def twin_whatif(pid: str, req: WhatIf):
    t = store().get(pid)
    if t is None:
        raise HTTPException(404, "unknown twin")
    return t.whatif(req.dose_shift_min, req.extra_dose_clock_h, req.dose_mg, req.sleep_delta_h)


@router.get("/twin/{pid}/replay")
def twin_replay(pid: str, run: str | None = None):
    t = store().get(pid)
    if t is None or t.kind != "daphnet":
        raise HTTPException(404, "replay is available for Daphnet-anchored twins")
    r = replay(t.rec["daphnet_subject"], run)
    if r is None:
        raise HTTPException(404, "forecast model or out-of-fold predictions missing — run python -m training.fog_forecast_train")
    return r


@router.post("/twin/ingest_game")
def ingest_game(req: GameSession):
    return store().ingest_game(req.model_dump())


def _warm():
    try:
        store().list()          # simulate every twin once so the dashboard opens instantly
    except Exception as e:      # never block the service
        import logging
        logging.getLogger("parkiscanml").warning("twin warm-up failed: %s", e)


def mount(app):
    app.include_router(router)
    threading.Thread(target=_warm, daemon=True).start()
    if DASH.exists():
        app.mount("/dashboard/static", StaticFiles(directory=DASH / "static"), name="dashboard-static")
