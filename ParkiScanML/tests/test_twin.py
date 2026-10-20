"""Digital twin: synthetic EHR, levodopa model, simulator, API and dashboard."""
import math
import os
import tempfile

os.environ.setdefault("PARKISCAN_TWIN_DIR", tempfile.mkdtemp(prefix="twins-"))   # never write test twins into sessions/

import numpy as np
import pytest
from fastapi.testclient import TestClient

from parkiscan.twin import pkpd
from parkiscan.twin.daphnet_meta import DAPHNET_TABLE_I
from parkiscan.twin.ehr import daphnet_records, synthetic_cohort
from parkiscan.twin.sim import hours_label, simulate


def test_levodopa_pk_matches_reported_values():
    assert abs(pkpd.tmax_h() - 0.95) < 0.05                    # Tmax ≈ 1 h
    t = np.arange(0, 12, 0.05)
    c = pkpd.plasma(t, [(0.0, 100.0)])
    assert abs(c.max() - 1.0) < 0.01                           # 100 mg peaks at C = 1 by construction
    i = int(np.argmax(c))
    j = i + int(1.5 / 0.05)
    assert 0.40 < c[j + 20] / c[i + 20] < 0.60                 # elimination half-life ≈ 1.5 h


def test_synthetic_records_are_tagged_and_fhir_shaped():
    for rec in synthetic_cohort(5) + daphnet_records()[:2]:
        b = rec["fhir"]
        assert b["resourceType"] == "Bundle"
        assert all(e["resource"]["meta"]["tag"][0]["code"] == "SYNTHETIC" for e in b["entry"])
        kinds = {e["resource"]["resourceType"] for e in b["entry"]}
        assert {"Patient", "Condition", "Observation", "MedicationRequest"} <= kinds


def test_daphnet_twins_keep_the_real_clinical_header():
    for rec in daphnet_records():
        h = DAPHNET_TABLE_I[rec["daphnet_subject"]]
        s = rec["summary"]
        assert (s["age"], s["sex"], s["hoehn_yahr"], s["disease_duration_y"]) == (h["age"], h["sex"], h["hoehn_yahr_on"], h["disease_duration_y"])


def test_simulator_is_reproducible_and_what_ifs_only_change_what_they_change():
    s = synthetic_cohort(1)[0]["summary"]
    a, b = simulate(s, 3, seed=5), simulate(s, 3, seed=5)
    assert np.array_equal(a["steps"], b["steps"]) and np.array_equal(a["fog_events"], b["fog_events"])
    c = simulate(s, 3, seed=5, dose_shift_h=-0.5)
    assert [n["total_sleep_h"] for n in a["nights"]] == [n["total_sleep_h"] for n in c["nights"]]   # same nights
    d = simulate(s, 3, seed=5, sleep_delta_h=1.0, sleep_from_day=2)
    assert d["nights"][1]["total_sleep_h"] == a["nights"][1]["total_sleep_h"]
    assert d["nights"][2]["total_sleep_h"] >= a["nights"][2]["total_sleep_h"]


def test_two_hour_label():
    ev = np.zeros(20, int); ev[10] = 1
    y = hours_label(np.arange(20) * 0.25, ev, 2.0)
    assert y[1] == 0 and y[2] == 1 and y[9] == 1 and y[10] == 0


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def test_twin_api_roundtrip(client):
    r = client.get("/twin/patients"); assert r.status_code == 200
    ps = r.json()["patients"]; assert len(ps) >= 10
    pid = next(p["id"] for p in ps if p["kind"] == "daphnet")
    d = client.get(f"/twin/{pid}").json()
    assert len(d["series"]["t_h"]) == len(d["series"]["risk"]) and d["provenance"]["ehr"].startswith("REAL clinical header")
    w = client.post(f"/twin/{pid}/whatif", json={"dose_shift_min": -30}).json()
    assert set(w) >= {"baseline", "scenario", "risk_scenario"}
    assert w["scenario"]["off_hours_awake_24h"] <= w["baseline"]["off_hours_awake_24h"] + 1e-9
    assert client.post(f"/twin/{pid}/whatif", json={"dose_shift_min": 999}).status_code == 422
    g = client.post("/twin/ingest_game", json={"session_id": "test-1", "metrics": {"x": {"value": 1.0, "label": "x"}}}).json()
    assert g["twin_id"] == "game-test-1"
    assert client.get("/twin/game-test-1").json()["game"][0]["metrics"]["x"]["value"] == 1.0
    assert client.get("/dashboard").status_code == 200
    assert client.get("/dashboard/static/app.js").status_code == 200


def test_game_twins_survive_a_restart(tmp_path, monkeypatch):
    from parkiscan.twin.engine import TwinStore
    monkeypatch.setenv("PARKISCAN_TWIN_DIR", str(tmp_path))
    a = TwinStore(n_synthetic=1)
    a.ingest_game({"session_id": "restart-1", "avatar": "grandpa", "metrics": {"m": {"value": 2.0, "label": "m"}}})
    b = TwinStore(n_synthetic=1)                                # a new process reading the same folder
    tw = b.get("game-restart-1")
    assert tw is not None and tw.kind == "game" and tw.s["sex"] == "M"
    assert tw.game[0]["metrics"]["m"]["value"] == 2.0
    b.ingest_game({"session_id": "restart-1", "metrics": {}})
    assert len(TwinStore(n_synthetic=1).get("game-restart-1").game) == 2
