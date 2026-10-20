"""The Digital Twin engine: one virtual patient = static record (EHR) + streaming wearables + game observations +
mechanistic levodopa state + ML risk models. Used by the API (app/twin_api.py) and the doctor dashboard.

Clock: every twin is simulated for 8 days; "now" is day 6, 13:00. The 24 h before "now" are the observed past, the
24 h after are the forecast (dose schedule known, activity = expected, no future sensor data used).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import pkpd
from .daphnet_meta import CITATION, DAPHNET_TABLE_I
from .ehr import build_record, daphnet_records, synthetic_cohort
from .risk import GROUPS, features
from .sim import PER_DAY, STEP_MIN, simulate

ROOT = Path(__file__).resolve().parents[2]
DAYS = 8
NOW_IDX = 6 * PER_DAY + int(13 * 60 / STEP_MIN)       # day 6, 13:00
H24 = PER_DAY
TWIN_COLS = GROUPS["digital_twin (ehr+wearable+pk)"]


def _seed(pid: str) -> int:
    return int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16)


@lru_cache(maxsize=1)
def _risk_model():
    import joblib
    p = ROOT / "models" / "twin_risk.joblib"
    if not p.exists():
        return None, {}
    meta = json.loads((ROOT / "models" / "twin_risk.json").read_text())
    return joblib.load(p), meta


def _fmt_h(h: float) -> str:
    h = h % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


class Twin:
    def __init__(self, record: dict, kind: str):
        self.rec = record
        self.s = record["summary"]
        self.kind = kind                   # daphnet | synthetic | game
        self.id = self.s["id"]
        self.game: list[dict] = []         # game-session observations (digital biomarkers)
        self._cache: dict = {}

    # ---------------------------------------------------------------- simulation + risk
    def scenario(self, dose_shift_h=0.0, extra_doses=None, dose_mg=None, sleep_delta_h=0.0) -> dict:
        key = (round(dose_shift_h, 3), tuple(extra_doses or []), dose_mg, round(sleep_delta_h, 2))
        if key in self._cache:
            return self._cache[key]
        # what-ifs act on the FUTURE only: the past 24 h stay as observed
        sim = simulate(self.s, days=DAYS, seed=_seed(self.id), dose_shift_h=0.0, extra_doses=None, dose_mg=None, sleep_delta_h=0.0)
        if key != (0.0, (), None, 0.0):
            fut = simulate(self.s, days=DAYS, seed=_seed(self.id), dose_shift_h=dose_shift_h, extra_doses=extra_doses, dose_mg=dose_mg,
                           sleep_delta_h=sleep_delta_h, sleep_from_day=NOW_IDX // PER_DAY + 1)   # "tonight" = the next night
            for k in ("plasma", "effect", "off", "hazard", "hr"):
                sim[k] = np.concatenate([sim[k][:NOW_IDX], fut[k][NOW_IDX:]])
            for d in range(int(NOW_IDX // PER_DAY) + 1, DAYS):
                sim["nights"][d] = fut["nights"][d]
            now_h = NOW_IDX * STEP_MIN / 60
            sim["doses"] = [d for d in sim["doses"] if d[0] < now_h] + [d for d in fut["doses"] if d[0] >= now_h]
            for k in ("sleep_debt", "hrv_z"):
                sim[k] = np.concatenate([sim[k][:NOW_IDX], fut[k][NOW_IDX:]])
        # forecast window: expected activity, no future sensor events
        exp_steps = _expected_steps(self.s, sim)
        sim["steps"] = np.concatenate([sim["steps"][:NOW_IDX], exp_steps[NOW_IDX:]])
        inst = sim.get("instability", np.zeros(len(sim["t_h"])))
        sim["instability"] = np.concatenate([inst[:NOW_IDX], np.full(len(inst) - NOW_IDX, float(np.mean(inst[NOW_IDX - H24:NOW_IDX])))])
        df = features(self.s, sim)
        model, meta = _risk_model()
        risk = model.predict_proba(df[TWIN_COLS])[:, 1] if model is not None else np.full(len(df), np.nan)
        asleep = (df["hour"] < 6.5) | (df["hour"] >= 22.0)
        risk = np.where(asleep, 0.0, risk)
        out = {"sim": sim, "df": df, "risk": risk, "thr": float(meta.get("top_decile", {}).get("threshold", 0.25))}
        self._cache[key] = out
        return out

    # ---------------------------------------------------------------- views
    def card(self) -> dict:
        if "card" in self._cache:
            return self._cache["card"]
        sc = self.scenario()
        r = sc["risk"]
        fut = r[NOW_IDX:NOW_IDX + H24]
        peak_i = int(np.argmax(fut))
        out = {"id": self.id, "name": self.s["name"], "age": self.s["age"], "sex": self.s["sex"], "kind": self.kind,
                "hoehn_yahr": self.s["hoehn_yahr"], "disease_duration_y": self.s["disease_duration_y"],
                "city": self.s["city"], "risk_now": round(float(r[NOW_IDX]), 3), "risk_peak_24h": round(float(fut[peak_i]), 3),
                "peak_at": _fmt_h((NOW_IDX + peak_i) * STEP_MIN / 60), "alerts": len(self.alerts(sc, explain=False)),
                "status": _status(float(fut.max()), sc["thr"]), "daphnet_subject": self.rec.get("daphnet_subject"),
                "game_sessions": len(self.game)} | {}
        self._cache["card"] = out
        return out

    def alerts(self, sc=None, explain: bool = True) -> list[dict]:
        sc = sc or self.scenario()
        r, df = sc["risk"], sc["df"]
        awake = ((df["hour"] >= 6.5) & (df["hour"] < 22.0)).to_numpy()
        win = slice(NOW_IDX, NOW_IDX + H24)
        # alert = population threshold AND a peak above this patient's own usual level (personalised excursions)
        thr = max(sc["thr"], 1.3 * float(np.median(r[win][awake[win]]))) if awake[win].any() else sc["thr"]
        out = []
        i = NOW_IDX
        end = NOW_IDX + H24
        while i < end:
            if r[i] >= thr:
                j = i
                while j < end and r[j] >= thr * 0.85:
                    j += 1
                seg = slice(i, j)
                k = i + int(np.argmax(r[seg]))
                drivers = self.drivers(sc, k) if explain else []
                out.append({"from": _fmt_h(i * STEP_MIN / 60), "to": _fmt_h(j * STEP_MIN / 60), "peak": round(float(r[k]), 3),
                            "level": "high" if r[k] >= 1.5 * sc["thr"] else "elevated", "drivers": drivers,
                            "message": _alert_text(drivers), "t_from_h": i * STEP_MIN / 60 - NOW_IDX * STEP_MIN / 60})
                i = j
            i += 1
        return out

    def drivers(self, sc, k: int) -> list[dict]:
        """Local explanation: how much the risk at index k drops if one factor is set to this patient's best state."""
        model, _ = _risk_model()
        if model is None:
            return []
        row = sc["df"].iloc[[k]][TWIN_COLS].copy()
        base = float(model.predict_proba(row)[:, 1][0])
        tests = {
            "Levodopa wearing off (low dopaminergic effect)": {"pk_effect_now": 0.95, "pk_effect_min_2h": 0.95, "pk_off_frac_2h": 0.0, "min_since_dose": 60.0},
            "Short sleep last night": {"sleep_debt_h": 0.0, "sleep_total_h": float(self.s["sleep_need_h"])},
            "Low overnight HRV (stress / poor recovery)": {"rmssd_rel": 1.0},
            "Recent gait instability on the wearable": {"instab_2h": 0.0, "instab_24h": 0.3},
            "Walking a lot right now": {"steps_1h": 150.0, "steps_2h": 300.0},
        }
        res = []
        for name, ch in tests.items():
            alt = row.copy()
            for c, v in ch.items():
                alt[c] = v
            p = float(model.predict_proba(alt)[:, 1][0])
            if base - p > 0.01:
                res.append({"factor": name, "risk_reduction": round(base - p, 3)})
        res.sort(key=lambda d: -d["risk_reduction"])
        return res[:3]

    def detail(self) -> dict:
        sc = self.scenario()
        sim, df, r = sc["sim"], sc["df"], sc["risk"]
        lo, hi = NOW_IDX - H24, NOW_IDX + H24
        t0 = NOW_IDX * STEP_MIN / 60
        # 7-day hourly history (steps, HR, instability) and nights
        hrs = []
        for h in range(0, NOW_IDX // 4):
            sl = slice(h * 4, h * 4 + 4)
            hrs.append({"t_h": round(h - t0, 2), "steps": int(sim["steps"][sl].sum()), "hr": round(float(np.mean(sim["hr"][sl])), 1),
                        "instability": round(float(sim["instability"][sl].sum()), 3)})
        nights = [n | {"t_h": n["day"] * 24 + 7 - t0} for n in sim["nights"][1:NOW_IDX // PER_DAY + 1]]
        doses = [{"t_h": round(d - t0, 2), "mg": mg, "clock": _fmt_h(d)} for d, mg in sim["doses"] if -24 <= d - t0 <= 24]
        series = {"t_h": [round(x - t0, 3) for x in sim["t_h"][lo:hi]], "effect": _r(sim["effect"][lo:hi]), "plasma": _r(sim["plasma"][lo:hi]),
                  "risk": _r(r[lo:hi]), "hazard": _r(sim["hazard"][lo:hi] * 8)}   # hazard ×8 = expected episodes per 2 h
        fog_past = [round(float(sim["t_h"][i] - t0), 2) for i in range(NOW_IDX - 7 * 24 * 4 + 4 * 24, NOW_IDX) if i >= 0 and sim.get("fog_events", np.zeros(1))[i] > 0] if self.kind != "game" else []
        exp_ep = float(sim["hazard"][NOW_IDX:hi].sum())
        off_h = float((sim["effect"][NOW_IDX:hi] < pkpd.OFF_THRESHOLD)[(df["hour"].to_numpy()[NOW_IDX:hi] >= 6.5) & (df["hour"].to_numpy()[NOW_IDX:hi] < 22)].sum() * STEP_MIN / 60)
        s = dict(self.s)
        return {"patient": s, "kind": self.kind, "card": self.card(), "fhir": self.rec["fhir"], "now_clock": _fmt_h(t0),
                "history_hourly": hrs, "nights": nights, "doses": doses, "series": series, "alerts": self.alerts(sc),
                "forecast": {"expected_fog_episodes_24h": round(exp_ep, 2), "off_hours_awake_24h": round(off_h, 2),
                             "risk_threshold": sc["thr"]},
                "fog_events_past_7d_simulated": fog_past, "game": self.game,
                "provenance": _provenance(self)}

    def whatif(self, dose_shift_min=0.0, extra_dose_clock_h=None, dose_mg=None, sleep_delta_h=0.0) -> dict:
        base = self.scenario()
        extra = None
        if extra_dose_clock_h is not None:
            day0 = NOW_IDX // PER_DAY
            t_abs = day0 * 24 + float(extra_dose_clock_h)
            if t_abs < NOW_IDX * STEP_MIN / 60:
                t_abs += 24
            extra = [(round(t_abs, 3), float(dose_mg or self.s["regimen"]["levodopa_mg"]))]
        sc = self.scenario(dose_shift_h=float(dose_shift_min) / 60.0, extra_doses=extra, dose_mg=dose_mg, sleep_delta_h=float(sleep_delta_h))
        sl = slice(NOW_IDX, NOW_IDX + H24)
        hrs = (base["df"]["hour"].to_numpy()[sl] >= 6.5) & (base["df"]["hour"].to_numpy()[sl] < 22)

        def summ(x):
            return {"expected_fog_episodes_24h": round(float(x["sim"]["hazard"][sl].sum()), 2),
                    "off_hours_awake_24h": round(float((x["sim"]["effect"][sl] < pkpd.OFF_THRESHOLD)[hrs].sum() * STEP_MIN / 60), 2),
                    "mean_risk_awake": round(float(x["risk"][sl][hrs].mean()), 4), "peak_risk": round(float(x["risk"][sl].max()), 4),
                    "alert_hours": round(float((x["risk"][sl] >= x["thr"]).sum() * STEP_MIN / 60), 2)}
        t0 = NOW_IDX * STEP_MIN / 60
        return {"baseline": summ(base), "scenario": summ(sc),
                "t_h": [round(x - t0, 3) for x in base["sim"]["t_h"][sl]],
                "risk_baseline": _r(base["risk"][sl]), "risk_scenario": _r(sc["risk"][sl]),
                "effect_baseline": _r(base["sim"]["effect"][sl]), "effect_scenario": _r(sc["sim"]["effect"][sl]),
                "doses_scenario": [{"t_h": round(d - t0, 2), "mg": mg, "clock": _fmt_h(d)} for d, mg in sc["sim"]["doses"] if 0 <= d - t0 <= 24],
                "note": "Simulation of the twin's levodopa model and the 2-h risk model. For discussion with a clinician — not a dosing recommendation."}


def _expected_steps(s: dict, sim: dict) -> np.ndarray:
    from .sim import _circadian
    act = _circadian(sim["hour"]) * (1 - 0.45 * sim["off"]) * (1 - 0.05 * sim["sleep_debt"])
    return s["steps_baseline"] / (PER_DAY * 0.52) * act


def _r(a, nd=4):
    return [round(float(x), nd) for x in np.asarray(a)]


def _status(peak: float, thr: float) -> str:
    return "high" if peak >= 1.5 * thr else "elevated" if peak >= thr else "stable"


def _alert_text(drivers: list[dict]) -> str:
    if not drivers:
        return "Elevated freezing risk"
    m = drivers[0]["factor"]
    return {"Levodopa wearing off (low dopaminergic effect)": "Predicted wearing-off: freezing risk rises before the next dose",
            "Short sleep last night": "Short sleep last night raises today's freezing risk",
            "Low overnight HRV (stress / poor recovery)": "Low overnight HRV — poorer recovery, higher risk",
            "Recent gait instability on the wearable": "Wearable shows recent gait instability",
            "Walking a lot right now": "Long walking bouts while OFF — plan rests and cueing"}.get(m, m)


def _provenance(t: "Twin") -> dict:
    if t.kind == "daphnet":
        return {"ehr": "REAL clinical header (Bächlin et al. 2010, Table I: age, sex, disease duration, H&Y, medication state) + SYNTHETIC extension",
                "wearables": "SIMULATED 7-day streams; the seconds-ahead freeze replay uses this participant's REAL Daphnet accelerometer recording",
                "citation": CITATION}
    if t.kind == "game":
        return {"ehr": "SYNTHETIC record for the game persona", "wearables": "SIMULATED",
                "game": "Observed in ParkiScan Neuroverse sessions (webcam/mouse/microphone game tasks; engagement metrics, not diagnosis)"}
    return {"ehr": "SYNTHETIC (Synthea-style generator)", "wearables": "SIMULATED"}


class TwinStore:
    def __init__(self, n_synthetic: int = 6):
        self.twins: dict[str, Twin] = {}
        for r in daphnet_records():
            self.twins[r["summary"]["id"]] = Twin(r, "daphnet")
        for r in synthetic_cohort(n_synthetic, seed=2026):
            r["summary"]["id"] = r["summary"]["id"].replace("syn-", "demo-")
            for e in r["fhir"]["entry"]:
                if e["resource"]["resourceType"] == "Patient":
                    e["resource"]["id"] = r["summary"]["id"]
            self.twins[r["summary"]["id"]] = Twin(r, "synthetic")
        self.game_dir = Path(os.environ.get("PARKISCAN_TWIN_DIR", ROOT / "sessions" / "twins"))
        self._load_saved_games()

    def _game_twin(self, pid: str, header: dict, name: str) -> "Twin":
        rec = build_record(pid, np.random.default_rng(_seed(pid)), header=header, source="game")
        rec["summary"]["name"] = name
        return Twin(rec, "game")

    def _load_saved_games(self):
        """Game twins survive a service restart: rebuild each from its saved header and observations."""
        if not self.game_dir.is_dir():
            return
        for f in sorted(self.game_dir.glob("game-*.json")):
            try:
                d = json.loads(f.read_text())
                s = d.get("summary", {})
                pid = str(s.get("id") or f.stem)
                header = d.get("header") or {"sex": s.get("sex", "F"), "age": int(s.get("age", 68)),
                                             "disease_duration_y": float(s.get("disease_duration_y", 6)),
                                             "hoehn_yahr_on": float(s.get("hoehn_yahr", 2.5)), "tested": "ON"}
                tw = self._game_twin(pid, header, str(s.get("name") or "You — game session"))
                tw.game = list(d.get("game", []))
                tw.header = header
                self.twins[pid] = tw
            except (OSError, ValueError, KeyError, TypeError):
                continue      # a damaged file never stops the service

    def get(self, pid: str) -> Twin | None:
        return self.twins.get(pid)

    def list(self) -> list[dict]:
        return [t.card() for t in self.twins.values()]

    def ingest_game(self, payload: dict) -> dict:
        """A finished ParkiScan Neuroverse session (Unity) → observations on the player's twin."""
        sid = str(payload.get("session_id") or "local")
        safe = "".join(ch for ch in sid if ch.isalnum() or ch in "-_.")[:48] or "local"
        pid = "game-" + safe
        if pid not in self.twins:
            avatar = str(payload.get("avatar", "grandma"))
            header = {"sex": "F" if avatar == "grandma" else "M", "age": int(payload.get("age", 68 if avatar == "grandma" else 72)),
                      "disease_duration_y": float(payload.get("disease_duration_y", 6)), "hoehn_yahr_on": float(payload.get("hoehn_yahr", 2.5)), "tested": "ON"}
            self.twins[pid] = self._game_twin(pid, header, str(payload.get("display_name") or "You — game session"))
            self.twins[pid].header = header
        tw = self.twins[pid]
        obs = {"received": payload.get("finished_at"), "metrics": payload.get("metrics", {}), "tasks": payload.get("tasks", []),
               "flags": payload.get("flags", []), "language": payload.get("language")}
        tw.game.append(obs)
        tw._cache.pop("card", None)
        try:
            self.game_dir.mkdir(parents=True, exist_ok=True)
            (self.game_dir / f"{pid}.json").write_text(json.dumps({"summary": tw.s, "header": getattr(tw, "header", None), "game": tw.game}, indent=1))
        except OSError:
            pass
        return {"twin_id": pid, "observations": len(tw.game)}


# ---------------------------------------------------------------- real-data replay (Daphnet, seconds ahead)
@lru_cache(maxsize=1)
def _oof() -> pd.DataFrame | None:
    p = ROOT / "reports" / "fog_forecast_oof.csv.gz"
    if not p.exists():
        return None
    return pd.read_csv(p)


def replay(subject: str, run: str | None = None) -> dict | None:
    df = _oof()
    meta_p = ROOT / "models" / "fog_forecast.json"
    if df is None or not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text())
    d = df[df["subject"] == subject]
    if d.empty:
        return None
    runs = sorted(d["run"].unique())
    if run is None:
        # the run with the most freezing episodes makes the clearest demonstration
        run = max(runs, key=lambda r: int(d[d["run"] == r]["fog"].diff().clip(lower=0).sum()))
    g = d[d["run"] == run].sort_values("t_now")
    thr = float(meta["threshold"])
    rp = ROOT / "reports" / "fog_forecast_report.json"
    if rp.exists():
        per = json.loads(rp.read_text()).get("deployed", {}).get("per_subject", {})
        thr = float(per.get(subject, {}).get("threshold", thr))   # the threshold chosen without this subject
    p = g["p"].to_numpy()
    al = (p >= thr) & np.concatenate([[False], p[:-1] >= thr])
    return {"subject": subject, "run": run, "runs": runs, "t_s": _r(g["t_now"], 2), "risk": _r(p, 4), "fog": [int(x) for x in g["fog"]],
            "alarm": [int(x) for x in al], "threshold": thr, "horizon_s": meta.get("horizon_s", 3.0),
            "model": {"configuration": meta.get("configuration"), "auc": meta["metrics"].get("auc"),
                      "early_warning_rate": meta["metrics"].get("early_warning_rate"), "median_lead_s": meta["metrics"].get("median_lead_s"),
                      "false_alarms_per_hour": meta["metrics"].get("false_alarms_per_hour"),
                      "false_cues_per_hour": meta["metrics"].get("false_cues_per_hour")},
            "note": "Out-of-fold (leave-one-subject-out) risk on this participant's real Daphnet recording — the model never saw this person in training."}
