"""Export an offline copy of the clinician dashboard that runs without the Python service.

    python tools/export_static_dashboard.py [--out ../docs/dashboard]
    python tools/export_static_dashboard.py --unity      # data only, for the in-game dashboard (Unity StreamingAssets)

Every API response the dashboard needs is computed in-process (FastAPI TestClient) and written as JSON next to the
page. The what-if simulator is pre-computed on a grid (dose shift × extra dose × sleep change) for every twin; any
other combination needs the live service. The copy can be opened from GitHub Pages or any static host
(browsers block fetch() from file://, so use a local web server to open it from disk: python -m http.server).
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UNITY_OUT = ROOT.parent / "Assets" / "StreamingAssets" / "TwinSnapshot"

GRID = {"shift": [-60, -30, 0, 30, 60], "extra": [14.0, 16.0, 18.0, 20.0], "sleep": [-1.0, 0.0, 1.0]}


def main(out: Path, unity: bool = False):
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    # the in-game snapshot holds only the data (the game draws the dashboard itself) and no game-session twins:
    # in the game, the player's own latest session is read from this computer instead
    data = out if unity else out / "data"
    if out.exists():
        shutil.rmtree(out)
    for d in ("twin", "replay", "whatif"):
        (data / d).mkdir(parents=True, exist_ok=True)

    def get(p):
        r = c.get(p)
        r.raise_for_status()
        return r.json()

    def save(rel, obj):
        (data / rel).write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))

    pats = get("/twin/patients")
    if unity:
        pats = {**pats, "patients": [p for p in pats["patients"] if p.get("kind") != "game"]}
    save("patients.json", pats)
    save("whatif_grid.json", GRID)
    save("model_card.json", get("/twin/model_card"))
    n_wi = 0
    for p in pats["patients"]:
        pid = p["id"]
        det = get(f"/twin/{pid}")
        save(f"twin/{pid}.json", det)
        if det.get("kind") == "daphnet":
            r = get(f"/twin/{pid}/replay")
            save(f"replay/{pid}.json", r)
            for run in r["runs"]:
                save(f"replay/{pid}__{run}.json", get(f"/twin/{pid}/replay?run={run}"))
        sc = {}
        for sh, ex, sl in itertools.product(GRID["shift"], [None] + GRID["extra"], GRID["sleep"]):
            body = {"dose_shift_min": sh, "sleep_delta_h": sl}
            if ex is not None:
                body["extra_dose_clock_h"] = ex
            r = c.post(f"/twin/{pid}/whatif", json=body)
            r.raise_for_status()
            key = f"{sh}|{'' if ex is None else _num(ex)}|{_num(sl)}"
            sc[key] = r.json()
            n_wi += 1
        save(f"whatif/{pid}.json", {"grid": GRID, "scenarios": sc})
        print(f"  {pid}: {len(sc)} what-if scenarios", flush=True)

    if unity:
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"wrote {out} · {len(pats['patients'])} twins · {n_wi} what-if scenarios · {size / 1e6:.1f} MB")
        return
    dash = ROOT / "app" / "dashboard"
    shutil.copytree(dash / "static", out / "static")
    html = (dash / "index.html").read_text()
    html = html.replace('href="/dashboard/static/', 'href="static/').replace('src="/dashboard/static/', 'src="static/')
    cfg = json.dumps({"shift": GRID["shift"], "extra": GRID["extra"], "sleep": GRID["sleep"]})
    html = html.replace('<script src="static/chart.umd.min.js"></script>',
                        f'<script>window.PARKISCAN_STATIC = {cfg};</script>\n<script src="static/chart.umd.min.js"></script>', 1)
    html = html.replace('SYNTHETIC / ANONYMISED DATA ONLY</span>',
                        'SYNTHETIC / ANONYMISED DATA ONLY</span>\n    <span class="pill" title="Pre-computed copy: the what-if offers a grid of scenarios; run the service for any combination">OFFLINE COPY</span>', 1)
    (out / "index.html").write_text(html)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} · {len(pats['patients'])} twins · {n_wi} what-if scenarios · {size / 1e6:.1f} MB")


def _num(x: float) -> str:
    """Match JavaScript's number-to-string (16 → '16', 16.5 → '16.5', -1 → '-1')."""
    return str(int(x)) if float(x).is_integer() else str(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--unity", action="store_true", help="write the in-game snapshot (Assets/StreamingAssets/TwinSnapshot)")
    a = ap.parse_args()
    out = a.out or (UNITY_OUT if a.unity else ROOT.parent / "docs" / "dashboard")
    main(out.resolve(), a.unity)
