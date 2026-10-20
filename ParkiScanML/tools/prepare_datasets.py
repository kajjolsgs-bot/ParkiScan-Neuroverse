"""Unpack the downloaded dataset archives into data/raw/<name>/ and write data/raw/MANIFEST.json.

    python tools/prepare_datasets.py --zips ~/Downloads
    python tools/prepare_datasets.py --zips ~/Downloads --data data/raw

Matches archive names case-insensitively:
  gait-in-neurodegenerative-disease-database*  -> gaitndd
  parkinsonstelemonitoring*                    -> telemonitoring
  parkinsonspeechdataset*                      -> speech_multi (contains a .rar: needs `brew install unar`)
  parkinsondiseasespiral*                      -> spiral
  daphnetfreezingofgait*                       -> daphnet
  ET_PD_VIS*                                   -> etpdvis
  parkinsons.zip / parkinsons*                 -> oxford_voice   (checked last)
Nothing is downloaded from the internet; only archives already on this computer are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RULES = [
    ("gait-in-neurodegenerative", "gaitndd"), ("telemonitoring", "telemonitoring"),
    ("speechdataset", "speech_multi"), ("spiral", "spiral"), ("daphnet", "daphnet"), ("et_pd_vis", "etpdvis"),
    ("parkinsons", "oxford_voice"),
]
CITATIONS = {
    "oxford_voice": "Little MA et al. (2007) BioMedical Engineering OnLine 6:23 — UCI ML Repository 'Parkinsons'",
    "telemonitoring": "Tsanas A et al. (2010) IEEE TBME 57(4):884-893 — UCI 'Parkinsons Telemonitoring'",
    "speech_multi": "Sakar BE et al. (2013) IEEE JBHI 17(4):828-834 — UCI 'Parkinson Speech Dataset with Multiple Types of Sound Recordings'",
    "spiral": "Isenkul ME et al. (2014) — UCI 'Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet'",
    "gaitndd": "Hausdorff JM et al. (2000) J Appl Physiol 88:2045-2053 — PhysioNet 'Gait in Neurodegenerative Disease'",
    "daphnet": "Bächlin M et al. (2010) IEEE TITB 14(2):436-446 — UCI 'Daphnet Freezing of Gait'",
    "etpdvis": "ET_PD_VIS eye-tracking visualisation dataset on the Trail Making Test (cite per the dataset's own README/licence)",
}


def classify(name: str) -> str | None:
    n = name.lower()
    for key, target in RULES:
        if key in n:
            return target
    return None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_extract(zf: zipfile.ZipFile, dest: Path):
    dest = dest.resolve()
    for m in zf.infolist():
        target = (dest / m.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"unsafe path in archive: {m.filename}")
    zf.extractall(dest)


def extract_nested(dest: Path):
    for z in list(dest.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            safe_extract(zf, z.parent)
    for r in list(dest.rglob("*.rar")):
        tool = shutil.which("unar")
        if tool:
            subprocess.run([tool, "-q", "-f", "-o", str(r.parent), str(r)], check=True)
        else:
            print(f"  ! {r.name} is a RAR archive — install `unar` (brew install unar) and re-run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="folder containing the downloaded .zip files")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    a = ap.parse_args()
    src, data = Path(a.zips).expanduser(), Path(a.data).expanduser()
    data.mkdir(parents=True, exist_ok=True)
    manifest = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for z in sorted(src.glob("*.zip")):
        target = classify(z.name)
        if not target:
            continue
        dest = data / target
        print(f"[prepare] {z.name} -> {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            safe_extract(zf, dest)
        extract_nested(dest)
        files = [p for p in dest.rglob("*") if p.is_file()]
        manifest["datasets"][target] = {"archive": z.name, "archive_sha256": sha256(z), "files": len(files),
                                        "bytes": sum(p.stat().st_size for p in files), "citation": CITATIONS[target]}
    (data / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from parkiscan.datasets.loaders import available
    print("[prepare] found:", json.dumps(available(data), indent=1))


if __name__ == "__main__":
    main()
