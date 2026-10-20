"""Loaders for the public datasets attached to the project.

Every loader returns (X: DataFrame, y: Series, groups: Series, meta: dict) where `groups` is the
participant identifier, so cross-validation never splits one person's recordings across train and
test. Expected layout under PARKISCAN_DATA_DIR (created by tools/prepare_datasets.py):

  oxford_voice/parkinsons.data                       UCI Parkinson's (Little et al. 2007)
  telemonitoring/parkinsons_updrs.data               UCI Parkinson's Telemonitoring (Tsanas et al. 2010)
  speech_multi/**/train_data.txt, test_data.txt      UCI multiple sound recordings (Sakar et al. 2013)
  spiral/hw_dataset/{control,parkinson}/*.txt        UCI spiral drawings, digitised tablet (Isenkul et al. 2014)
  gaitndd/**/<subject>.ts + subject-description.txt  PhysioNet Gait in Neurodegenerative Disease (Hausdorff et al.)
  daphnet/**/dataset/S##R##.txt                      Daphnet Freezing of Gait (Bächlin et al. 2010)
  etpdvis/**/ET_PD_VIS/{Heatmaps,Scanpaths,Gridded AOI}/TMT_{A,B,C}/{HC,PD}/*  ET_PD_VIS visualisations
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.gait import accel_window_features, stride_features
from ..features.spiral import load_tablet_file, tablet_features


def _find(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def available(root: Path) -> dict:
    root = Path(root)
    checks = {
        "oxford_voice": "parkinsons.data", "telemonitoring": "parkinsons_updrs.data",
        "speech_multi": "train_data.txt", "spiral": "C_*.txt", "gaitndd": "subject-description.txt",
        "daphnet": "S??R0?.txt", "etpdvis": "*-P*.jpg",
    }
    return {k: bool(_find(root / k, pat)) if (root / k).exists() else False for k, pat in checks.items()}


# ---------------------------------------------------------------- voice
def load_oxford_voice(root: Path):
    f = _find(Path(root) / "oxford_voice", "parkinsons.data")[0]
    df = pd.read_csv(f)
    groups = df["name"].str.extract(r"(S\d+)")[0]
    y = df["status"].astype(int)
    X = df.drop(columns=["name", "status"])
    return X, y, groups, {"task": "classification", "label": "PD (1) vs healthy (0)", "unit": "sustained vowel recording",
                          "source": str(f.name)}


def load_telemonitoring(root: Path):
    f = _find(Path(root) / "telemonitoring", "parkinsons_updrs.data")[0]
    df = pd.read_csv(f)
    groups = df["subject#"].astype(str)
    y = df["motor_UPDRS"].astype(float)
    X = df.drop(columns=["subject#", "motor_UPDRS", "total_UPDRS", "test_time", "age", "sex"])
    return X, y, groups, {"task": "regression", "label": "motor UPDRS (all participants have early PD)",
                          "unit": "home voice recording", "source": f.name,
                          "note": "age/sex/test_time removed so the model cannot shortcut via demographics or time"}


SPEECH_COLS = (["subject"] + [f"f{i}" for i in range(1, 27)] + ["updrs", "class"])


def load_speech_multi(root: Path):
    f = _find(Path(root) / "speech_multi", "train_data.txt")[0]
    df = pd.read_csv(f, header=None)
    df.columns = SPEECH_COLS[: df.shape[1]]
    groups = df["subject"].astype(str)
    y = df["class"].astype(int)
    X = df.drop(columns=["subject", "updrs", "class"])
    return X, y, groups, {"task": "classification", "label": "PD (1) vs healthy (0)",
                          "unit": "one of 26 speech recordings per participant", "source": f.name,
                          "note": "UPDRS column removed (it is only defined for PD and would leak the label)"}


# ---------------------------------------------------------------- spiral
def load_spiral(root: Path, test_id: int = 0, include_new_dataset: bool = False):
    base = Path(root) / "spiral"
    rows, ys, gs = [], [], []
    folders = [("hw_dataset/control", 0), ("hw_dataset/parkinson", 1)]
    if include_new_dataset:
        folders.append(("new_dataset/parkinson", 1))
    for sub, label in folders:
        for f in _find(base, f"{sub.split('/')[-1]}/*.txt"):
            if sub.split("/")[0] not in f.parts:
                continue
            tests = load_tablet_file(f)
            if test_id not in tests or len(tests[test_id]) < 50:
                continue
            r = tablet_features(tests[test_id])
            if not r.get("valid"):
                continue
            rows.append(r["features"])
            ys.append(label)
            gs.append(f"{sub.split('/')[0]}:{f.stem}")
    X = pd.DataFrame(rows).drop(columns=["samples", "native_rate_hz"], errors="ignore")
    return X, pd.Series(ys, name="label"), pd.Series(gs, name="participant"), {
        "task": "classification", "label": "PD (1) vs control (0)", "unit": f"spiral test {test_id}, one file per participant",
        "note": "features computed by parkiscan.features.spiral (same code used for in-game traces)"}


# ---------------------------------------------------------------- gait (PhysioNet gaitndd)
def load_gaitndd(root: Path, groups_included=("control", "park")):
    base = Path(root) / "gaitndd"
    desc = _find(base, "subject-description.txt")[0]
    rows, ys, gs = [], [], []
    for ts in sorted(desc.parent.glob("*.ts")):
        m = re.match(r"([a-z]+)(\d+)$", ts.stem)
        if not m or m.group(1) not in groups_included:
            continue
        a = np.loadtxt(ts)
        if a.ndim != 2 or a.shape[0] < 20:
            continue
        a = a[a[:, 0] > 20.0]  # drop the first 20 s (start-up transients), as in the database notes
        r = stride_features(left_stride=a[:, 1], right_stride=a[:, 2], swing_pct=(a[:, 5] + a[:, 6]) / 2,
                            stance_pct=(a[:, 9] + a[:, 10]) / 2, double_support_pct=a[:, 12])
        feats = {k: v for k, v in r["features"].items() if not k.endswith("_n")}
        rows.append(feats)
        ys.append(int(m.group(1) == "park"))
        gs.append(ts.stem)
    X = pd.DataFrame(rows)
    return X, pd.Series(ys), pd.Series(gs), {"task": "classification", "label": "PD (1) vs control (0)",
                                             "unit": "one ~5 min walk per participant (force-sensitive insoles)"}


# ---------------------------------------------------------------- Daphnet FoG
DAPHNET_FS = 64.0


def load_daphnet(root: Path, window_s: float = 4.0, hop_s: float = 1.0, sensor: str = "ankle"):
    base = Path(root) / "daphnet"
    cols = {"ankle": (1, 2, 3), "thigh": (4, 5, 6), "trunk": (7, 8, 9)}[sensor]
    w, h = int(window_s * DAPHNET_FS), int(hop_s * DAPHNET_FS)
    rows, ys, gs = [], [], []
    for f in _find(base, "S??R0?.txt"):
        a = np.loadtxt(f)
        subj = f.stem[:3]
        ann = a[:, 10]
        for s in range(0, a.shape[0] - w, h):
            seg_ann = ann[s:s + w]
            if np.any(seg_ann == 0):
                continue  # outside the experiment protocol
            label = int(np.mean(seg_ann == 2) >= 0.5)
            acc = a[s:s + w, cols[0]:cols[2] + 1] / 1000.0  # mg -> g
            rows.append(accel_window_features(acc[:, 0], acc[:, 1], acc[:, 2], DAPHNET_FS))
            ys.append(label)
            gs.append(subj)
    X = pd.DataFrame(rows)
    return X, pd.Series(ys), pd.Series(gs), {"task": "classification (event detection)",
                                             "label": "freezing-of-gait window (1) vs walking/standing (0)",
                                             "unit": f"{window_s}s window, {hop_s}s hop, {sensor} accelerometer",
                                             "note": "detects FoG episodes in people already diagnosed; not a diagnostic model"}


def load_gaitndd_groups(root: Path):
    """All four groups for differential sub-stratification: control / Parkinson's / Huntington's / ALS."""
    names = {"control": 0, "park": 1, "hunt": 2, "als": 3}
    X, _, g, meta = load_gaitndd(root, groups_included=tuple(names))
    y = pd.Series([names[re.match(r"([a-z]+)", s).group(1)] for s in g])
    return X, y, g, {"task": "multiclass", "classes": ["control", "parkinsons", "huntingtons", "als"],
                     "label": "gait pattern group (differential, research)", "unit": meta["unit"]}


def load_daphnet_raw(root: Path, window_s: float = 4.0, hop_s: float = 1.0, sensor: str = "ankle"):
    """Raw 3-axis windows (N, 3, L) in g for the 1-D CNN, with the same windowing/labels as load_daphnet."""
    base = Path(root) / "daphnet"
    cols = {"ankle": (1, 2, 3), "thigh": (4, 5, 6), "trunk": (7, 8, 9)}[sensor]
    w, h = int(window_s * DAPHNET_FS), int(hop_s * DAPHNET_FS)
    Xs, ys, gs = [], [], []
    for f in _find(base, "S??R0?.txt"):
        a = np.loadtxt(f)
        ann = a[:, 10]
        for s in range(0, a.shape[0] - w, h):
            seg = ann[s:s + w]
            if np.any(seg == 0):
                continue
            Xs.append((a[s:s + w, cols[0]:cols[2] + 1] / 1000.0).T.astype(np.float32))
            ys.append(int(np.mean(seg == 2) >= 0.5))
            gs.append(f.stem[:3])
    return np.stack(Xs), np.asarray(ys), np.asarray(gs)


# ---------------------------------------------------------------- ET_PD_VIS (eye tracking visualisations)
def _image_stats(path: Path) -> dict:
    from PIL import Image

    im = np.asarray(Image.open(path).convert("RGB").resize((256, 256)), float) / 255.0
    gray = im.mean(2)
    sat = im.max(2) - im.min(2)
    colored = sat > 0.25
    p = gray / (gray.sum() + 1e-9)
    hist, _ = np.histogram(sat[colored], bins=16, range=(0, 1)) if colored.any() else (np.zeros(16), None)
    hp = hist / (hist.sum() + 1e-9)
    ys, xs = np.nonzero(colored)
    return {
        "colored_fraction": float(colored.mean()),
        "mean_saturation": float(sat.mean()),
        "gray_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
        "sat_hist_entropy": float(-(hp[hp > 0] * np.log(hp[hp > 0])).sum()),
        "spread_x": float(xs.std() / 256) if xs.size else 0.0,
        "spread_y": float(ys.std() / 256) if ys.size else 0.0,
        "centroid_x": float(xs.mean() / 256) if xs.size else 0.5,
        "centroid_y": float(ys.mean() / 256) if ys.size else 0.5,
    }


def load_etpdvis(root: Path):
    base = Path(root) / "etpdvis"
    vis_root = next((p for p in base.rglob("ET_PD_VIS") if p.is_dir()), None)
    if vis_root is None:
        raise FileNotFoundError("ET_PD_VIS folder not found")
    per: dict[str, dict] = {}
    # "Gridded AOI" is excluded: HC images there have inconsistent canvas sizes (an acquisition/rendering
    # artefact that differs by group and would leak the label).
    for kind in ("Heatmaps", "Scanpaths"):
        for f in sorted((vis_root / kind).rglob("*-P*.*")):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            tmt = f.parent.parent.name  # TMT_A/B/C
            pid = f.stem  # e.g. PD-P10 (HC and PD numbering are separate people)
            d = per.setdefault(pid, {})
            for k, v in _image_stats(f).items():
                d[f"{kind.replace(' ', '')}_{tmt}_{k}"] = v
    pids = sorted(per)
    X = pd.DataFrame([per[p] for p in pids])
    y = pd.Series([int(p.startswith("PD")) for p in pids])
    return X, y, pd.Series(pids), {"task": "classification", "label": "PD (1) vs healthy control (0)",
                                   "unit": "one participant = 6 visualisation images (TMT A/B/C x heatmap + scanpath)",
                                   "note": "dataset ships rendered images, not raw gaze samples; image statistics are a "
                                           "weak proxy and the sample is very small (12 HC / 28 PD). Ink coverage mostly tracks how long the "
                                           "participant looked (task duration), so high scores here are not evidence of a gaze biomarker"}


LOADERS = {
    "oxford_voice": load_oxford_voice,
    "telemonitoring": load_telemonitoring,
    "speech_multi": load_speech_multi,
    "spiral": load_spiral,
    "gaitndd": load_gaitndd,
    "daphnet": load_daphnet,
    "etpdvis": load_etpdvis,
}


# ---------------------------------------------------------------- severity (within PD only — sub-stratification, not diagnosis)
def _gait_pd_with_hy(root: Path):
    base = Path(root) / "gaitndd"
    desc = _find(base, "subject-description.txt")[0]
    hy = {}
    for line in desc.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 8 and parts[1] == "park":
            try:
                hy[parts[0]] = float(parts[7])
            except ValueError:
                pass
    X, y, g, _ = load_gaitndd(root)
    keep = [i for i, s in enumerate(g) if s in hy]
    X = X.iloc[keep].reset_index(drop=True)
    g = g.iloc[keep].reset_index(drop=True)
    return X, np.array([hy[s] for s in g]), g


def load_gait_hy(root: Path):
    X, hy, g = _gait_pd_with_hy(root)
    return X, hy, g, {"task": "regression", "label": "Hoehn & Yahr stage (PD participants only)",
                      "unit": "one walk per participant", "loo": True}


def load_gait_hy_tier(root: Path):
    X, hy, g = _gait_pd_with_hy(root)
    return X, (hy >= 3.0).astype(int), g, {"task": "classification", "label": "H&Y >= 3 (moderate-advanced) vs <= 2.5 (early)",
                                           "unit": "one walk per participant", "loo": True}


def load_telemonitoring_tier(root: Path):
    X, y, g, meta = load_telemonitoring(root)
    return X, (np.asarray(y) >= 20.0).astype(int), g, {"task": "classification", "label": "motor UPDRS >= 20 vs < 20 (early-PD cohort)",
                                                       "unit": "home voice recording"}


def load_speech_updrs(root: Path):
    f = _find(Path(root) / "speech_multi", "train_data.txt")[0]
    df = pd.read_csv(f, header=None)
    df.columns = SPEECH_COLS[: df.shape[1]]
    df = df[df["class"] == 1].reset_index(drop=True)
    X = df.drop(columns=["subject", "updrs", "class"])
    return X, df["updrs"].astype(float).to_numpy(), df["subject"].astype(str), {"task": "regression", "label": "UPDRS (PD participants only)",
                                                                                 "unit": "speech recording"}


SEVERITY_LOADERS = {
    "gaitndd:hoehn_yahr": load_gait_hy,
    "gaitndd:early_vs_advanced": load_gait_hy_tier,
    "telemonitoring:motor_updrs_tier": load_telemonitoring_tier,
    "speech_multi:updrs": load_speech_updrs,
}
