# ParkiScan Neuroverse

A Unity 6 game in which NIA guides an older adult through three lands of short, playful tasks. Each task records
how it was performed — voice, rhythm, drawing, memory, reaction time, movement seen by the webcam — on this
computer only. A local Python service (`ParkiScanML/`) extracts features, keeps a per-session record and produces
a transparent, research-only assessment.

> **Research prototype — not a medical device.** No result is a diagnosis. The overall Parkinson's classification
> is deliberately withheld until a model is validated on ParkiScan sessions with clinician-confirmed labels
> (see [`docs/ASSESSMENT_MAPPING.md`](docs/ASSESSMENT_MAPPING.md)).

## Repository layout

```
Assets/ParkiScan/            game: Scripts/, Art/ (models, fonts), Audio/ (music + SFX), Resources/ (localisation, config), Scenes/Demo.unity
  Scripts/Demo/              DemoDirector_* — the whole game flow (entry, language, levels 1–3, India, navigation, seating)
  Scripts/Audio/             AudioManager (music states, ducking), NiaVoice (NIA speech in 5 languages)
  Scripts/Tracking/          BlazePose webcam pose tracker + pose features (hand raises, steps, arm elevation)
  Scripts/Sensors/           SensorHub — interfaces for future wrist/ankle IMUs (no hardware yet)
  Scripts/Telemetry/         TaskRecorder / TelemetryManager (local JSON + CSV), ParkiScanService (client of the local API)
  Scripts/Editor/            DemoSceneBuilder (menu ParkiScan → Build Demo Scene), QA playtests (menu ParkiScan → QA)
ParkiScanML/                 FastAPI service, feature extractors, dataset loaders, training, ESP32 firmware, tests
docs/                        task → feature → model → output mapping
Packages/, ProjectSettings/  Unity project configuration
```

## Requirements

* Unity **6000.3.25f1** (6.3 LTS) with URP (packages resolve from `Packages/manifest.json`)
* macOS (tested) — NIA's voice uses the system speech voices; on other platforms lines stay text-only unless
  pre-generated clips exist
* Python **3.10–3.13** for the service, a webcam and a microphone for the full experience
* **Git LFS** for the large binary assets (`git lfs install` once)

## Run the game

1. Open the folder in Unity Hub (Unity 6000.3.25f1). First import takes a while.
2. Open `Assets/ParkiScan/Scenes/Demo.unity` and press **Play**.
3. If the scene is missing or out of date: menu **ParkiScan → Build Demo Scene**.

Flow: planet → title (PLAY) → **language** → traveller → bedroom → wardrobe → the Three Lands → any level.
**MENU** (top-left, or Esc) moves freely between levels: back to the previous level, restart, the Three Lands,
traveller & outfit, or language. Completed task results are never erased by navigating.

**Clinician dashboard inside the game:** press **F2** at any time, or CLINICIAN DASHBOARD on the title screen,
the Three Lands or in MENU; at the end of Level 3 the twin report opens it on the player's own twin. The game is
paused behind it; Esc or CLOSE resumes. With the analysis service running (below) the pill top-right reads
**LIVE**; without it the dashboard opens the bundled **OFFLINE SNAPSHOT** (`Assets/StreamingAssets/TwinSnapshot`,
refresh with `python ParkiScanML/tools/export_static_dashboard.py --unity`). Automated check: menu
**ParkiScan → QA → Run Dashboard check**.

### Languages and NIA's voice

English, Tamil, Hindi, Malayalam and Telugu. On macOS NIA speaks every line, prompt and sentence with the
installed system voice for the language (English: Tara/Samantha, Hindi: Lekha, Tamil: Vani, Telugu: Geeta).
**Malayalam has no macOS voice**, so Malayalam lines are shown as text and labelled. Add voices in
*System Settings → Accessibility → Spoken Content → System voice → Manage Voices*. Higher-quality clips can be
pre-generated with ElevenLabs (`ParkiScanML/tools/generate_nia_voice.py`, key in `ParkiScanML/.env` only).
Translations are drafts awaiting native-speaker review; clinical check-in questions stay in English (labelled)
until marked reviewed. Indic text uses Noto fonts as TextMesh Pro fallbacks; TMP does not perform full complex-
script shaping, so some conjunct glyphs may render imperfectly.

## Run the analysis service

```bash
cd ParkiScanML
./run_local.sh                     # creates .venv, installs requirements, starts http://127.0.0.1:8765
# or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/uvicorn app.main:app --port 8765
.venv/bin/python -m pytest         # tests
```

The game finds the service via `Assets/ParkiScan/Resources/ParkiScanServiceConfig.json`. Without it the game
still runs and saves everything locally (one "ANALYSIS SERVICE OFFLINE" notice).

* Per-session results: `GET /session/<id>/summary`, assessment: `GET /session/<id>/assessment`
* Session folders: `~/Library/Application Support/DefaultCompany/ParkiScan_Neuroverse/Telemetry/<session>/`
  (task JSON + raw CSV + `recordings/*.wav` from the Memory Core). These are participant data — never commit them.

## Train the models

```bash
cd ParkiScanML
.venv/bin/python tools/prepare_datasets.py --zips ~/Downloads   # the seven public datasets → data/raw/
.venv/bin/python -m training.train_all                          # ~15 min; writes models/ and reports/training_report.md
```

Participant-grouped cross-validation, an independent held-out participant set, confusion matrices,
sensitivity/specificity/precision/F1 with confidence intervals, automatic quality flags, a NumPy 1-D CNN on raw
accelerometer windows, a 4-class gait differential and severity sub-stratification. Results and their limits are
in `ParkiScanML/reports/training_report.md` and `ParkiScanML/README.md`.

## Wearable sensors (future)

`ParkiScanML/firmware/esp32_mpu6050/` (ESP32 + MPU-6050 sketch), `parkiscan/sensors/` (packet schema, serial
bridge, simulator) and `Assets/ParkiScan/Scripts/Sensors/SensorHub.cs` (Unity interface: `ISensorSource`,
calibration, clock alignment, task windows). No hardware has been connected or tested; until it is, every
record states "not connected" and simulated data is always labelled and excluded from assessments.

## Publishing to GitHub

```bash
git init && git lfs install
git add .gitattributes .gitignore && git commit -m "repo config"
git add . && git commit -m "ParkiScan Neuroverse"
```

`.gitignore` excludes Unity's generated folders, local backups (`Backup_*~`), screenshots, raw source models,
Python virtualenvs, datasets, session data, recordings and every `.env`. `.gitattributes` routes models, audio,
images and fonts through Git LFS (several models are 50–125 MB; GitHub rejects files over 100 MB without LFS).
Check `git status` before the first push to confirm no participant data or secrets are staged.

## Credits

Datasets: UCI Parkinson's (Little et al. 2007), UCI Telemonitoring (Tsanas et al. 2010), UCI multiple sound
recordings (Sakar et al. 2013), UCI spiral drawings (Isenkul et al. 2014), PhysioNet Gait in Neurodegenerative
Disease (Hausdorff et al.), Daphnet Freezing of Gait (Bächlin et al. 2010), ET_PD_VIS. Fonts: Noto (SIL OFL).
Music and effects: generated procedurally by `ParkiScanML/tools/compose_music.py`.
