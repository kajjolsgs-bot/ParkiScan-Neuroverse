# Publishing the repository to GitHub

The project folder is about **1.9 GB**. Every file is under GitHub's 100 MB per-file limit (the largest 3-D model is ~80 MB), so
**no Git LFS is needed** (LFS's free quota is 1 GB and would not fit). GitHub limits a single push to 2 GB, so the steps
below push in three parts.

Ignored automatically (`.gitignore` files): Unity `Library/ Temp/ Logs/ UserSettings/`, editor backups `Assets/ParkiScan/Backup_*~/`,
QA screenshots, `SourceAssets/`, `_Assembled/`, and in `ParkiScanML/`: `.venv/`, `.env`, raw datasets, sessions, caches.

## Before you push

1. **Revoke the old ElevenLabs API key** in your ElevenLabs account (it was exposed earlier). Put the new key only in
   `ParkiScanML/.env` — that file is ignored and never committed. Check: `git check-ignore ParkiScanML/.env` must print the path.
2. Fill in the team table at the top of `README.md` (and `TEAM` in `docs/src/fill.py` if you re-generate the slides).
3. Add the unlisted YouTube link of the demo video to `README.md` § 8.

## Commands (macOS Terminal, in the project folder)

```bash
cd ~/ParkiScan_Neuroverse
git init -b main
git config http.postBuffer 524288000          # large pushes over HTTPS

# safety check — none of these may appear in the list
git status --ignored --short | grep -E "\.env$|Library/|\.venv/" || true

# part 1 — code, docs, settings, ML service, everything except the 3-D art
git add .gitignore README.md LICENSE THIRD_PARTY_NOTICES.md docs Packages ProjectSettings ParkiScanML
git add Assets --  ':!Assets/ParkiScan/Art'
git commit -m "ParkiScan Neuroverse (game + digital twin): game code, twin service, models, docs"

# create an empty PUBLIC repository on github.com (no README / licence — they are already here), then:
git remote add origin https://github.com/<your-account>/ParkiScan-Neuroverse.git
git push -u origin main

# part 2 — characters, Level 1–3 art
git add Assets/ParkiScan/Art/Characters* Assets/ParkiScan/Art/Level* Assets/ParkiScan/Art/Fonts* Assets/ParkiScan/Art/Environment* Assets/ParkiScan/Art/Generated*
git commit -m "Game art: characters and level assets"
git push

# part 3 — worlds and India environment
git add Assets/ParkiScan/Art
git commit -m "Game art: worlds and India environment"
git push
```

GitHub will warn that some files are larger than 50 MB — that is a warning, not an error.

## Turn on the no-install dashboard demo

Repository → Settings → Pages → Source "Deploy from a branch", branch `main`, folder `/docs` → Save. After a minute the
dashboard opens at `https://<your-account>.github.io/ParkiScan-Neuroverse/dashboard/` — put that link in the README and the
submission form so judges can try the twin without installing Python.

## Check the published repository

* The README renders with the results tables, and `docs/ParkiScan_Neuroverse_Game_DigitalTwin_Architecture.pdf` and
  `docs/ParkiScan_Neuroverse_Game_DigitalTwin_Presentation.pdf` open in the browser.
* `LICENSE` shows "Apache License 2.0" in the repository sidebar.
* Search the repository for `sk_` and `xi-api-key` — there must be no results.
