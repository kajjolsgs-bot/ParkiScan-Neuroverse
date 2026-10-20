#!/usr/bin/env bash
# Start the ParkiScanML service on http://127.0.0.1:8765 (localhost only).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "[run_local] creating .venv"
  "$PY" -m venv .venv
  ./.venv/bin/pip install --upgrade pip >/dev/null
  ./.venv/bin/pip install -r requirements.txt
fi
[ -f .env ] || { cp .env.example .env; echo "[run_local] created .env from .env.example (edit it to add a NEW ElevenLabs key if you need NIA voice)"; }
HOST="$(grep -E '^PARKISCAN_HOST=' .env | cut -d= -f2 || true)"; HOST="${HOST:-127.0.0.1}"
PORT="$(grep -E '^PARKISCAN_PORT=' .env | cut -d= -f2 || true)"; PORT="${PORT:-8765}"
echo "[run_local] ParkiScanML on http://$HOST:$PORT  (docs: /docs, Ctrl+C to stop)"
exec ./.venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT"
