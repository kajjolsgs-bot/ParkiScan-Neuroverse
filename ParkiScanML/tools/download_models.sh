#!/usr/bin/env bash
# Optional model downloads for Python-side tools. Unity already ships its own BlazePose model, and the
# ParkiScanML service runs without any of these. Re-running is safe (existing files are skipped).
#   ./tools/download_models.sh            # MediaPipe face landmarker + pose landmarker (.task)
#   ./tools/download_models.sh --asr      # also install faster-whisper and fetch the 'small' (analysis) and
#                                         # 'base' (live word follow in the Memory Core) models
set -euo pipefail
cd "$(dirname "$0")/.."
MP=mediapipe_models
mkdir -p "$MP"
fetch () {  # url dest
  if [ -s "$2" ]; then echo "[models] exists: $2"; return; fi
  echo "[models] downloading $(basename "$2")"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
fetch https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task "$MP/face_landmarker.task"
fetch https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task "$MP/pose_landmarker_full.task"
shasum -a 256 "$MP"/*.task > "$MP/SHA256SUMS.txt"
echo "[models] MediaPipe models in $MP (checksums recorded in SHA256SUMS.txt)"

if [[ "${1:-}" == "--asr" ]]; then
  PY=./.venv/bin/python; [ -x "$PY" ] || PY=python3
  "$PY" -m pip install "faster-whisper>=1.0"
  "$PY" - <<'PYEOF'
from faster_whisper import WhisperModel
# downloads once into ./asr_models; runs fully offline afterwards (no audio leaves the computer)
WhisperModel("small", device="cpu", compute_type="int8", download_root="asr_models")
WhisperModel("base", device="cpu", compute_type="int8", download_root="asr_models")
print("[models] faster-whisper 'small' + 'base' ready in asr_models/")
PYEOF
fi
