"""Pre-generate NIA's spoken lines with ElevenLabs and drop them where Unity loads them.

    python tools/generate_nia_voice.py --lang en            # English lines found in the Unity scripts
    python tools/generate_nia_voice.py --lang ta hi ml te   # translated lines (non-clinical drafts only)
    python tools/generate_nia_voice.py --lang en --dry-run  # list what would be generated, no API calls

Output: Assets/ParkiScan/Resources/NiaVoice/<lang>/<id>.mp3 where <id> = first 16 hex chars of
SHA1(UTF-8 line) — identical to LocalizationManager.VoiceId in Unity, so DialogueManager finds the clip
automatically and the "TEXT ONLY" label disappears for that line.

Security: the key is read from ParkiScanML/.env (ELEVENLABS_API_KEY) and never printed. Only scripted
game text is sent; participant recordings are never sent. Clinical questionnaire items are skipped
unless their translation is marked reviewed=true.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.tts import TtsError, synthesize, voice_id_for_text  # noqa: E402

UNITY = ROOT.parent / "Assets" / "ParkiScan"
SAY = re.compile(r'\bSay\(\s*"((?:[^"\\]|\\.)*)"')


def _unescape(s: str) -> str:
    """Resolve C# string escapes (\\uXXXX, \\n, \\", \\\\) without touching other characters."""
    rep = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "'": "'"}
    return re.sub(r"\\(u[0-9a-fA-F]{4}|.)",
                  lambda m: chr(int(m.group(1)[1:], 16)) if m.group(1)[0] == "u" and len(m.group(1)) == 5
                  else rep.get(m.group(1), m.group(1)), s)


def english_lines() -> list[str]:
    out = []
    for cs in sorted((UNITY / "Scripts").rglob("*.cs")):
        for m in SAY.finditer(cs.read_text(encoding="utf-8", errors="ignore")):
            s = _unescape(m.group(1))
            if len(s) >= 3 and s not in out:
                out.append(s)
    return out


def translated_lines(lang: str) -> list[str]:
    p = UNITY / "Resources" / "Localization" / f"{lang}.json"
    if not p.exists():
        return []
    tab = json.loads(p.read_text(encoding="utf-8"))
    eng = set(english_lines())
    out = []
    for e in tab.get("entries", []):
        if e.get("clinical") and not e.get("reviewed"):
            continue
        if e.get("key") in eng and e.get("value"):
            out.append(e["value"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["en"], choices=["en", "ta", "hi", "ml", "te"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    if not a.dry_run and not s.tts_configured:
        sys.exit("ElevenLabs is not configured. Put a NEW key in ParkiScanML/.env (ELEVENLABS_API_KEY=...). "
                 "Any previously exposed key must be revoked first.")
    for lang in a.lang:
        lines = english_lines() if lang == "en" else translated_lines(lang)
        out_dir = UNITY / "Resources" / "NiaVoice" / lang
        print(f"[nia-voice] {lang}: {len(lines)} lines -> {out_dir}")
        for i, text in enumerate(lines, 1):
            vid = voice_id_for_text(text)
            if a.dry_run:
                print(f"  {vid}  {text[:70]}")
                continue
            try:
                synthesize(s, text, out_dir)
                print(f"  [{i}/{len(lines)}] {vid}")
            except TtsError as e:
                sys.exit(f"[nia-voice] stopped: {e}")
            time.sleep(0.3)  # stay well under rate limits
    print("[nia-voice] done. Return to Unity; it imports the new clips automatically.")


if __name__ == "__main__":
    main()
