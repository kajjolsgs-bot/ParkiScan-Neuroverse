"""ElevenLabs text-to-speech for NIA's scripted lines (server side only).

* The API key is read from the environment (ParkiScanML/.env) and is never logged or returned.
* Only fixed, scripted game text is sent. Participant recordings are NEVER sent to ElevenLabs.
* Results are cached on disk by SHA1(text) so each line is synthesised once.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from .config import Settings

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"


def voice_id_for_text(text: str) -> str:
    """Same id as Unity's LocalizationManager.VoiceId: first 8 bytes of SHA1(UTF-8) as hex."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class TtsError(RuntimeError):
    pass


def synthesize(settings: Settings, text: str, cache_dir: Path, timeout: float = 30.0) -> Path:
    if not settings.tts_configured:
        raise TtsError("ElevenLabs is not configured (set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in ParkiScanML/.env)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{voice_id_for_text(text)}.mp3"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        r = httpx.post(
            API.format(voice=settings.elevenlabs_voice_id),
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": settings.elevenlabs_api_key, "accept": "audio/mpeg"},
            json={"text": text, "model_id": settings.elevenlabs_model_id,
                  "voice_settings": {"stability": 0.55, "similarity_boost": 0.75, "style": 0.15,
                                     "use_speaker_boost": True}},
            timeout=timeout,
        )
    except httpx.HTTPError as e:  # network failure: message has no secrets
        raise TtsError(f"network error contacting ElevenLabs: {type(e).__name__}") from None
    if r.status_code == 401:
        raise TtsError("ElevenLabs rejected the API key (401). Revoke/replace it in ParkiScanML/.env")
    if r.status_code != 200:
        raise TtsError(f"ElevenLabs returned HTTP {r.status_code}")
    tmp = out.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(out)
    return out
