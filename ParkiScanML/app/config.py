"""Runtime configuration for the ParkiScanML service.

Values come from environment variables, optionally loaded from ParkiScanML/.env (python-dotenv).
Secrets (ELEVENLABS_API_KEY) are never logged, echoed in responses or written to disk; only a
boolean "configured" flag and a masked suffix are exposed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

VERSION = "1.0.0"
SCHEMA_VERSION = "parkiscan.session.v1"


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    p = Path(v).expanduser() if v else default
    return p if p.is_absolute() else (ROOT / p)


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: os.getenv("PARKISCAN_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PARKISCAN_PORT", "8765")))
    data_dir: Path = field(default_factory=lambda: _path("PARKISCAN_DATA_DIR", ROOT / "data" / "raw"))
    session_dir: Path = field(default_factory=lambda: _path("PARKISCAN_SESSION_DIR", ROOT / "sessions"))
    models_dir: Path = field(default_factory=lambda: _path("PARKISCAN_MODELS_DIR", ROOT / "models"))
    reports_dir: Path = field(default_factory=lambda: _path("PARKISCAN_REPORTS_DIR", ROOT / "reports"))
    # research-only model outputs are OFF unless explicitly enabled; they are never diagnoses
    enable_research_scores: bool = field(default_factory=lambda: _bool("PARKISCAN_ENABLE_RESEARCH_SCORES", False))
    # raw paths sent by Unity are only read if they sit under one of these roots (comma separated)
    allowed_raw_roots: tuple = field(default_factory=lambda: tuple(
        Path(p).expanduser() for p in os.getenv(
            "PARKISCAN_ALLOWED_RAW_ROOTS",
            "~/Library/Application Support/DefaultCompany/ParkiScan_Neuroverse," + str(ROOT / "sessions"),
        ).split(",") if p.strip()))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("PARKISCAN_MAX_UPLOAD_MB", "25")))
    # ElevenLabs — used ONLY to synthesise NIA's scripted lines. Participant audio is never sent.
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", "").strip(), repr=False)
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "").strip())
    elevenlabs_model_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip())
    enable_tts_endpoint: bool = field(default_factory=lambda: _bool("PARKISCAN_ENABLE_TTS", False))

    @property
    def tts_configured(self) -> bool:
        k = self.elevenlabs_api_key
        return bool(k) and not k.lower().startswith("replace") and bool(self.elevenlabs_voice_id)

    def key_hint(self) -> str:
        """Description safe for logs: reveals only whether a key is present, never any part of it."""
        return "set" if self.elevenlabs_api_key else "not set"


def get_settings() -> Settings:
    s = Settings()
    s.session_dir.mkdir(parents=True, exist_ok=True)
    return s
