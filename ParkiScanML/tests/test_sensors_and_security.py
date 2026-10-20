import json
import re
from pathlib import Path

from parkiscan.sensors.imu import parse_line, summarize
from parkiscan.sensors.simulator import generate

ROOT = Path(__file__).resolve().parents[1]


def test_parse_line_valid_and_invalid():
    ok = parse_line('{"dev":"esp32-wrist-01","seq":1,"t_ms":10,"ax":0.0,"ay":-1.0,"az":0.0,"gx":0,"gy":0,"gz":0}')
    assert ok is not None and ok.sim is False
    assert parse_line("garbage") is None
    assert parse_line('{"dev":"x","seq":-1,"t_ms":0,"ax":0,"ay":0,"az":0}') is None


def test_simulator_always_flagged():
    pk = generate(seconds=1, dev="esp32")
    assert all(p.sim and p.dev.startswith("SIM-") for p in pk)


def test_summary_detects_sequence_gaps():
    pk = generate(seconds=3)
    del pk[50:60]
    s = summarize(pk)
    assert s["features"]["sequence_gaps"] == 1


def test_env_example_has_placeholders_only():
    txt = (ROOT / ".env.example").read_text()
    m = re.search(r"^ELEVENLABS_API_KEY=(.*)$", txt, re.M)
    assert m and m.group(1).strip() in {"", "replace-with-your-new-key"}


def test_gitignore_excludes_env_and_data():
    gi = (ROOT / ".gitignore").read_text().splitlines()
    for pat in (".env", "sessions/", "data/raw/"):
        assert pat in gi


def test_no_secret_shaped_strings_in_repo():
    # ElevenLabs keys are long hex/base62 tokens prefixed "sk_"; nothing like that may be committed
    pat = re.compile(r"sk_[A-Za-z0-9]{20,}")
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".txt", ".sh", ".example", ".json", ".ino", ".cfg", ".toml"} \
                and ".venv" not in p.parts and p.name != ".env":
            assert not pat.search(p.read_text(errors="ignore")), f"secret-like token in {p}"


def test_unity_service_config_points_to_localhost():
    cfg = ROOT.parent / "Assets" / "ParkiScan" / "Resources" / "ParkiScanServiceConfig.json"
    if cfg.exists():
        assert "127.0.0.1" in json.loads(cfg.read_text())["baseUrl"]
