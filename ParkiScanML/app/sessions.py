"""File-based session store. Everything stays on this computer under PARKISCAN_SESSION_DIR/<session_id>/."""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

_ID = re.compile(r"^[A-Za-z0-9._-]{1,96}$")
_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(s: str) -> str:
    if not s or not _ID.match(s):
        raise ValueError("session_id must be 1-96 chars of letters, digits, '.', '_' or '-'")
    return s


class SessionStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, sid: str) -> Path:
        return self.root / safe_id(sid)

    def exists(self, sid: str) -> bool:
        return (self.dir(sid) / "session.json").exists()

    def read(self, sid: str) -> dict:
        return json.loads((self.dir(sid) / "session.json").read_text())

    def write(self, sid: str, data: dict):
        d = self.dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "session.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(d / "session.json")

    def start(self, sid: str, meta: dict) -> dict:
        with _lock:
            data = self.read(sid) if self.exists(sid) else {"session_id": sid, "created_utc": now(), "status": "active"}
            data.update({k: v for k, v in meta.items() if v is not None})
            data["status"] = "active"
            data["updated_utc"] = now()
            self.write(sid, data)
            return data

    def stop(self, sid: str) -> dict:
        with _lock:
            if not self.exists(sid):
                raise KeyError(sid)
            data = self.read(sid)
            data["status"] = "stopped"
            data["stopped_utc"] = now()
            self.write(sid, data)
            return data

    def ensure(self, sid: str) -> None:
        if not self.exists(sid):
            self.start(sid, {"source": "implicit"})

    def save_result(self, sid: str, kind: str, name: str, payload: dict) -> Path:
        self.ensure(sid)
        d = self.dir(sid) / kind
        d.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80] or "item"
        p = d / f"{stem}.json"
        i = 1
        while p.exists():
            p = d / f"{stem}_{i}.json"
            i += 1
        p.write_text(json.dumps(payload, indent=2))
        return p

    def results(self, sid: str, kind: str) -> list[dict]:
        d = self.dir(sid) / kind
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
        return out
