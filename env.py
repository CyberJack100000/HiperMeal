"""Read KEY=VALUE lines from .env (never committed) so the API key does not have to be typed into every terminal."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load() -> None:
    f = ROOT / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
