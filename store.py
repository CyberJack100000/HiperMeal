"""SQLite storage and photo files. Everything lives under HIPERMEAL_DATA_DIR (default ./data)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DEFAULT_TARGETS = {"kcal": 2000.0, "protein_g": 75.0, "carbs_g": 250.0, "fat_g": 70.0}


def data_dir() -> Path:
    d = Path(os.environ.get("HIPERMEAL_DATA_DIR") or ROOT / "data")
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    return d


def upload_path(analysis_id: str) -> Path:
    if not UUID.match(analysis_id or ""):            # ids become file names, so only real UUIDs are ever accepted
        raise ValueError("bad id")
    return data_dir() / "uploads" / f"{analysis_id}.jpg"


@contextmanager
def db():
    conn = sqlite3.connect(data_dir() / "hipermeal.sqlite3", timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS meals (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, analysis_id TEXT, foods_json TEXT NOT NULL, totals_json TEXT NOT NULL)""")
        cols = {r["name"] for r in c.execute("PRAGMA table_info(meals)")}       # databases from the first version lack these
        for col, ddl in (("local_date", "TEXT"), ("meal_type", "TEXT"), ("note", "TEXT"), ("has_image", "INTEGER NOT NULL DEFAULT 0")):
            if col not in cols:
                c.execute(f"ALTER TABLE meals ADD COLUMN {col} {ddl}")
        c.execute("UPDATE meals SET local_date=substr(created_at,1,10) WHERE local_date IS NULL")
        c.execute("CREATE INDEX IF NOT EXISTS ix_meals_date ON meals(local_date)")
        c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def meal_row(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "created_at": r["created_at"], "local_date": r["local_date"], "meal_type": r["meal_type"], "note": r["note"] or "",
            "analysis_id": r["analysis_id"], "foods": json.loads(r["foods_json"]), "totals": json.loads(r["totals_json"]),
            "image_url": f"/api/meals/{r['id']}/image" if r["has_image"] else None}


def get_targets() -> dict:
    with db() as c:
        row = c.execute("SELECT value FROM settings WHERE key='targets'").fetchone()
    return {**DEFAULT_TARGETS, **(json.loads(row["value"]) if row else {})}


def set_targets(t: dict) -> dict:
    with db() as c:
        c.execute("INSERT INTO settings(key,value) VALUES('targets',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(t),))
    return get_targets()


def purge_orphan_photos(hours: float) -> int:
    """Photos that were analysed but never attached to a saved meal are deleted after `hours`."""
    with db() as c:
        keep = {r["analysis_id"] for r in c.execute("SELECT analysis_id FROM meals WHERE analysis_id IS NOT NULL")}
    removed, cutoff = 0, time.time() - hours * 3600
    for p in (data_dir() / "uploads").glob("*.jpg"):
        if p.stem not in keep and p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def wipe() -> None:
    with db() as c:
        c.execute("DELETE FROM meals")
    for p in (data_dir() / "uploads").glob("*"):
        p.unlink(missing_ok=True)
