"""HiperMeal API. Run:  python -m uvicorn main:app --reload"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Path as PathParam, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent


def _load_env_file() -> None:
    """Read KEY=VALUE lines from .env (never committed) so the API key does not have to be typed into every terminal."""
    f = ROOT / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env_file()

import ai  # noqa: E402  (after .env so the key is visible)
import nutrition  # noqa: E402
import store  # noqa: E402

STATIC_DIR = ROOT / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SIDE = 1600
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ORPHAN_HOURS = float(os.environ.get("HIPERMEAL_ORPHAN_HOURS", "24"))
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")

app = FastAPI(title="HiperMeal API", description="Backend for HiperMeal's visual nutrition diary.", version="1.0.0")
store.init()
store.purge_orphan_photos(ORPHAN_HOURS)

UUID_PATH = Annotated[str, PathParam(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")]


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"          # responses hold personal data
    return resp


_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request, limit: int = 20, window: int = 60) -> None:
    """Photo analysis costs money and time; a runaway script should not be able to spend either."""
    ip = request.client.host if request.client else "?"
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many photos in a short time. Wait a minute and try again.")
    q.append(now)
    if len(_hits) > 500:                                    # forget quiet addresses so this cannot grow without bound
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > window]:
            del _hits[k]


# ----------------------------------------------------------------------------------------------------------- models
class Custom(BaseModel):
    kcal: float = Field(ge=0, le=900)
    protein_g: float = Field(ge=0, le=100)
    carbs_g: float = Field(ge=0, le=100)
    fat_g: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def plausible(self):
        if self.protein_g + self.carbs_g + self.fat_g > 100.5:
            raise ValueError("Protein, carbs and fat together cannot exceed 100 g per 100 g of food.")
        return self


class FoodInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    grams: float = Field(gt=0, le=5000)
    custom: Custom | None = None            # per-100-g values typed by the user, for foods the built-in table does not know


class MealInput(BaseModel):
    analysis_id: str | None = Field(None, pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    foods: list[FoodInput] = Field(min_length=1, max_length=50)
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"] | None = None
    note: str = Field("", max_length=300)
    local_date: str | None = None           # the person's own calendar day (the server does not know their time zone)

    @field_validator("local_date")
    @classmethod
    def real_date(cls, v):
        if v is not None:
            date.fromisoformat(v)
        return v


class LookupInput(BaseModel):
    food_name: str = Field(min_length=1, max_length=100)


class Targets(BaseModel):
    kcal: float = Field(ge=500, le=10000)
    protein_g: float = Field(ge=0, le=500)
    carbs_g: float = Field(ge=0, le=1500)
    fat_g: float = Field(ge=0, le=500)


def _today(s: str | None) -> date:
    try:
        return date.fromisoformat(s) if s else datetime.now(timezone.utc).date()
    except ValueError:
        raise HTTPException(422, "today must look like 2026-01-31.")


# ----------------------------------------------------------------------------------------------------------- images
def prepare_image(contents: bytes) -> bytes:
    """Decode, fix rotation, shrink and re-encode as JPEG. Re-encoding drops all metadata (phone photos carry GPS location), and the
    smaller picture is faster and cheaper to send to the model."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = 40_000_000
    try:
        with Image.open(BytesIO(contents)) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            out = BytesIO()
            im.save(out, "JPEG", quality=85, optimize=True)
            return out.getvalue()
    except Exception as exc:
        raise HTTPException(400, "Could not decode this image. Try another JPG, PNG, or WEBP file.") from exc


# ----------------------------------------------------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "HiperMeal API", "version": app.version, "ai_configured": ai.configured(), "model": ai.model_name(),
            "nutrition_records": len(nutrition.NUTRITION)}


@app.post("/api/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    rate_limit(request)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported image type. Upload JPG, PNG, or WEBP.")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB limit.")
    jpeg = prepare_image(contents)
    analysis_id = str(uuid.uuid4())
    store.upload_path(analysis_id).write_bytes(jpeg)         # kept only if the meal is saved (see purge_orphan_photos)

    try:
        suggestions = await ai.analyse(jpeg)
    except ai.AIError as e:
        raise HTTPException(502, str(e))
    ai_enabled = suggestions is not None
    if suggestions is None:
        suggestions = [{"name": "roti", "grams": 60, "confidence": "manual-review"}, {"name": "dal", "grams": 150, "confidence": "manual-review"},
                       {"name": "cooked rice", "grams": 120, "confidence": "manual-review"}]
    foods = []
    for s in suggestions:
        row = {"name": s["name"], "grams": s["grams"], "confidence": s["confidence"], "nutrition": None, "matched_name": None, "match": None}
        m = nutrition.find(s["name"])
        if m:
            row.update(m.as_dict())
        else:
            row["notes"] = "No nutrition record for this name. Pick a known food or type its values per 100 g."
        foods.append(row)
    return {"analysis_id": analysis_id, "ai_enabled": ai_enabled, "foods": foods,
            "message": ("AI suggestions generated. Food identity and portion size may be wrong; review each item." if ai_enabled else
                        "AI vision is not configured (no GEMINI_API_KEY), so these are demo suggestions for trying the app, not an analysis of your photo.")}


@app.post("/api/nutrition/lookup")
def nutrition_lookup(body: LookupInput) -> dict[str, Any]:
    m = nutrition.find(body.food_name)
    if not m:
        raise HTTPException(404, f"No nutrition record for '{body.food_name.strip()}'. Type its values per 100 g instead.")
    return {**m.as_dict(), "food": m.per_100g, "basis": "per 100 g", "estimated": True}


@app.post("/api/meals")
def save_meal(body: MealInput) -> dict[str, Any]:
    # Totals are always recalculated here from per-100-g values; whatever the browser calculated is ignored.
    total = {k: 0.0 for k in nutrition.KEYS}
    stored = []
    for f in body.foods:
        if f.custom is not None:
            per100, matched, how, source = f.custom.model_dump(), None, None, "user"
        else:
            m = nutrition.find(f.name)
            if not m:
                raise HTTPException(422, f"No nutrition values for '{f.name.strip()}'. Type its values per 100 g, or choose a known food.")
            per100, matched, how, source = m.per_100g, m.name, m.how, "table"
        part = nutrition.scale(per100, f.grams)
        for k in total:
            total[k] += part[k]
        stored.append({"name": " ".join(f.name.split()), "grams": f.grams, "matched_name": matched, "match": how, "source": source,
                       "nutrition_per_100g": per100, "nutrition": part})
    meal_id, now = str(uuid.uuid4()), datetime.now(timezone.utc)
    has_image = 0
    if body.analysis_id:
        try:
            has_image = int(store.upload_path(body.analysis_id).is_file())
        except ValueError:
            pass
    with store.db() as c:
        c.execute("INSERT INTO meals(id,created_at,analysis_id,foods_json,totals_json,local_date,meal_type,note,has_image) VALUES(?,?,?,?,?,?,?,?,?)",
                  (meal_id, now.isoformat(), body.analysis_id, json.dumps(stored), json.dumps(total), body.local_date or now.date().isoformat(),
                   body.meal_type, body.note.strip(), has_image))
        row = c.execute("SELECT * FROM meals WHERE id=?", (meal_id,)).fetchone()
    return {**store.meal_row(row), "notice": "Nutrition values are rough estimates."}


@app.get("/api/meals")
def list_meals(days: int | None = Query(None, ge=1, le=366), today: str | None = None, limit: int = Query(200, ge=1, le=500)) -> dict[str, Any]:
    sql, args = "SELECT * FROM meals", []
    if days:
        sql += " WHERE local_date >= ?"
        args.append((_today(today) - timedelta(days=days - 1)).isoformat())
    sql += " ORDER BY local_date DESC, created_at DESC LIMIT ?"        # by the day a meal belongs to, so a back-dated meal lands on its own day
    with store.db() as c:
        rows = c.execute(sql, (*args, limit)).fetchall()
    return {"meals": [store.meal_row(r) for r in rows]}


@app.get("/api/meals/{meal_id}/image")
def meal_image(meal_id: UUID_PATH):
    with store.db() as c:
        row = c.execute("SELECT analysis_id, has_image FROM meals WHERE id=?", (meal_id,)).fetchone()
    if not row or not row["has_image"]:
        raise HTTPException(404, "No photo for this meal.")
    p = store.upload_path(row["analysis_id"])
    if not p.is_file():
        raise HTTPException(404, "No photo for this meal.")
    return FileResponse(p, media_type="image/jpeg")


@app.delete("/api/meals/{meal_id}")
def delete_meal(meal_id: UUID_PATH) -> dict[str, bool]:
    with store.db() as c:
        row = c.execute("SELECT analysis_id FROM meals WHERE id=?", (meal_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Meal not found.")
        c.execute("DELETE FROM meals WHERE id=?", (meal_id,))
    if row["analysis_id"]:
        try:
            store.upload_path(row["analysis_id"]).unlink(missing_ok=True)
        except ValueError:
            pass
    return {"deleted": True}


@app.delete("/api/data")
def delete_everything() -> dict[str, bool]:
    store.wipe()
    return {"deleted": True}


@app.get("/api/summary")
def summary(days: int = Query(7, ge=1, le=90), today: str | None = None) -> dict[str, Any]:
    end = _today(today)
    start = end - timedelta(days=days - 1)
    with store.db() as c:
        rows = c.execute("SELECT local_date, foods_json, totals_json FROM meals WHERE local_date >= ? AND local_date <= ?", (start.isoformat(), end.isoformat())).fetchall()
    per_day = {(start + timedelta(days=i)).isoformat(): {k: 0.0 for k in nutrition.KEYS} | {"meals": 0} for i in range(days)}
    foods: Counter = Counter()
    grams: Counter = Counter()
    for r in rows:
        d = per_day.get(r["local_date"])
        if d is None:
            continue
        t = json.loads(r["totals_json"])
        for k in nutrition.KEYS:
            d[k] += t.get(k, 0.0)
        d["meals"] += 1
        for f in json.loads(r["foods_json"]):
            name = (f.get("matched_name") or f["name"]).lower()
            foods[name] += 1
            grams[name] += f.get("grams", 0)
    logged = [d for d in per_day.values() if d["meals"]]
    avg = {k: (sum(d[k] for d in logged) / len(logged) if logged else 0.0) for k in nutrition.KEYS}
    return {"days": [{"date": k, **v} for k, v in per_day.items()], "today": {"date": end.isoformat(), **per_day[end.isoformat()]}, "average_per_logged_day": avg,
            "logged_days": len(logged), "top_foods": [{"name": n, "times": c, "grams": round(grams[n])} for n, c in foods.most_common(8)], "targets": store.get_targets()}


@app.get("/api/settings")
def read_settings() -> dict[str, Any]:
    return {"targets": store.get_targets()}


@app.put("/api/settings")
def write_settings(body: Targets) -> dict[str, Any]:
    return {"targets": store.set_targets(body.model_dump())}


@app.get("/api/foods")
def known_foods() -> dict[str, Any]:
    """Names the built-in table knows (used for suggestions in the editor)."""
    return {"foods": sorted(nutrition.NUTRITION), "estimated": True}


# ----------------------------------------------------------------------------------------------------------- pages
# Python takes file types from the operating system. On some Windows machines the registry says ".js" is "text/plain", and together with
# "nosniff" the browser then refuses to run the page's scripts. So the types the app needs are fixed here.
FILE_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8", ".svg": "image/svg+xml"}


class Static(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        kind = FILE_TYPES.get(Path(path).suffix.lower())
        if kind and resp.status_code in (200, 304):
            resp.headers["content-type"] = kind
        return resp


app.mount("/static", Static(directory=STATIC_DIR), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type=FILE_TYPES[".html"])
