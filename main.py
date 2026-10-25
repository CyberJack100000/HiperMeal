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

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Path as PathParam, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent


import env  # noqa: E402

env.load()

import ai  # noqa: E402  (after .env so the key is visible)
import nutrition  # noqa: E402
import store  # noqa: E402

STATIC_DIR = ROOT / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SIDE = 1600
MAX_PIXELS = 40_000_000
# What the file really is decides, not the type the browser claims (browsers often report HEIC photos with no type at all).
# Pillow's format name -> the name shown to people.
ALLOWED_FORMATS = {"JPEG": "JPG", "MPO": "JPG", "PNG": "PNG", "WEBP": "WEBP", "GIF": "GIF", "BMP": "BMP", "TIFF": "TIFF", "AVIF": "AVIF", "HEIF": "HEIC"}
try:                                                        # iPhone photos are HEIC; this needs the pillow-heif package
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False


def image_formats() -> list[str]:
    names = ["JPG", "PNG", "WEBP", "GIF", "BMP", "TIFF"]
    from PIL import features
    if features.check("avif"):
        names.append("AVIF")
    if HEIC_OK:
        names.append("HEIC")
    return names
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


def rate_limit(request: Request, limit: int = 20, window: int = 60, bucket: str = "analyze") -> None:
    """Model requests cost money and time; a runaway script should not be able to spend either."""
    ip = f"{bucket}:{request.client.host if request.client else '?'}"
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many photos in a short time. Wait a minute and try again.")
    q.append(now)
    if len(_hits) > 500:                                    # forget quiet addresses so this cannot grow without bound
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > window]:
            del _hits[k]


# ----------------------------------------------------------------------------------------------------------- the person's own API key
class Creds:
    """A Gemma API key (and optionally a model) the person entered in the app. It arrives in a header with the one request that needs it, goes straight to
    Google, and is never stored, logged or echoed by this server. Without one, the key in the server's environment (if any) is used."""
    def __init__(self, key: str | None, model: str | None):
        self.key, self.model = key, model


def creds(x_gemini_key: str | None = Header(None), x_gemma_model: str | None = Header(None)) -> Creds:
    key = (x_gemini_key or "").strip() or None
    model = (x_gemma_model or "").strip() or None
    if key is not None and not ai.VALID_KEY.match(key):
        raise HTTPException(422, "That does not look like an API key. Paste just the key, with no spaces.")
    if model is not None and not ai.VALID_MODEL.match(model):
        raise HTTPException(422, "That does not look like a model name, for example gemma-4-31b-it.")
    return Creds(key, model)


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
    """Identify the file by its content, decode its first picture, fix rotation, shrink and re-encode as JPEG. Re-encoding drops all metadata (phone photos
    carry GPS location), and the smaller picture is faster and cheaper to send to the model. Animated GIFs, multi-page TIFFs and similar use their first frame."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2                 # Pillow itself refuses anything beyond twice this
    supported = ", ".join(image_formats())
    try:
        src = Image.open(BytesIO(contents))
    except Image.DecompressionBombError as exc:
        raise HTTPException(400, "This picture has too many pixels to process. Use a smaller or lower-resolution photo.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(415, f"That file is not a supported image. Use {supported}.") from exc
    if src.format not in ALLOWED_FORMATS:
        raise HTTPException(415, f"{(src.format or 'That').title()} files are not supported. Use {supported}.")
    if src.width * src.height > MAX_PIXELS:
        raise HTTPException(400, "This picture has too many pixels to process. Use a smaller or lower-resolution photo.")
    try:
        with src as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "PA", "P"):
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
    except Image.DecompressionBombError as exc:
        raise HTTPException(400, "This picture has too many pixels to process. Use a smaller or lower-resolution photo.") from exc
    except Exception as exc:
        raise HTTPException(400, "Could not read this image. It may be damaged; try another file.") from exc


# ----------------------------------------------------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "HiperMeal API", "version": app.version, "ai_configured": ai.server_key_set(), "model": ai.model_name(), "ai_mode": ai.mode(), "image_formats": image_formats(),
            "nutrition_records": len(nutrition.NUTRITION)}


@app.post("/api/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...), hint: str | None = Form(None), c: Creds = Depends(creds)) -> dict[str, Any]:
    rate_limit(request)
    hint = (hint or "").strip().lower() or None
    if hint is not None and hint not in ai.CUISINES:
        raise HTTPException(422, "Unknown cuisine hint.")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB limit.")
    jpeg = prepare_image(contents)
    analysis_id = str(uuid.uuid4())
    store.upload_path(analysis_id).write_bytes(jpeg)         # kept only if the meal is saved (see purge_orphan_photos)

    try:
        suggestions = await ai.analyse(jpeg, hint, c.key, c.model)
    except ai.AIError as e:
        raise HTTPException(502, str(e))
    ai_enabled = suggestions is not None
    if suggestions is None:
        suggestions = [{"name": "roti", "grams": 60, "confidence": "manual-review"}, {"name": "dal", "grams": 150, "confidence": "manual-review"},
                       {"name": "cooked rice", "grams": 120, "confidence": "manual-review"}]
    foods = []
    for s in suggestions:
        row = {"name": s["name"], "grams": s["grams"], "confidence": s["confidence"], "preparation": s.get("preparation", ""), "reasoning": s.get("reasoning", ""),
               "nutrition": None, "matched_name": None, "match": None}
        m = nutrition.find(s["name"])
        if m:
            row.update(m.as_dict())
        else:
            row["notes"] = "No nutrition record for this name. Pick a known food or type its values per 100 g."
        foods.append(row)
    return {"analysis_id": analysis_id, "ai_enabled": ai_enabled, "foods": foods,
            "message": ("AI suggestions generated. Food identity and portion size may be wrong; review each item." if ai_enabled else
                        "AI vision is not set up (no API key), so these are demo suggestions for trying the app, not an analysis of your photo. Add your Gemma API key in Settings to analyse real photos.")}


@app.post("/api/nutrition/lookup")
def nutrition_lookup(body: LookupInput) -> dict[str, Any]:
    m = nutrition.find(body.food_name)
    if not m:
        raise HTTPException(404, f"No nutrition record for '{body.food_name.strip()}'. Type its values per 100 g instead.")
    return {**m.as_dict(), "food": m.per_100g, "basis": "per 100 g", "estimated": True}


class EstimateInput(BaseModel):
    food_name: str = Field(min_length=1, max_length=100)


@app.post("/api/ai/test")
async def ai_test(request: Request, c: Creds = Depends(creds)) -> dict[str, Any]:
    """One tiny request so a key and model can be checked in Settings before a photo is sent."""
    rate_limit(request, limit=10, bucket="test")
    try:
        return await ai.check(c.key, c.model)
    except ai.AIError as exc:
        raise HTTPException(502, str(exc))


@app.post("/api/nutrition/estimate")
async def nutrition_estimate(request: Request, body: EstimateInput, c: Creds = Depends(creds)) -> dict[str, Any]:
    """Ask the model for rough per-100-g values of a food the table does not know. The browser fills them in for the person to check; nothing is saved."""
    if not ai.configured(c.key):
        raise HTTPException(503, "AI estimates need an API key. Add yours in Settings, or type the values yourself.")
    rate_limit(request, limit=30, bucket="estimate")
    try:
        e = await ai.estimate(" ".join(body.food_name.split()), c.key, c.model)
    except ai.AIError as exc:
        raise HTTPException(502, str(exc))
    return {"nutrition": {k: e[k] for k in nutrition.KEYS}, "note": e["note"], "estimated_by": "ai", "basis": "per 100 g"}


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


@app.get("/api/photos/{analysis_id}")
def analysed_photo(analysis_id: UUID_PATH):
    """The cleaned JPEG of a photo that was just analysed, so the review screen can show it even when the browser cannot display the original (HEIC, TIFF...)."""
    p = store.upload_path(analysis_id)
    if not p.is_file():
        raise HTTPException(404, "Photo not found.")
    return FileResponse(p, media_type="image/jpeg")


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
