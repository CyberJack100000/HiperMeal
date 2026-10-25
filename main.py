from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "hipermeal.sqlite3"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="HiperMeal API",
    description="Backend for HiperMeal's visual nutrition diary prototype.",
    version="0.3.0",
)

NUTRITION: dict[str, dict[str, float]] = {
    # Illustrative per-100-g values for prototype testing. Replace with verified,
    # cited food-composition data before real-world dietary use.
    "roti": {"kcal": 297, "protein_g": 8.2, "carbs_g": 55.0, "fat_g": 7.5},
    "chapati": {"kcal": 297, "protein_g": 8.2, "carbs_g": 55.0, "fat_g": 7.5},
    "dal": {"kcal": 116, "protein_g": 7.6, "carbs_g": 20.1, "fat_g": 0.4},
    "cooked rice": {"kcal": 130, "protein_g": 2.7, "carbs_g": 28.2, "fat_g": 0.3},
    "white rice": {"kcal": 130, "protein_g": 2.7, "carbs_g": 28.2, "fat_g": 0.3},
    "paneer": {"kcal": 265, "protein_g": 18.3, "carbs_g": 1.2, "fat_g": 20.8},
    "mixed vegetables": {"kcal": 75, "protein_g": 2.5, "carbs_g": 12.0, "fat_g": 2.2},
    "banana": {"kcal": 89, "protein_g": 1.1, "carbs_g": 22.8, "fat_g": 0.3},
    "oats": {"kcal": 389, "protein_g": 16.9, "carbs_g": 66.3, "fat_g": 6.9},
    "egg": {"kcal": 155, "protein_g": 13.0, "carbs_g": 1.1, "fat_g": 11.0},
    "boiled egg": {"kcal": 155, "protein_g": 13.0, "carbs_g": 1.1, "fat_g": 11.0},
    "apple": {"kcal": 52, "protein_g": 0.3, "carbs_g": 13.8, "fat_g": 0.2},
    "yogurt": {"kcal": 61, "protein_g": 3.5, "carbs_g": 4.7, "fat_g": 3.3},
    "curd": {"kcal": 61, "protein_g": 3.5, "carbs_g": 4.7, "fat_g": 3.3},
    "chicken breast": {"kcal": 165, "protein_g": 31.0, "carbs_g": 0.0, "fat_g": 3.6},
    "idli": {"kcal": 146, "protein_g": 4.0, "carbs_g": 29.0, "fat_g": 0.7},
    "dosa": {"kcal": 168, "protein_g": 4.0, "carbs_g": 29.0, "fat_g": 3.7},
    "poha": {"kcal": 130, "protein_g": 2.5, "carbs_g": 25.0, "fat_g": 2.5},
    "samosa": {"kcal": 308, "protein_g": 6.0, "carbs_g": 32.0, "fat_g": 17.0},
}

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                analysis_id TEXT,
                foods_json TEXT NOT NULL,
                totals_json TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

class FoodInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    grams: float = Field(gt=0, le=5000)

class TotalsInput(BaseModel):
    kcal: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)

class MealInput(BaseModel):
    analysis_id: str | None = None
    foods: list[FoodInput] = Field(min_length=1, max_length=50)
    totals: TotalsInput

class LookupInput(BaseModel):
    food_name: str = Field(min_length=1, max_length=100)

def lookup_food(name: str) -> tuple[str, dict[str, float]]:
    key = " ".join(name.lower().strip().split())
    if key in NUTRITION:
        return key, NUTRITION[key]
    # Avoid pretending an unknown food is an exact match.
    aliases = {
        "rice": "cooked rice",
        "chapati/roti": "roti",
        "lentils": "dal",
        "lentil curry": "dal",
        "vegetables": "mixed vegetables",
        "curd/yogurt": "curd",
    }
    if key in aliases:
        match = aliases[key]
        return match, NUTRITION[match]
    raise HTTPException(
        status_code=404,
        detail=f"No nutrition record for '{name}'. Add a verified entry to the local nutrition dataset.",
    )

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

async def gemma_image_analysis(image_bytes: bytes, mime_type: str) -> list[dict[str, Any]] | None:
    """Analyze a meal image with Google's hosted Gemma 4 model.

    If GEMINI_API_KEY is missing, return None so the UI can show clearly labelled
    demo suggestions. Provider failures are surfaced as a useful 502 response.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it").strip() or "gemma-4-26b-a4b-it"
    prompt = (
        "Look at this meal photograph and identify only food items that are visibly likely. "
        "Return valid JSON only, with this exact shape: "
        '{"foods":[{"name":"common food name","suggested_grams":100,"confidence":"low|medium|high"}]}. '
        "Include up to 8 items. Suggested grams are rough estimates, not measurements; use 100 "
        "when portion size is unclear. Do not infer hidden ingredients or make medical claims."
    )

    try:
        # Google recommends the google-genai SDK for Gemma 4 hosted API access.
        from google import genai
        from google.genai import types
        import asyncio

        def request_model() -> str:
            client = genai.Client(api_key=api_key)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model=model,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            result_text = getattr(response, "text", None)
            if not result_text:
                raise ValueError("The model returned no text content.")
            return result_text

        text = await asyncio.to_thread(request_model)
        # Be tolerant of accidental Markdown fences even though JSON output is requested.
        cleaned_text = text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.split("\n", 1)[-1]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()
        parsed = json.loads(cleaned_text)
        found = parsed.get("foods", []) if isinstance(parsed, dict) else []
        if not isinstance(found, list):
            raise ValueError("The model response did not contain a foods list.")

        cleaned: list[dict[str, Any]] = []
        for item in found[:8]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()[:100]
            if not name:
                continue
            try:
                grams = float(item.get("suggested_grams", 100))
            except (TypeError, ValueError):
                grams = 100.0
            confidence = str(item.get("confidence", "low")).lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            cleaned.append({
                "name": name,
                "grams": min(max(grams, 1), 1000),
                "confidence": confidence,
            })
        if not cleaned:
            raise ValueError("The model did not identify any food items in a usable format.")
        return cleaned
    except Exception as exc:
        # Keep API keys and raw provider payloads out of the response to the browser.
        print(f"Gemma 4 image request failed ({type(exc).__name__}): {exc}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemma 4 image recognition failed. Check your API key, model access, "
                "internet connection, and server terminal for the error details."
            ),
        ) from exc

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "HiperMeal API",
        "ai_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "model": os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it"),
        "nutrition_records": len(NUTRITION),
    }

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type. Upload JPG, PNG, or WEBP.")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    # Verify the bytes actually represent an image.
    try:
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(contents)) as im:
            im.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode this image. Try another JPG, PNG, or WEBP file.") from exc

    analysis_id = str(uuid.uuid4())
    stored_name = f"{analysis_id}.{ {'image/jpeg':'jpg','image/png':'png','image/webp':'webp'}[file.content_type] }"
    # Uploaded photos are only stored temporarily for this demo. Do not expose them publicly.
    (UPLOAD_DIR / stored_name).write_bytes(contents)

    ai_foods = await gemma_image_analysis(contents, file.content_type)
    ai_enabled = ai_foods is not None
    if ai_foods is None:
        ai_foods = [
            {"name": "roti", "grams": 60, "confidence": "manual-review"},
            {"name": "dal", "grams": 150, "confidence": "manual-review"},
            {"name": "cooked rice", "grams": 120, "confidence": "manual-review"},
        ]

    foods = []
    for item in ai_foods:
        row = {"name": item["name"], "grams": item.get("grams", 100), "confidence": item.get("confidence", "low"), "nutrition": None}
        try:
            matched_name, nutrients = lookup_food(item["name"])
            row["matched_name"] = matched_name
            row["nutrition"] = nutrients
        except HTTPException:
            row["notes"] = "No local nutrition record; select a known food or add a verified dataset entry."
        foods.append(row)

    return {
        "analysis_id": analysis_id,
        "ai_enabled": ai_enabled,
        "message": (
            "AI suggestions generated. Food identity and portion size may be wrong; review each item."
            if ai_enabled else
            "Backend connected. AI vision is not configured because GEMINI_API_KEY is absent. Demo suggestions are shown for UI testing, not as analysis of your photo."
        ),
        "foods": foods,
    }

@app.post("/api/nutrition/lookup")
def nutrition_lookup(body: LookupInput) -> dict[str, Any]:
    matched_name, food = lookup_food(body.food_name)
    return {"matched_name": matched_name, "food": food, "basis": "per 100 g", "estimated": True}

@app.post("/api/meals")
def save_meal(body: MealInput) -> dict[str, Any]:
    # Recalculate on the server from known food data rather than trusting client totals.
    total = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    stored_foods = []
    for food in body.foods:
        matched_name, nutrients = lookup_food(food.name)
        multiplier = food.grams / 100
        for key in total:
            total[key] += nutrients[key] * multiplier
        stored_foods.append({
            "name": food.name.strip(),
            "matched_name": matched_name,
            "grams": food.grams,
            "nutrition_per_100g": nutrients,
        })
    meal_id = str(uuid.uuid4())
    created_at = utc_now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO meals(id,created_at,analysis_id,foods_json,totals_json) VALUES(?,?,?,?,?)",
            (meal_id, created_at, body.analysis_id, json.dumps(stored_foods), json.dumps(total)),
        )
        conn.commit()
    return {"id": meal_id, "created_at": created_at, "foods": stored_foods, "totals": total, "notice": "Nutrition values are estimates based on the prototype dataset."}

@app.get("/api/meals")
def list_meals() -> dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM meals ORDER BY created_at DESC LIMIT 100").fetchall()
    meals = []
    for row in rows:
        meals.append({
            "id": row["id"],
            "created_at": row["created_at"],
            "analysis_id": row["analysis_id"],
            "foods": json.loads(row["foods_json"]),
            "totals": json.loads(row["totals_json"]),
        })
    return {"meals": meals}

@app.delete("/api/meals/{meal_id}")
def delete_meal(meal_id: str) -> dict[str, bool]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM meals WHERE id=?", (meal_id,))
        conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Meal not found.")
    return {"deleted": True}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
