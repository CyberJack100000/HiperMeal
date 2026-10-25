"""Meal photo -> likely foods, using Google's hosted Gemma 4 model through the google-genai SDK.

Without GEMINI_API_KEY nothing is sent anywhere and `analyse` returns None, so the app can show clearly labelled demo suggestions."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

PROMPT = (
    "Look at this meal photograph and identify only food items that are visibly likely. "
    "Return valid JSON only, with this exact shape: "
    '{"foods":[{"name":"common food name","suggested_grams":100,"confidence":"low|medium|high"}]}. '
    "Include up to 8 items. Use short, common names (for example 'cooked rice', 'dal', 'roti'). Suggested grams are rough estimates, "
    "not measurements; use 100 when portion size is unclear. Do not infer hidden ingredients or make medical claims."
)
TIMEOUT_MS = 60_000


class AIError(Exception):
    """The model request failed or returned something unusable. The message is safe to show to the user."""


def configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def model_name() -> str:
    return (os.getenv("GEMMA_MODEL", "").strip() or "gemma-4-26b-a4b-it")


def parse(text: str) -> list[dict[str, Any]]:
    """Tolerant of Markdown fences and odd values; keeps only what the app can use."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    try:
        data = json.loads(t.strip())
    except ValueError as e:
        raise AIError("The model's answer was not valid JSON.") from e
    items = data.get("foods", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        raise AIError("The model's answer had no list of foods.")
    out, seen = [], set()
    for it in items[:8]:
        if not isinstance(it, dict):
            continue
        name = " ".join(str(it.get("name", "")).split())[:100]
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            grams = float(it.get("suggested_grams", 100))
        except (TypeError, ValueError):
            grams = 100.0
        if grams != grams:                                  # NaN
            grams = 100.0
        conf = str(it.get("confidence", "low")).lower()
        out.append({"name": name, "grams": min(max(grams, 1.0), 1000.0), "confidence": conf if conf in ("low", "medium", "high") else "low"})
    if not out:
        raise AIError("The model did not identify any foods in a usable format.")
    return out


def _request(jpeg: bytes) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip(), http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    resp = client.models.generate_content(
        model=model_name(),
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
    )
    text = getattr(resp, "text", None)
    if not text:
        raise AIError("The model returned no text.")
    return text


async def analyse(jpeg: bytes) -> list[dict[str, Any]] | None:
    if not configured():
        return None
    last: Exception | None = None
    for attempt in range(2):                                # one retry covers a brief network or quota blip
        try:
            return parse(await asyncio.to_thread(_request, jpeg))
        except AIError as e:
            last = e
        except Exception as e:                              # provider errors can contain request details; log them, never send them to the browser
            print(f"Gemma request failed ({type(e).__name__}): {e}")
            last = e
        if attempt == 0:
            await asyncio.sleep(1.0)
    raise AIError("Gemma 4 image recognition failed. Check the API key, model access and internet connection, and the server log for details.") from last
