"""Check your Gemma API key against the real service, once, before relying on it.

    python scripts/check_ai.py                 # uses GEMINI_API_KEY / GEMMA_MODEL from the environment or .env
    python scripts/check_ai.py meal.jpg        # analyse your own photo
    python scripts/check_ai.py --key KEY --model gemma-4-31b-it

It prints which Gemma models your key can see, which structured-output mode the API accepted (schema, json or plain), what the model made of a photo,
and a sample nutrition estimate. Nothing is saved. The key is never printed in full."""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402

env.load()
import ai  # noqa: E402


def sample_photo() -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 480), (245, 240, 230))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 40, 580, 440), fill=(255, 255, 255), outline=(200, 200, 200), width=6)
    d.ellipse((120, 110, 300, 290), fill=(240, 200, 120))
    d.ellipse((320, 120, 500, 280), fill=(255, 250, 240))
    d.ellipse((230, 260, 410, 400), fill=(180, 80, 40))
    b = io.BytesIO()
    im.save(b, "JPEG")
    return b.getvalue()


def list_models(key: str) -> list[str]:
    from google import genai
    client = genai.Client(api_key=key)                      # keep a reference: the client closes when it is garbage-collected, even mid-iteration
    return sorted(m.name.split("/")[-1] for m in client.models.list() if "gemma" in (m.name or "").lower())


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", nargs="?", help="a photo of a meal (default: a drawn sample, which a model may say has no food)")
    ap.add_argument("--key"), ap.add_argument("--model")
    a = ap.parse_args()
    key = ai.resolve_key(a.key)
    if not key:
        print("No key. Set GEMINI_API_KEY (or put it in .env), or pass --key.")
        return 1
    model = ai.model_name(a.model)
    print(f"Key: {key[:4]}…{key[-4:]}   Model: {model}")
    try:
        names = list_models(key)
        print(f"Gemma models this key can see ({len(names)}): {', '.join(names) or 'none'}")
        if names and model not in names:
            print(f"  ! “{model}” is not in that list. Set GEMMA_MODEL (or the Model box in Settings) to one that is.")
    except Exception as e:
        print(f"Could not list models ({type(e).__name__}); continuing. {ai.redact(str(e), key)[:200]}")
    print("\n1. Key check")
    try:
        print("  OK:", await ai.check(key, model))
    except ai.AIError as e:
        print("  FAILED:", e)
        return 2
    print("\n2. Photo analysis")
    jpeg = Path(a.photo).read_bytes() if a.photo else sample_photo()
    try:
        foods = await ai.analyse(jpeg, None, key, model)
        print(f"  structured-output mode that worked: {ai.mode()}")
        for f in foods or []:
            m = ai.nutrition.find(f["name"])
            print(f"  - {f['name']} ~{f['grams']:.0f} g ({f['confidence']}; {f['preparation'] or 'no preparation noted'}) -> {'table: ' + m.name + ' (' + m.how + ')' if m else 'NOT in table'}\n      {f['reasoning']}")
    except ai.AIError as e:
        print("  Result:", e)
    print("\n3. Nutrition estimate for “chicken tikka masala”")
    try:
        print("  ", await ai.estimate("chicken tikka masala", key, model))
    except ai.AIError as e:
        print("  Result:", e)
    print("\nDone. If step 2 listed sensible foods and step 3 gave plausible numbers, the integration works with your key.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
