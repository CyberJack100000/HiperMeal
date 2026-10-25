"""Gemma 4 (Google's hosted API, google-genai SDK): meal photo -> foods with portions, and nutrition estimates for foods the table does not know.

Without GEMINI_API_KEY nothing is sent anywhere and `analyse` returns None, so the app can show clearly labelled demo suggestions.

How the request is made, and why:
- One carefully structured prompt does the whole job (look, name, size, describe preparation) and the answer comes back in a fixed JSON shape.
- Hosted Gemma models have not always accepted every structured-output feature, so the request starts at the strictest mode (a JSON schema) and, only when the
  API says that mode is not supported, steps down to plain JSON mode and then to prompt-only JSON. The mode that works is remembered for the process.
  (`python scripts/check_ai.py` shows which one your key gets.)
- Foods the app already knows are listed in the prompt, so the model reuses those exact names where they fit, which raises the share of foods that match
  the nutrition table without any guessing on our side.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field

import nutrition

TIMEOUT_MS = 60_000
MODES = ("schema", "json", "plain")           # strictest first
_mode = "schema"
CUISINES = ("indian", "chinese", "italian", "mexican", "middle eastern", "japanese", "western")


class AIError(Exception):
    """The model request failed or returned something unusable. The message is safe to show to the user. `status` is Google's HTTP status when there was one."""
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# Which model actually answered the current request (set by analyse/estimate; read by the endpoint right after). None when no model was called.
served_by: contextvars.ContextVar[str | None] = contextvars.ContextVar("served_by", default=None)
DEFAULT_BACKUP = "gemma-4-26b-a4b-it"
OVERLOADED = (500, 503, 504)
RETRY_DELAYS = {"server": (1.5, 4.0, 9.0), "quota": (3.0,), "network": (1.5, 4.0)}      # seconds to wait before each retry, by kind of failure
DEADLINE_S = 75.0                                                                        # stop retrying after this long overall


class FoodItem(BaseModel):
    name: str = Field(description="Short common name of one food on the plate")
    suggested_grams: float = Field(description="Estimated weight in grams of the cooked or served portion visible")
    confidence: str = Field(description="low, medium or high")
    preparation: str = Field(description="How it looks prepared, e.g. boiled, fried, curry, raw, grilled; empty if unclear")
    reasoning: str = Field(description="One short sentence (at most 20 words): what you saw and how you judged the portion")


class MealResult(BaseModel):
    foods: list[FoodItem]


class Estimate(BaseModel):
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    note: str


def resolve_key(key: str | None = None) -> str:
    """A key the person entered in the app wins; otherwise the one in the server's environment."""
    return (key or os.getenv("GEMINI_API_KEY", "")).strip()


def configured(key: str | None = None) -> bool:
    return bool(resolve_key(key))


def server_key_set() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def model_name(model: str | None = None) -> str:
    return (model or os.getenv("GEMMA_MODEL", "")).strip() or "gemma-4-31b-it"


def backup_model(backup: str | None = None) -> str | None:
    """The model to try when the main one stays overloaded. `none`/`off`/empty turns it off."""
    v = (backup if backup is not None else os.getenv("GEMMA_BACKUP_MODEL", DEFAULT_BACKUP)).strip()
    return None if v.lower() in ("", "none", "off") else v


VALID_KEY = re.compile(r"^[\x21-\x7e]{16,256}$")                 # printable ASCII, no spaces
VALID_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")


def mode() -> str:
    return _mode


# ----------------------------------------------------------------------------------------------------------- prompts
def meal_prompt(hint: str | None = None) -> str:
    known = ", ".join(sorted(nutrition.NUTRITION))
    cuisine = f"The meal is probably {hint} cuisine. " if hint in CUISINES else ""
    return (
        "You are a careful food-photo analyst for a nutrition diary. " + cuisine +
        "Work in this order. (1) Find every distinct food item on the plate or in the bowl; ignore packaging, cutlery, drinks you cannot see clearly and the background. "
        "(2) Give each item a short, common name. When an item is one of these known foods, use that exact name: " + known + ". "
        "Otherwise use the plain common name (for example 'chicken tikka masala'). Keep different foods separate; never merge a whole plate into one item, and do not "
        "invent hidden ingredients. (3) Estimate the weight in grams of the portion you can see, using the plate, bowl, cutlery and hands for scale; typical single servings "
        "are 50 to 300 g, so use 100 when you truly cannot tell. (4) Say how it is prepared if visible (fried, boiled, grilled, curry, raw). (5) Rate your confidence low, "
        "medium or high for that item. (6) Add one short sentence of reasoning (at most 20 words). Include at most 8 items. If the photo has no food, return an empty list. "
        'Answer with JSON only, in exactly this shape: {"foods":[{"name":"...","suggested_grams":100,"confidence":"low|medium|high","preparation":"...","reasoning":"..."}]}. '
        "Do not give medical or dietary advice."
    )


def estimate_prompt(name: str) -> str:
    return (
        f'Give typical nutrition values per 100 g of "{name}" as it is usually eaten (cooked or prepared where that applies). '
        "Round sensibly and be physically consistent: protein, carbs and fat together cannot exceed 100 g, and kcal should be close to 4 x protein + 4 x carbs + 9 x fat. "
        'Answer with JSON only: {"kcal":0,"protein_g":0,"carbs_g":0,"fat_g":0,"note":"one short sentence on what you assumed"}. '
        "This is a rough estimate for a food diary, not medical advice."
    )


# ----------------------------------------------------------------------------------------------------------- parsing
def _json_from(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    try:
        return json.loads(t)
    except ValueError:
        a, b = t.find("{"), t.rfind("}")                    # prompt-only mode: the model may wrap the JSON in a sentence
        if a != -1 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except ValueError:
                pass
    raise AIError("The model's answer was not valid JSON.")


def _clean_text(v: Any, limit: int) -> str:
    return " ".join("".join(c for c in str(v or "") if c.isprintable()).split())[:limit]


def parse(text: str) -> list[dict[str, Any]]:
    """Tolerant of fences, extra prose and odd values; keeps only what the app can use."""
    data = _json_from(text)
    items = data.get("foods", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        raise AIError("The model's answer had no list of foods.")
    out, seen = [], set()
    for it in items[:8]:
        if not isinstance(it, dict):
            continue
        name = _clean_text(it.get("name"), 100)
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
        out.append({"name": name, "grams": min(max(grams, 1.0), 1000.0), "confidence": conf if conf in ("low", "medium", "high") else "low",
                    "preparation": _clean_text(it.get("preparation"), 40), "reasoning": _clean_text(it.get("reasoning"), 160)})
    if not out:
        raise AIError("The AI did not see any food in this photo. Try a clearer, closer photo, or add the foods yourself.")
    return out


def parse_estimate(text: str) -> dict[str, Any]:
    d = _json_from(text)
    if not isinstance(d, dict):
        raise AIError("The model's answer was not usable.")
    try:
        v = {k: float(d[k]) for k in nutrition.KEYS}
    except (KeyError, TypeError, ValueError):
        raise AIError("The model's estimate was incomplete.")
    if any(x != x or x < 0 for x in v.values()) or v["kcal"] > 900 or any(v[k] > 100 for k in ("protein_g", "carbs_g", "fat_g")) or v["protein_g"] + v["carbs_g"] + v["fat_g"] > 100.5:
        raise AIError("The model's estimate was not physically possible, so it was discarded. Type the values yourself.")
    atwater = 4 * v["protein_g"] + 4 * v["carbs_g"] + 9 * v["fat_g"]
    if abs(atwater - v["kcal"]) > max(60.0, 0.35 * v["kcal"]):
        raise AIError("The model's estimate was inconsistent (calories do not match the macros), so it was discarded. Type the values yourself.")
    return {**{k: round(x, 1) for k, x in v.items()}, "note": _clean_text(d.get("note"), 160)}


# ----------------------------------------------------------------------------------------------------------- the request
def _status_code(e: Exception) -> int | None:
    for attr in ("code", "status_code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


_UNSUPPORTED = ("json mode", "mime", "schema", "response_", "controlled generation", "not enabled for", "not supported for this model")


def is_key_problem(e: Exception) -> bool:
    text = str(e).lower()
    return _status_code(e) in (401, 403) or (_status_code(e) == 400 and ("api key" in text or "api_key" in text))


def unsupported_mode(e: Exception) -> bool:
    """A 400 that is about the structured-output feature itself (not about the image, the key or a quota)."""
    return _status_code(e) == 400 and any(w in str(e).lower() for w in _UNSUPPORTED)


def soft_400(e: Exception) -> bool:
    """Any other 400 that is not about the key. In a structured mode it is worth trying a simpler mode for this one request, because hosted models differ in
    which options they accept and the error text is not always recognisable."""
    return _status_code(e) == 400 and not is_key_problem(e)


def retry_kind(e: Exception) -> str | None:
    """What sort of temporary failure this is (so it gets the right waits), or None when retrying cannot help."""
    c = _status_code(e)
    if c is None:                                           # no HTTP status: only network-type failures are worth a retry, not programming errors
        net = isinstance(e, (OSError, TimeoutError)) or any(w in type(e).__name__.lower() for w in ("timeout", "connect", "network", "protocol"))
        return "network" if net else None
    if c == 429:
        return "quota"
    return "server" if c >= 500 else None


def transient(e: Exception) -> bool:
    return retry_kind(e) is not None


def redact(msg: str, key: str | None) -> str:
    k = resolve_key(key)
    return msg.replace(k, "***") if k else msg


def provider_reason(e: Exception, key: str | None = None) -> str:
    """Google's own one-sentence explanation (the `message` of its error), cleaned: key removed, control characters dropped, shortened."""
    msg = getattr(e, "message", None)
    if not isinstance(msg, str) or not msg.strip():
        text = str(e)
        m = re.search(r"'message':\s*'([^']{3,400})'", text) or re.search(r'"message":\s*"([^"]{3,400})"', text)
        msg = m.group(1) if m else ""
    return _clean_text(redact(msg, key), 220)


def friendly(e: Exception, model: str, key: str | None = None) -> str:
    """What the person should be told: a plain-language cause where it is known, otherwise Google's own explanation. Never contains the key."""
    code = _status_code(e)
    if is_key_problem(e):
        return "Google rejected the API key (or this key cannot use that model). Check the key and model in Settings."
    if code == 404:
        return f"Google does not know the model “{model}” for this key. Pick another Gemma 4 model in Settings."
    if code == 429:
        return "Google says this key is out of quota or is being rate-limited. Wait a bit and try again."
    if isinstance(e, ImportError):
        return "The google-genai package is not installed on the server. Run: python -m pip install -r requirements.txt"
    if code in OVERLOADED:
        why = provider_reason(e, key)
        return f"Google's {model} is busy or having a problem right now (HTTP {code}{': ' + why if why else ''}). This is usually temporary: try again in a minute."
    why = provider_reason(e, key)
    where = f"HTTP {code}" if code else type(e).__name__
    return f"Gemma 4 request failed ({where}{': ' + why if why else ''}). The server log has the full details."


def _call(contents: list, mode_: str, schema: type[BaseModel], key: str | None = None, model: str | None = None) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=resolve_key(key), http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    cfg: dict[str, Any] = {"temperature": 0.1}
    if mode_ in ("schema", "json"):
        cfg["response_mime_type"] = "application/json"
    if mode_ == "schema":
        cfg["response_schema"] = schema
    resp = client.models.generate_content(model=model_name(model), contents=contents, config=types.GenerateContentConfig(**cfg))
    text = getattr(resp, "text", None)
    if not text:
        raise AIError("The model returned no text.")
    return text


async def _generate(contents: list, schema: type[BaseModel], key: str | None = None, model: str | None = None) -> str:
    """Run a request at the strictest mode that works. If the API rejects a mode (explicitly, or with some other 400), try the next simpler one for this request;
    what finally works is remembered. Temporary failures are retried with growing waits (a busy model often recovers within seconds). A mode that only failed
    because of something unrelated (a bad key, say) is never remembered."""
    global _mode
    i, last, used, t0 = MODES.index(_mode), None, {}, time.monotonic()
    while i < len(MODES):
        mode_ = MODES[i]
        try:
            text = await asyncio.to_thread(_call, contents, mode_, schema, key, model)
            if mode_ != _mode:
                print(f"Gemma 4 works in {mode_} mode here; using it from now on.")
                _mode = mode_
            return text
        except AIError:
            raise
        except Exception as e:                              # provider errors can contain request details: log them (key removed), never send them raw to the browser
            last = e
            print(f"Gemma request failed in {mode_} mode ({type(e).__name__}): {redact(str(e), key)}")
            if (unsupported_mode(e) or soft_400(e)) and i < len(MODES) - 1:
                i += 1
                print(f"Trying {MODES[i]} mode for this request.")
                continue
            kind = retry_kind(e)
            n = used.get(kind, 0) if kind else 0
            if kind and n < len(RETRY_DELAYS[kind]) and time.monotonic() - t0 < DEADLINE_S:
                used[kind] = n + 1
                delay = RETRY_DELAYS[kind][n]
                print(f"Temporary problem ({kind}); retry {n + 1} in {delay:.1f}s.")
                await asyncio.sleep(delay)
                continue
            break
    raise AIError(friendly(last, model_name(model), key), status=_status_code(last)) from last


async def _with_backup(contents: list, schema: type[BaseModel], key: str | None, model: str | None, backup: str | None) -> str:
    """Ask the main model; if it stays overloaded after the retries, ask the backup model once. Records which one answered in `served_by`."""
    global _mode
    primary = model_name(model)
    try:
        text, used = await _generate(contents, schema, key, model), primary
    except AIError as e:
        b = backup_model(backup)
        if e.status in OVERLOADED and not b:
            raise AIError(f"{e} You can choose a backup model in Settings.", status=e.status) from e
        if e.status not in OVERLOADED or b == primary:
            raise
        print(f"{primary} is overloaded; trying the backup model {b}.")
        saved = _mode
        try:
            text, used = await _generate(contents, schema, key, b), b
        except AIError:
            raise e                                         # report the main model's problem, not the backup's
        finally:
            _mode = saved                                   # whatever the backup needed must not change how the main model is called
    served_by.set(used)
    return text


async def analyse(jpeg: bytes, hint: str | None = None, key: str | None = None, model: str | None = None, backup: str | None = None) -> list[dict[str, Any]] | None:
    if not configured(key):
        return None
    from google.genai import types
    text = await _with_backup([types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), meal_prompt(hint)], MealResult, key, model, backup)
    return parse(text)


async def estimate(name: str, key: str | None = None, model: str | None = None, backup: str | None = None) -> dict[str, Any]:
    if not configured(key):
        raise AIError("No API key is set. Add one in Settings.")
    return parse_estimate(await _with_backup([estimate_prompt(name)], Estimate, key, model, backup))


async def check(key: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Make one tiny request so the person can confirm a key and model work before sending a photo."""
    if not configured(key):
        raise AIError("No API key is set. Add one in Settings.")
    last: Exception | None = None
    text = ""
    try:
        text = await asyncio.to_thread(_call, ["Reply with the single word OK."], "plain", Estimate, key, model)
    except Exception as e:
        last = e
        print(f"Gemma key check failed ({type(e).__name__}): {redact(str(e), key)}")
    if last is not None:
        raise AIError(friendly(last, model_name(model), key)) from last
    return {"ok": True, "model": model_name(model), "reply": _clean_text(text, 40)}
