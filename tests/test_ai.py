import asyncio

import pytest

import ai
import main
import nutrition
from tests.conftest import png, upload

KEY = "AIzaSyTestKeyTestKeyTestKeyTestKey1234"


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@pytest.fixture(autouse=True)
def fresh_mode(monkeypatch):
    monkeypatch.setattr(ai, "_mode", "schema")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMMA_MODEL", raising=False)


GOOD = '{"foods":[{"name":"dal","suggested_grams":150,"confidence":"high","preparation":"curry","reasoning":"A bowl of yellow lentil curry."}]}'


# ------------------------------------------------------------------------------------------------ defaults and prompts
def test_default_model_and_overrides(monkeypatch):
    assert ai.model_name() == "gemma-4-31b-it"
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4-test")
    assert ai.model_name() == "gemma-4-test" and ai.model_name("gemma-4-mine") == "gemma-4-mine"


def test_prompt_lists_known_foods_and_the_hint():
    p = ai.meal_prompt("indian")
    assert all(name in p for name in ("roti", "cooked rice", "chicken curry")) and "probably indian cuisine" in p
    assert "probably" not in ai.meal_prompt(None) and "probably" not in ai.meal_prompt("<script>")        # unknown hints are not copied into the prompt
    assert "JSON only" in p and "at most 20 words" in p


def test_key_resolution_prefers_the_persons_key(monkeypatch):
    assert not ai.configured() and ai.configured(KEY)
    monkeypatch.setenv("GEMINI_API_KEY", "server-key-server-key-1234")
    assert ai.configured() and ai.resolve_key() == "server-key-server-key-1234" and ai.resolve_key(KEY) == KEY and ai.server_key_set()


# ------------------------------------------------------------------------------------------------ parsing
def test_parse_reads_reasoning_and_preparation_and_tolerates_prose():
    f = ai.parse("Here you go:\n" + GOOD + "\nHope that helps!")[0]
    assert f == {"name": "dal", "grams": 150.0, "confidence": "high", "preparation": "curry", "reasoning": "A bowl of yellow lentil curry."}
    f = ai.parse('{"foods":[{"name":"rice","suggested_grams":"x","confidence":"??","preparation":null,"reasoning":"a" ,"extra":1}]}')[0]
    assert f["grams"] == 100 and f["confidence"] == "low" and f["preparation"] == ""
    long = ai.parse('{"foods":[{"name":"x","reasoning":"' + "w" * 500 + '\\u0000"}]}')[0]
    assert len(long["reasoning"]) <= 160 and "\x00" not in long["reasoning"]


def test_no_food_gives_a_friendly_message():
    with pytest.raises(ai.AIError, match="did not see any food"):
        ai.parse('{"foods": []}')


def test_estimates_are_checked_for_physical_sense():
    ok = ai.parse_estimate('{"kcal":165,"protein_g":31,"carbs_g":0,"fat_g":3.6,"note":"grilled"}')
    assert ok["kcal"] == 165 and ok["note"] == "grilled"
    for bad in ('{"kcal":100,"protein_g":50,"carbs_g":50,"fat_g":30,"note":""}',            # more than 100 g of macros in 100 g
                '{"kcal":900,"protein_g":1,"carbs_g":1,"fat_g":1,"note":""}',               # calories do not match the macros
                '{"kcal":-5,"protein_g":1,"carbs_g":1,"fat_g":1}', '{"kcal":100}', "[]", "nope"):
        with pytest.raises(ai.AIError):
            ai.parse_estimate(bad)


# ------------------------------------------------------------------------------------------------ the strict-to-plain ladder
def run_with(monkeypatch, behaviours):
    seen = []

    def fake_call(contents, mode_, schema, key=None, model=None):
        seen.append((mode_, key, model))
        b = behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return b
    monkeypatch.setattr(ai, "_call", fake_call)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0))                 # do not really wait between retries
    return seen


def test_steps_down_when_the_api_rejects_a_structured_mode_and_remembers(monkeypatch):
    seen = run_with(monkeypatch, [ApiError(400, "JSON mode is not enabled for models/gemma"), ApiError(400, "response_schema is not supported"), GOOD, GOOD])
    assert asyncio.run(ai._generate(["x"], ai.MealResult, KEY, None)) == GOOD
    assert [m for m, *_ in seen] == ["schema", "json", "plain"] and ai.mode() == "plain"
    asyncio.run(ai._generate(["x"], ai.MealResult, KEY, None))
    assert seen[-1][0] == "plain"                                                # the next request does not retry the failed modes


def test_other_400s_do_not_downgrade_and_become_clear_messages(monkeypatch):
    run_with(monkeypatch, [ApiError(400, "API key not valid. Please pass a valid API key.")])
    with pytest.raises(ai.AIError, match="rejected the API key") as e:
        asyncio.run(ai._generate(["x"], ai.MealResult, KEY, None))
    assert ai.mode() == "schema" and KEY not in str(e.value)
    run_with(monkeypatch, [ApiError(404, "models/zzz is not found")])
    with pytest.raises(ai.AIError, match="does not know the model “gemma-4-zzz”"):
        asyncio.run(ai._generate(["x"], ai.MealResult, KEY, "gemma-4-zzz"))
    run_with(monkeypatch, [ApiError(429, "quota"), ApiError(429, "quota")])
    with pytest.raises(ai.AIError, match="out of quota") as e:                    # a 429 is retried once first, then explained
        asyncio.run(ai._generate(["x"], ai.MealResult, KEY, None))


def test_transient_failures_retry_once(monkeypatch):
    seen = run_with(monkeypatch, [ApiError(503, "overloaded"), GOOD])
    assert asyncio.run(ai._generate(["x"], ai.MealResult, KEY, None)) == GOOD and len(seen) == 2
    seen = run_with(monkeypatch, [ApiError(503, "x"), ApiError(503, "x")])
    with pytest.raises(ai.AIError):
        asyncio.run(ai._generate(["x"], ai.MealResult, KEY, None))
    assert len(seen) == 2


def test_key_never_reaches_logs(monkeypatch, capsys):
    run_with(monkeypatch, [ApiError(400, f"bad request for key={KEY}")])
    with pytest.raises(ai.AIError):
        asyncio.run(ai._generate(["x"], ai.MealResult, KEY, None))
    out = capsys.readouterr().out
    assert KEY not in out and "***" in out
    assert ai.redact(f"a {KEY} b", KEY) == "a *** b" and ai.redact("nothing", None) == "nothing"


# ------------------------------------------------------------------------------------------------ the API with a person's own key
def test_header_key_and_model_reach_the_model_and_are_never_echoed(client, monkeypatch):
    got = {}

    async def fake(jpeg, hint=None, key=None, model=None):
        got.update(key=key, model=model, hint=hint)
        return [{"name": "dal", "grams": 150, "confidence": "high", "preparation": "curry", "reasoning": "yellow lentils"}]
    monkeypatch.setattr(ai, "analyse", fake)
    r = client.post("/api/analyze", files={"file": ("m.png", png(), "image/png")}, data={"hint": "Indian"}, headers={"X-Gemini-Key": KEY, "X-Gemma-Model": "gemma-4-mine"})
    d = r.json()
    assert r.status_code == 200 and got == {"key": KEY, "model": "gemma-4-mine", "hint": "indian"} and KEY not in r.text
    assert d["foods"][0]["reasoning"] == "yellow lentils" and d["foods"][0]["preparation"] == "curry" and d["foods"][0]["matched_name"] == "dal"
    h = client.get("/api/health")
    assert h.json()["ai_configured"] is False and KEY not in h.text                        # a person's key never changes what the server reports about itself


@pytest.mark.parametrize("hdr", [{"X-Gemini-Key": "short"}, {"X-Gemini-Key": "has space inside the key value 12345"}, {"X-Gemini-Key": "k" * 300},
                                 {"X-Gemma-Model": "../etc"}, {"X-Gemma-Model": "x"}, {"X-Gemma-Model": "a b"}])
def test_odd_key_or_model_headers_are_refused(client, hdr):
    hdr = {"X-Gemini-Key": KEY, **hdr}
    for path, kw in (("/api/analyze", {"files": {"file": ("m.png", png(), "image/png")}}), ("/api/ai/test", {}), ("/api/nutrition/estimate", {"json": {"food_name": "x"}})):
        r = client.post(path, headers=hdr, **kw)
        assert r.status_code == 422 and hdr.get("X-Gemini-Key", "") not in r.text, path


def test_bad_cuisine_hint_is_refused(client):
    assert client.post("/api/analyze", files={"file": ("m.png", png(), "image/png")}, data={"hint": "x" * 50}).status_code == 422


def test_without_any_key_the_demo_is_labelled_and_points_to_settings(client):
    d = upload(client, png()).json()
    assert d["ai_enabled"] is False and "Settings" in d["message"]


def test_key_check_endpoint(client, monkeypatch):
    r = client.post("/api/ai/test")
    assert r.status_code == 502 and "No API key" in r.json()["detail"]

    def fake_call(contents, mode_, schema, key=None, model=None):
        assert key == KEY and model is None and mode_ == "plain"          # the default model is filled in by the module, not the endpoint
        return "OK\n"
    monkeypatch.setattr(ai, "_call", fake_call)
    r = client.post("/api/ai/test", headers={"X-Gemini-Key": KEY})
    assert r.status_code == 200 and r.json() == {"ok": True, "model": "gemma-4-31b-it", "reply": "OK"}

    def bad_call(*a, **k):
        raise ApiError(403, f"permission denied for {KEY}")
    monkeypatch.setattr(ai, "_call", bad_call)
    r = client.post("/api/ai/test", headers={"X-Gemini-Key": KEY})
    assert r.status_code == 502 and "rejected the API key" in r.json()["detail"] and KEY not in r.text


def test_estimate_endpoint(client, monkeypatch):
    r = client.post("/api/nutrition/estimate", json={"food_name": "lamb vindaloo"})
    assert r.status_code == 503 and "Settings" in r.json()["detail"]
    monkeypatch.setattr(ai, "_call", lambda c, m, s, key=None, model=None: '{"kcal":170,"protein_g":14,"carbs_g":6,"fat_g":10,"note":"with some gravy"}')
    r = client.post("/api/nutrition/estimate", json={"food_name": "  lamb   vindaloo "}, headers={"X-Gemini-Key": KEY})
    d = r.json()
    assert r.status_code == 200 and d["nutrition"] == {"kcal": 170, "protein_g": 14, "carbs_g": 6, "fat_g": 10} and d["estimated_by"] == "ai" and d["note"] == "with some gravy"
    monkeypatch.setattr(ai, "_call", lambda c, m, s, key=None, model=None: '{"kcal":900,"protein_g":1,"carbs_g":1,"fat_g":1,"note":""}')
    r = client.post("/api/nutrition/estimate", json={"food_name": "x"}, headers={"X-Gemini-Key": KEY})
    assert r.status_code == 502 and "inconsistent" in r.json()["detail"]
    assert client.post("/api/nutrition/estimate", json={"food_name": ""}, headers={"X-Gemini-Key": KEY}).status_code == 422


def test_estimates_have_their_own_rate_limit(client, monkeypatch):
    monkeypatch.setattr(ai, "_call", lambda c, m, s, key=None, model=None: '{"kcal":100,"protein_g":5,"carbs_g":15,"fat_g":2,"note":""}')
    codes = [client.post("/api/nutrition/estimate", json={"food_name": "x"}, headers={"X-Gemini-Key": KEY}).status_code for _ in range(32)]
    assert codes[:30] == [200] * 30 and codes[30:] == [429, 429]
    assert upload(client, png()).status_code == 200                                    # photo analysis is a separate allowance
