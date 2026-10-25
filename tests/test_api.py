import io
import sqlite3
import time
from datetime import date, timedelta

import pytest
from PIL import Image

import ai
import main
import nutrition
import store
from tests.conftest import jpeg_with_gps, png, upload

UNKNOWN = {"kcal": 120, "protein_g": 5, "carbs_g": 20, "fat_g": 2}


# ------------------------------------------------------------------------------------------------------ nutrition matching
@pytest.mark.parametrize("name,expect,how", [
    ("roti", "roti", "exact"), ("Chapati", "roti", "alias"), ("  COOKED   Rice ", "cooked rice", "exact"), ("boiled white rice", "white rice", "variant"),
    ("a bowl of dal", "dal", "variant"), ("Eggs", "egg", "alias"), ("bananas", "banana", "variant"), ("chapathi", "roti", "close"),
    ("Dal (yellow lentils)", "dal", "exact"),
])
def test_matches_that_are_the_same_food(name, expect, how):
    m = nutrition.find(name)
    assert m and m.name == expect and m.how == how


@pytest.mark.parametrize("name", ["fried egg", "lamb vindaloo", "rice pudding", "chicken tikka masala", "", "123", "banana split sundae"])
def test_foods_that_are_not_clearly_known_are_not_guessed(name):
    assert nutrition.find(name) is None            # a wrong match would silently attach the wrong calories


def test_every_table_entry_is_physically_plausible():
    for name, n in nutrition.NUTRITION.items():
        assert n["protein_g"] + n["carbs_g"] + n["fat_g"] <= 100.5, name
        assert 0 <= n["kcal"] <= 900, name
        atwater = 4 * n["protein_g"] + 4 * n["carbs_g"] + 9 * n["fat_g"]
        assert abs(atwater - n["kcal"]) <= max(60, 0.35 * n["kcal"]), (name, atwater, n["kcal"])      # calories roughly agree with the macros
    for alias, target in nutrition.ALIASES.items():
        assert target in nutrition.NUTRITION, alias


# ------------------------------------------------------------------------------------------------------ analyze
def test_demo_mode_is_clearly_labelled(client):
    r = upload(client, png())
    d = r.json()
    assert r.status_code == 200 and d["ai_enabled"] is False and "not an analysis of your photo" in d["message"]
    assert [f["name"] for f in d["foods"]] == ["roti", "dal", "cooked rice"] and all(f["nutrition"] for f in d["foods"])
    assert client.get("/api/health").json()["ai_configured"] is False


def test_ai_suggestions_are_matched_and_unknowns_are_flagged(client, monkeypatch):
    async def fake(jpeg):
        return [{"name": "chapati", "grams": 70, "confidence": "high"}, {"name": "lamb vindaloo", "grams": 200, "confidence": "low"}]
    monkeypatch.setattr(ai, "analyse", fake)
    d = upload(client, png()).json()
    assert d["ai_enabled"] is True
    a, b = d["foods"]
    assert a["matched_name"] == "roti" and a["match"] == "alias" and a["nutrition"]["kcal"] == 297
    assert b["nutrition"] is None and "No nutrition record" in b["notes"]


def test_ai_failure_is_a_502_with_no_provider_details(client, monkeypatch):
    async def boom(jpeg):
        raise ai.AIError("Gemma 4 image recognition failed. Check the API key.")
    monkeypatch.setattr(ai, "analyse", boom)
    r = upload(client, png())
    assert r.status_code == 502 and "key" in r.json()["detail"].lower()


def test_ai_answer_parsing_is_tolerant_and_strict():
    good = ai.parse('```json\n{"foods":[{"name":" Dal ","suggested_grams":"150","confidence":"HIGH"},{"name":"dal"},{"name":"","suggested_grams":5},'
                    '{"name":"rice","suggested_grams":"lots"},{"name":"x","suggested_grams":99999}]}\n```')
    assert [f["name"] for f in good] == ["Dal", "rice", "x"]
    assert good[0]["grams"] == 150 and good[0]["confidence"] == "high" and good[1]["grams"] == 100 and good[2]["grams"] == 1000
    for bad in ("not json", '{"foods": "none"}', '{"foods": []}', "[]", ""):
        with pytest.raises(ai.AIError):
            ai.parse(bad)


@pytest.mark.parametrize("data,ctype,status", [(b"", "image/png", 400), (b"hello", "image/png", 400), (png(), "text/plain", 415), (png(), "image/gif", 415)])
def test_bad_uploads_are_refused(client, data, ctype, status):
    assert upload(client, data, ctype=ctype).status_code == status


def test_too_large_upload(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 1000)
    assert upload(client, png() + b'0' * 2000).status_code == 413


def test_photo_is_cleaned_before_it_is_stored(client):
    d = upload(client, jpeg_with_gps(), "p.jpg", "image/jpeg").json()
    saved = store.upload_path(d["analysis_id"]).read_bytes()
    im = Image.open(io.BytesIO(saved))
    assert im.format == "JPEG" and not im.getexif() and b"TestPhone" not in saved           # location and device details are gone
    big = upload(client, png((4000, 3000))).json()
    assert max(Image.open(store.upload_path(big["analysis_id"])).size) == main.MAX_SIDE      # shrunk before sending anywhere
    assert Image.open(io.BytesIO(main.prepare_image(png((40, 40), (0, 0, 0, 0), "RGBA")))).convert("RGB").getpixel((5, 5)) == (255, 255, 255)   # transparency becomes white, not black


def test_analyze_is_rate_limited(client):
    codes = [upload(client, png()).status_code for _ in range(22)]
    assert codes[:20] == [200] * 20 and codes[20:] == [429, 429]


# ------------------------------------------------------------------------------------------------------ meals
def meal(client, foods=None, **kw):
    body = {"foods": foods or [{"name": "roti", "grams": 100}], **kw}
    return client.post("/api/meals", json=body)


def test_server_recalculates_totals_and_ignores_the_browser(client):
    r = meal(client, [{"name": "roti", "grams": 200}, {"name": "dal", "grams": 100}], totals={"kcal": 1, "protein_g": 1, "carbs_g": 1, "fat_g": 1})
    t = r.json()["totals"]
    assert r.status_code == 200 and round(t["kcal"]) == 2 * 297 + 116 and round(t["protein_g"], 1) == round(2 * 8.2 + 7.6, 1)


def test_a_food_the_table_does_not_know_no_longer_blocks_saving(client):
    assert meal(client, [{"name": "lamb vindaloo", "grams": 150}]).status_code == 422                      # asks for values instead of guessing
    r = meal(client, [{"name": "roti", "grams": 60}, {"name": "lamb vindaloo", "grams": 150, "custom": UNKNOWN}])
    d = r.json()
    assert r.status_code == 200 and round(d["totals"]["kcal"]) == round(297 * .6 + 120 * 1.5)
    assert d["foods"][1]["source"] == "user" and d["foods"][1]["matched_name"] is None and d["foods"][0]["source"] == "table"


@pytest.mark.parametrize("bad", [
    {"foods": []}, {"foods": [{"name": "", "grams": 10}]}, {"foods": [{"name": "x", "grams": 0}]}, {"foods": [{"name": "x", "grams": 99999}]},
    {"foods": [{"name": "roti", "grams": 10, "custom": {**UNKNOWN, "carbs_g": 90, "fat_g": 30}}]},            # macros over 100 g
    {"foods": [{"name": "roti", "grams": 10, "custom": {**UNKNOWN, "kcal": -1}}]},
    {"foods": [{"name": "roti", "grams": 10}], "meal_type": "brunch"}, {"foods": [{"name": "roti", "grams": 10}], "local_date": "31-01-2026"},
    {"foods": [{"name": "roti", "grams": 10}], "analysis_id": "../../etc/passwd"}, {"foods": [{"name": "roti", "grams": 10}], "note": "x" * 301},
])
def test_invalid_meals_are_rejected(client, bad):
    assert client.post("/api/meals", json=bad).status_code == 422


def test_photo_follows_the_meal_and_goes_with_it(client):
    a = upload(client, png()).json()["analysis_id"]
    m = meal(client, analysis_id=a, meal_type="lunch", note="at work").json()
    assert m["image_url"] == f"/api/meals/{m['id']}/image" and m["meal_type"] == "lunch" and m["note"] == "at work"
    r = client.get(m["image_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and Image.open(io.BytesIO(r.content)).format == "JPEG"
    assert client.get(f"/api/meals").json()["meals"][0]["image_url"] == m["image_url"]
    assert client.delete(f"/api/meals/{m['id']}").json() == {"deleted": True}
    assert client.get(m["image_url"]).status_code == 404 and not store.upload_path(a).exists()              # the file is deleted too
    assert client.delete(f"/api/meals/{m['id']}").status_code == 404
    assert meal(client, analysis_id="00000000-0000-0000-0000-000000000000").json()["image_url"] is None     # unknown photo id: saved without a photo


@pytest.mark.parametrize("path", ["/api/meals/..%2F..%2Fx/image", "/api/meals/not-a-uuid/image", "/api/meals/12345/image"])
def test_ids_in_paths_must_be_real_uuids(client, path):
    assert client.get(path).status_code in (404, 422)
    assert client.delete(path.replace("/image", "")).status_code in (404, 422)


def test_unsaved_photos_are_cleaned_up_but_saved_ones_stay(client):
    keep = upload(client, png()).json()["analysis_id"]
    drop = upload(client, png()).json()["analysis_id"]
    meal(client, analysis_id=keep)
    old = time.time() - 3 * 86400
    for a in (keep, drop):
        import os
        os.utime(store.upload_path(a), (old, old))
    assert store.purge_orphan_photos(24) == 1
    assert store.upload_path(keep).exists() and not store.upload_path(drop).exists()


# ------------------------------------------------------------------------------------------------------ summary, settings, privacy
def test_todays_summary_only_counts_today(client):
    today = date(2026, 3, 10)
    for offset, grams in ((0, 100), (0, 100), (1, 100), (30, 100)):
        meal(client, [{"name": "roti", "grams": grams}], local_date=(today - timedelta(days=offset)).isoformat())
    s = client.get("/api/summary", params={"days": 7, "today": today.isoformat()}).json()
    assert round(s["today"]["kcal"]) == 2 * 297 and s["today"]["meals"] == 2          # the old code added up every meal ever saved
    assert [d["meals"] for d in s["days"]] == [0, 0, 0, 0, 0, 1, 2] and s["days"][-1]["date"] == "2026-03-10" and s["logged_days"] == 2
    assert round(s["average_per_logged_day"]["kcal"]) == round((297 + 594) / 2)
    assert s["top_foods"][0] == {"name": "roti", "times": 3, "grams": 300}
    assert len(client.get("/api/meals", params={"days": 7, "today": "2026-03-10"}).json()["meals"]) == 3
    assert client.get("/api/summary", params={"today": "nope"}).status_code == 422 and client.get("/api/summary", params={"days": 0}).status_code == 422


def test_targets_are_saved_and_validated(client):
    assert client.get("/api/settings").json()["targets"]["kcal"] == 2000
    ok = {"kcal": 1800, "protein_g": 90, "carbs_g": 200, "fat_g": 60}
    assert client.put("/api/settings", json=ok).json()["targets"] == ok and client.get("/api/settings").json()["targets"] == ok
    assert client.get("/api/summary").json()["targets"] == ok
    for bad in ({**ok, "kcal": 10}, {**ok, "protein_g": -1}, {"kcal": 2000}):
        assert client.put("/api/settings", json=bad).status_code == 422


def test_delete_everything(client):
    a = upload(client, png()).json()["analysis_id"]
    meal(client, analysis_id=a)
    assert client.delete("/api/data").json() == {"deleted": True}
    assert client.get("/api/meals").json()["meals"] == [] and not list((store.data_dir() / "uploads").glob("*"))


def test_database_from_the_first_version_is_upgraded_in_place(client):
    with store.db() as c:
        c.execute("DROP TABLE meals")
        c.execute("CREATE TABLE meals (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, analysis_id TEXT, foods_json TEXT NOT NULL, totals_json TEXT NOT NULL)")
        c.execute("INSERT INTO meals VALUES('11111111-1111-1111-1111-111111111111','2025-12-24T08:00:00+00:00',NULL,'[{\"name\":\"roti\",\"grams\":60}]','{\"kcal\":178,\"protein_g\":5,\"carbs_g\":33,\"fat_g\":4.5}')")
    store.init()
    m = client.get("/api/meals").json()["meals"][0]
    assert m["local_date"] == "2025-12-24" and m["image_url"] is None and m["note"] == ""


def test_headers_and_unknown_routes(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store" and "frame-ancestors 'none'" in r.headers["content-security-policy"]
    home = client.get("/")
    assert home.status_code == 200 and home.headers["content-type"].startswith("text/html")
    assert client.get("/nope").status_code == 404 and client.get("/api/nope").json() == {"detail": "Not Found"}
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/api/foods").json()["foods"].count("roti") == 1


def test_meals_are_listed_by_their_own_day_newest_first(client):
    for d in ("2026-03-08", "2026-03-10", "2026-03-09"):          # saved out of order on purpose
        meal(client, local_date=d)
    meal(client, [{"name": "dal", "grams": 100}], local_date="2026-03-10")
    got = client.get("/api/meals").json()["meals"]
    assert [m["local_date"] for m in got] == ["2026-03-10", "2026-03-10", "2026-03-09", "2026-03-08"]
    assert got[0]["foods"][0]["name"] == "dal"                  # within a day, the most recently saved first
