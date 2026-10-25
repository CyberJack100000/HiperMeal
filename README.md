# HiperMeal: a visual nutrition diary

Photograph a meal, check the foods and portions an AI suggests, and log a rough nutrition estimate. Over time you get your day, your week and your most-eaten foods.

**Stack:** FastAPI + SQLite backend, plain HTML/CSS/JavaScript front end (no build step), Google's hosted Gemma 4 model for photo recognition (optional).

> The numbers are estimates. Food recognition from a photo can be wrong, portions are guesses, and the built-in nutrition table is rounded and approximate. Nothing here is medical or dietary advice.

## Run it

**Windows (PowerShell):**
```powershell
cd HiperMeal
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass      # only if activating the venv is blocked
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```
**macOS / Linux:**
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000.

### Turn on real photo recognition
1. Get a Google AI Studio key: https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and put the key after `GEMINI_API_KEY=` (the app reads `.env` itself; no need to set variables in the terminal). Keep it private: it is only ever used by the backend and `.env` is git-ignored.
3. Restart. `http://127.0.0.1:8000/api/health` shows `"ai_configured": true`, and Settings says so too.

Without a key everything works, but analysing a photo returns three clearly labelled **demo suggestions** (they are not a reading of your photo). Which model your key can use, quotas and charges depend on your Google account; set `GEMMA_MODEL` if yours differs.

## What it does

| Page | What you get |
| --- | --- |
| **Home** | Today's totals, photo upload (drag and drop), the review editor, recent meals |
| **Review editor** | Edit every food and portion, add items, pick a meal type, add a note. Matches are shown ("Matched", "Matched as…", "Closest match, check this"). A food the table does not know asks for its values per 100 g instead of being guessed. |
| **Meal History** | Every meal from the last 30 days, grouped by day, with its photo; delete a meal and its photo goes too |
| **Nutrition Insights** | 7/14/30-day calorie chart against your target, averages, most-logged foods |
| **Settings** | Daily targets, light/dark/system theme, delete all data, whether AI is on |

## What changed from the first prototype (v0.3)

The original upload is the first commit in this repository, so you can diff against it. The improvements:

**Bugs fixed**
- **Saving a meal failed if any food was unknown.** The server answered "no nutrition record" and refused the whole meal, so most real AI answers could not be saved. Unknown foods now take your own per-100-g values, validated server-side.
- **"Today's summary" added up every meal ever saved.** It now counts only today, using the person's own calendar day (the server cannot know their time zone, so the browser sends it).
- **Photos disappeared after a reload** and meals never kept them. Photos now belong to the meal.
- **Wrong matches.** Names were matched loosely. Matching now accepts only the same food, a known alias, trivial variants ("boiled white rice") or a close spelling, labels guesses as guesses, and refuses things like "egg curry" matching "egg".
- **Three menu items were placeholders** ("planned for a later iteration") and the dark mode was a throwaway preview that was not remembered. They are real pages and a proper theme now.

**New**
- Meal History, Nutrition Insights, Settings (targets), meal type and notes, delete from the UI.
- 75 foods in the table (was 19), including common Indian dishes. Still approximate.
- Photos are re-encoded on upload: location and device metadata is removed, rotation fixed, size reduced before anything is sent to the model.
- Privacy: unsaved photos are deleted after 24 hours, deleting a meal deletes its photo, one-click delete-all, no third-party fonts or images, a strict content security policy.
- Robustness: rate limit on photo analysis, model timeout and one retry, tolerant parsing of the model's answer, validated ids, in-place upgrade of databases created by the first version.
- Accessibility: keyboard and screen-reader friendly, no automated-audit violations in light or dark at desktop or phone width.
- 51 automated tests.

## API

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Status, whether AI is configured, model name |
| `POST /api/analyze` | Multipart image (`file`) → suggested foods with nutrition matches |
| `POST /api/nutrition/lookup` | `{food_name}` → match, or 404 |
| `POST /api/meals` | Save a meal (totals are recalculated on the server) |
| `GET /api/meals?days=&today=` | Saved meals, newest day first |
| `GET /api/meals/{id}/image` | The meal's photo |
| `DELETE /api/meals/{id}` | Delete a meal and its photo |
| `GET /api/summary?days=&today=` | Per-day totals, today, averages, top foods, targets |
| `GET`/`PUT /api/settings` | Daily targets |
| `GET /api/foods` | Names in the built-in table |
| `DELETE /api/data` | Delete everything |

Interactive docs: http://127.0.0.1:8000/docs

## Privacy and security

- Meals and photos are stored on the machine running the app (`data/hipermeal.sqlite3`, `data/uploads/`). **There is no sign-in and no separation between users.** Run it on your own computer or network. Do not put it on the public internet with real food photos until authentication, access control and retention are added.
- The API key stays on the server. A photo is sent to Google only when a key is configured.
- Tests: `python -m pip install -r requirements-dev.txt` then `python -m pytest -q`.

## Known limits

- The nutrition table is rounded and approximate. Replace it with a cited source (for example IFCT or USDA FoodData Central) before anyone relies on it.
- Accuracy of recognition and portion size has not been measured on a set of real meals. Test with your own food before trusting it.
- The Gemma request itself (real key, real model) is covered by tests only through a stand-in; try it once with your key.
- Single user, local only, SQLite.
- Targets are whatever you type; the app does not calculate personal needs.

## Project layout

```
main.py          routes, image handling, rate limit, security headers
nutrition.py     the food table and the careful matching
ai.py            the Gemma 4 request and parsing of its answer
store.py         SQLite, photo files, cleanup
static/          index.html, app.css, app.js, theme.js
tests/           pytest suite
docs/            project presentation (.pptx) and walkthrough video from the first prototype
```
The deck and video in `docs/` describe the original prototype; the workflow they show is unchanged, the pages listed above are new.
