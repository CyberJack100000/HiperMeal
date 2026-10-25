# HiperMeal: a visual nutrition diary

Photograph a meal, check the foods and portions an AI suggests, and log a rough nutrition estimate. Over time you get your day, your week and your most-eaten foods.

**Stack:** FastAPI + SQLite backend, plain HTML/CSS/JavaScript front end (no build step), Google's hosted **Gemma 4** model (default `gemma-4-31b-it`) for photo recognition.

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

### Turn on real photo recognition (Gemma 4)
Get a Google AI Studio key at https://aistudio.google.com/apikey, then use **either** way:

- **In the app (easiest):** Settings → *AI photo recognition* → paste the key → **Save key**, then **Test key**. The key stays in your browser (this session only, or remembered on this device if you tick the box). It is sent with each photo or estimate request, passed straight to Google, and never stored or logged by the server. You can also change the model there.
- **On the server:** copy `.env.example` to `.env` and put the key after `GEMINI_API_KEY=` (the app reads `.env` itself). Everyone who uses this server then shares that key. A key entered in the app takes priority over this one.

The default model is `gemma-4-31b-it`; set `GEMMA_MODEL` (or the Model box in Settings) to any Gemma 4 model your key can use.

**Check it once with your real key** (I could not do this when building it):
```bash
python scripts/check_ai.py              # or: python scripts/check_ai.py my_meal.jpg
```
It lists the Gemma models your key can see, runs a key check, analyses a photo, shows which structured-output mode the API accepted, and asks for a sample nutrition estimate.

Without a key everything still works, but analysing a photo returns three clearly labelled **demo suggestions** (not a reading of your photo). Which models a key can use, quotas and charges depend on your Google account. If you use the app over a network, use HTTPS: the key travels with each request.

### How Gemma 4 is used
- **One structured prompt** asks the model to find each food, name it (reusing the app's own food names where they fit, so more foods match the table), estimate the grams using plate and cutlery for scale, note how it is prepared, rate its confidence, and give a one-line reason. The reason and preparation are shown in the editor so you can judge the guess.
- **Strict answers, with a safety net.** The request starts with a JSON schema; if the API says that mode is not supported for the model, it steps down to plain JSON mode and then to prompt-only JSON, and remembers which worked. A 400 about the key, the model or the image never triggers a step-down.
- **Estimates for unknown foods.** For a food the table does not know, **Ask Gemma 4 to estimate** fills in rough per-100-g values (checked for physical sense: macros cannot exceed 100 g, calories must match the macros). You check them before saving; they are stored as your own values.
- **Cuisine hint.** An optional hint on the upload box ("Indian", "Italian", …) helps with look-alike dishes.
- Requests time out after 60 seconds. **Temporary failures are retried with growing waits**: a busy or erroring Google server (HTTP 500/503/504) up to three times (1.5, 4 and 9 seconds), a rate limit once, a dropped connection twice, all within about 75 seconds.
- **Backup model.** If the main model is still overloaded after those retries, the request is tried once on a backup model (default `gemma-4-26b-a4b-it`; change it with `GEMMA_BACKUP_MODEL` or in Settings, or write `none` to turn it off). The review screen says when the backup answered. A backup is only used for overload errors, never for a bad key, a missing model or a quota problem, since another model would not fix those. If the backup also fails, you are told about the main model's problem.
- Errors show Google's own explanation, for example "HTTP 503: This model is currently experiencing high demand".

## What it does

| Page | What you get |
| --- | --- |
| **Home** | Today's totals, photo upload (drag and drop; JPG, PNG, WEBP, GIF, BMP, TIFF, AVIF, HEIC), a progress bar while it analyses, the review editor, recent meals |
| **Review editor** | Edit every food and portion, add items, pick a meal type, add a note. Matches are shown ("Matched", "Matched as…", "Closest match, check this"). A food the table does not know asks for its values per 100 g instead of being guessed. |
| **Meal History** | Every meal from the last 30 days, grouped by day, with its photo; delete a meal and its photo goes too |
| **Nutrition Insights** | 7/14/30-day calorie chart against your target, averages, most-logged foods |
| **Settings** | Your Gemma API key and model (with a Test button), daily targets, light/dark/system theme, delete all data |

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
- 288 foods in the table (was 19): Indian dishes from every course, East and Southeast Asian, Western, Middle Eastern, desserts, drinks, fruit, vegetables and staples, with about 300 alternative names ("chai", "gyoza", "brinjal"). Still approximate. Look-alikes keep their own numbers ("egg curry" is not "egg", "fried rice" is not "rice"), and a name the table does not know still asks for values.
- A real progress bar while a photo is analysed: true percentage while uploading, then a working animation with a timer and a Cancel button (the model's time cannot be predicted, so no made-up percentage). It scrolls into view on phones.
- More photo types: JPG, PNG, WEBP, GIF, BMP, TIFF, AVIF and HEIC (the iPhone default). The file's content decides, not the type the browser claims, and the review screen shows the cleaned copy so even HEIC previews. SVG, PDF and other non-photo files are refused.
- Photos are re-encoded on upload: location and device metadata is removed, rotation fixed, size reduced before anything is sent to the model.
- Privacy: unsaved photos are deleted after 24 hours, deleting a meal deletes its photo, one-click delete-all, no third-party fonts or images, a strict content security policy.
- Robustness: rate limit on photo analysis, model timeout and one retry, tolerant parsing of the model's answer, validated ids, in-place upgrade of databases created by the first version.
- Accessibility: keyboard and screen-reader friendly, no automated-audit violations in light or dark at desktop or phone width.
- 145 automated tests, plus browser tests of every page, the progress bar (under a throttled connection), every image type and the key flow.

## API

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Status, whether the server has a key, default model, structured-output mode in use, supported image formats |
| `GET /api/photos/{id}` | The cleaned JPEG of a photo just analysed (used for the preview) |
| `POST /api/analyze` | Multipart image (`file`, optional `hint`) → suggested foods with matches, preparation and reasoning. Optional headers `X-Gemini-Key`, `X-Gemma-Model` |
| `POST /api/nutrition/estimate` | `{food_name}` → rough per-100-g values from Gemma 4 (same optional headers) |
| `POST /api/ai/test` | Tiny request to check a key and model (same optional headers) |
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
- A photo is sent to Google only when a key is available, under that key's account and terms. A key you enter in the app lives only in your browser and is never stored or logged by the server; a key set in `.env` stays on the server.
- Tests: `python -m pip install -r requirements-dev.txt` then `python -m pytest -q`.

## Known limits

- The nutrition table is rounded and approximate. Replace it with a cited source (for example IFCT or USDA FoodData Central) before anyone relies on it.
- Accuracy of recognition and portion size has not been measured on a set of real meals. Test with your own food before trusting it.
- The request to Gemma 4 is covered by automated tests only through a stand-in, so run `python scripts/check_ai.py` once with your key. Which structured-output modes the hosted Gemma 4 API accepts was not verified; the step-down logic is there for that reason.
- HEIC photos need the `pillow-heif` package (it is in `requirements.txt`). If it fails to install on some machine, everything else still works and HEIC is simply left off the supported list (shown on the upload box). Only the first picture of an animated GIF or multi-page TIFF is used.
- Single user, local only, SQLite.
- Targets are whatever you type; the app does not calculate personal needs.

## Project layout

```
main.py          routes, image handling, rate limit, security headers, per-request API key
nutrition.py     the food table and the careful matching
ai.py            the Gemma 4 requests, the strict-to-plain fallback, parsing and checking of answers
env.py           reads .env
scripts/         check_ai.py: verify your key and model against the real API
store.py         SQLite, photo files, cleanup
static/          index.html, app.css, app.js, theme.js
tests/           pytest suite
docs/            project presentation (.pptx) and walkthrough video from the first prototype
```
The deck and video in `docs/` describe the original prototype; the workflow they show is unchanged, the pages listed above are new.


Developed by :-
Souparna Saha,
Aryan Kushwaha,
Gourav Das and
Priyanshu Singhal
