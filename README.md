# HiperMeal + Gemma 4 (Full Stack)

A local meal-photo diary prototype with an HTML/CSS/JavaScript frontend, FastAPI backend, SQLite meal history, and optional hosted Gemma 4 image recognition.

## Requirements
- Python 3.11 or newer
- Internet access for installing packages and calling hosted Gemma 4
- A Google AI Studio API key with access to Gemma 4 for real image recognition

## Windows setup (Command Prompt)
1. Extract the ZIP. Open Command Prompt **inside the `hipermeal_fullstack` folder** (the folder containing `main.py`).
2. Create and activate a virtual environment:
   ```bat
   py -m venv .venv
   .venv\Scripts\activate.bat
   ```
3. Install dependencies:
   ```bat
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Set your API key in that same terminal window (get a key at https://aistudio.google.com/apikey):
   ```bat
   set GEMINI_API_KEY=PASTE_YOUR_REAL_KEY_HERE
   set GEMMA_MODEL=gemma-4-26b-a4b-it
   ```
   Keep the key private. It is only read by the backend and must never be placed in frontend JavaScript or committed to GitHub.
5. Start the server:
   ```bat
   python -m uvicorn main:app --reload
   ```
6. Open http://127.0.0.1:8000

## PowerShell alternative
Use `$env:GEMINI_API_KEY="YOUR_KEY"` and `$env:GEMMA_MODEL="gemma-4-26b-a4b-it"` instead of the `set` commands. Activate the environment with ` .\.venv\Scripts\Activate.ps1 `.

## Verify it
- UI: http://127.0.0.1:8000
- Health: http://127.0.0.1:8000/api/health
- Interactive API docs: http://127.0.0.1:8000/docs
- Health response `ai_configured: true` means the key is present; upload an image to confirm the actual model request works.

Without an API key, the app returns clearly labelled demo suggestions for frontend testing. Those suggestions are **not** recognition of the uploaded photo. Hosted API availability, quotas, and charges depend on your Google account and current terms.

## API routes
- `GET /api/health`
- `POST /api/analyze` (multipart image field named `file`)
- `POST /api/nutrition/lookup`
- `POST /api/meals`
- `GET /api/meals`
- `DELETE /api/meals/{meal_id}`

## Important limitations
- The included nutrition lookup values are illustrative prototype values, not a verified clinical or official food-composition database.
- Portion sizes and food identification from photographs can be wrong. Review all suggestions before saving; this is not medical advice.
- Images are stored locally in `data/uploads/` and meals in `data/hipermeal.sqlite3`. There is no login or user separation. Do not expose this prototype publicly with personal food photos until authentication, access controls, retention, and deployment security are implemented.
