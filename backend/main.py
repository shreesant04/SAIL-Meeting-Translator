"""
main.py  -  SAIL ISP backend (START POINT)
==========================================
START IT from the backend folder with:

    uvicorn main:app --reload

PURPOSE
    This file only ASSEMBLES the application. All real work is done in
    small modules, each with one job. Read them in this order to understand
    the whole app:

        settings.py         all adjustable settings (and environment variables)
        languages.py        language names / codes, alphabet checks
        speech_models.py    loads the Whisper models (general + per language)
        translation.py      loads the offline text-translation model (NLLB)
        text_cleanup.py     removes noise, marks garbled text "(unclear audio)"
        romanize.py         Roman-letter line under the original text
        language_detect.py  Auto-detect and mixed-language planning
        pipeline.py         audio -> original text -> English text (the core)
        talkback.py         Piper voice that reads the English aloud
        routes_http.py      web addresses: health, talkback voice, file upload
        routes_live.py      live audio WebSocket

DATA FLOW (live meeting)
    browser audio --> routes_live.py (queue) --> pipeline.py
        --> language_detect.py (which language?)
        --> speech_models.py   (audio -> original text)
        --> text_cleanup.py    (hide garbled text)
        --> romanize.py        (Roman letters)
        --> translation.py     (original text -> English)
    --> routes_live.py --> browser (row by row)
"""

from fastapi import FastAPI                          # the web application
from fastapi.middleware.cors import CORSMiddleware   # lets the page connect

import settings   # allowed web addresses

# ---- LOAD THE MODELS ----
# WHAT : importing these modules loads the models and prints progress in
#        the backend window.
# WHY  : doing it here, at start-up, means the first live chunk is not
#        delayed. The order only affects the order of the messages
#        (Whisper first, then the translation model, then Roman letters).
import speech_models  # noqa: F401  (loads the Whisper models)
import translation    # noqa: F401  (loads the text-translation model)
import romanize       # noqa: F401  (reports the Roman-letter packages)

# ---- THE ADDRESSES (ROUTES) ----
from routes_http import router as http_router   # health, talkback, files
from routes_live import router as live_router   # live audio WebSocket

# WHAT : the application object that uvicorn runs ("main:app").
app = FastAPI(title="SAIL ISP")

# ---- CORS ----
# WHAT : tells browsers which web pages may talk to this backend.
# WHY  : the page runs on another port (the Vite server, 5173/5174); the
#        browser blocks such calls unless they are allowed here. The list
#        is in settings.ALLOWED_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- ATTACH THE ROUTES ----
# WHAT : adds the addresses defined in the two route files to the app.
app.include_router(http_router)
app.include_router(live_router)
