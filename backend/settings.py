"""
settings.py
===========
PURPOSE
    EVERY adjustable setting of the translator lives in this one file.
    No other module hard-codes a number, a folder or a language list; they
    all read it from here with `import settings`.

WHY IT EXISTS
    Before the refactor the settings were scattered through one very long
    file. Keeping them together means that tuning the app (speed vs.
    accuracy, which languages, where models are stored) never requires
    reading the program logic.

HOW TO CHANGE A SETTING WITHOUT EDITING CODE
    Many settings read an ENVIRONMENT VARIABLE first. In PowerShell, before
    starting the backend:

        $env:WHISPER_DETECT_MODEL="small"
        uvicorn main:app --reload

    The variable only lasts for that terminal window.

USED BY
    speech_models.py, translation.py, language_detect.py, pipeline.py,
    talkback.py, routes_http.py, routes_live.py, main.py
"""

import os  # needed to read environment variables and build folder paths

# -----------------------------------------------------------------
# BASE FOLDER
# -----------------------------------------------------------------
# WHAT : the folder that contains this file (the "backend" folder).
# WHY  : model folders and the Piper voice search are written relative to
#        it, so the app works wherever the project folder is copied
#        (for example another PC) without changing any path.
HERE = os.path.dirname(os.path.abspath(__file__))


# =================================================================
# 1. SPEECH-TO-TEXT (WHISPER) MODELS
# =================================================================

# WHAT : size of the GENERAL live Whisper model.
# WHY  : "base" is fast enough for live use on a CPU. "small", "medium" or
#        "large-v3" are more accurate but slower. For Hindi/Bengali/Korean
#        a specialist model (below) is used instead whenever it exists.
# USED : speech_models.py loads it at start-up.
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

# WHAT : where the models run ("cpu" or "cuda" for an NVIDIA GPU).
# WHY  : the prototype PC has no NVIDIA GPU, so "cpu" is the default. A
#        server with a GPU can set WHISPER_DEVICE=cuda for a big speed-up.
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")

# WHAT : numeric precision of the models ("int8" is small and fast on CPU,
#        "float16" is for GPUs).
# WHY  : int8 cuts memory use and time on a CPU with very little accuracy
#        loss, which is what lets a normal office PC run the models.
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")

# WHAT : how many CPU threads each model may use.
# WHY  : use every core except one, so the PC stays responsive for the
#        browser and the meeting itself. At least 2 are always used.
CPU_THREADS = max(2, (os.cpu_count() or 4) - 1)

# WHAT : the BIGGER Whisper model used for uploaded/recorded files
#        ("Best accuracy" in the menu).
# WHY  : files are not live, so waiting longer for a better result is fine.
#        It is loaded only when needed, and ONLY if there is no specialist
#        model for the chosen language (see speech_models.py).
ACCURATE_MODEL = os.environ.get("WHISPER_ACCURATE_MODEL", "small")

# WHAT : optional bigger model used ONLY to guess the language in
#        Auto-detect. Empty text = not used (the live model guesses).
# WHY  : guessing with "small" is more reliable than with "base", but it
#        costs extra time on every chunk, so it is opt-in.
DETECT_MODEL_NAME = os.environ.get("WHISPER_DETECT_MODEL", "")

# WHAT : folders of the language-specific ("specialist") models.
# WHY  : general Whisper is weak for Hindi. A model trained on Hindi is far
#        more accurate at the same speed. If a folder below exists (created
#        by convert_model.py) it is used automatically for that language;
#        if not, the general model is used.
# KEYS : language code -> folder. Each value can be overridden by its
#        environment variable (useful on a server with another layout).
SPECIALIST_DIRS = {
    "hi": os.environ.get(
        "WHISPER_HINDI_MODEL",
        os.path.join(HERE, "models", "whisper-hi-ct2")
    ),
    "bn": os.environ.get(
        "WHISPER_BENGALI_MODEL",
        os.path.join(HERE, "models", "whisper-bn-ct2")
    ),
    "ko": os.environ.get(
        "WHISPER_KOREAN_MODEL",
        os.path.join(HERE, "models", "whisper-ko-ct2")
    )
}

# WHAT : settings of Whisper's built-in "ignore silent parts" filter.
# WHY  : without it Whisper wastes time on silence and may invent words
#        there. 500 ms of silence is treated as a pause.
# USED : speech_models.decode_options().
VAD_SETTINGS = {"min_silence_duration_ms": 500}


# =================================================================
# 2. TEXT TRANSLATION MODEL (NLLB, OFFLINE)
# =================================================================
# TWO-STEP method (preferred):
#     audio -> original text (Whisper) -> English text (NLLB)
# Doing it in two steps lets the screen show the ORIGINAL text first and
# keeps both sides matched row by row. If the NLLB files are missing, the
# app falls back to Whisper's own audio -> English translation.

# WHAT : folder of the converted NLLB translation model.
# WHY  : created by download_models.py. Can be overridden for a server.
MT_MODEL_DIR = os.environ.get(
    "MT_MODEL_DIR",
    os.path.join(HERE, "models", "nllb-600m-ct2")
)

# WHAT : folder of the NLLB tokenizer (turns text into model pieces).
# WHY  : the translation model cannot read plain text without it.
MT_TOKENIZER_DIR = os.environ.get(
    "MT_TOKENIZER_DIR",
    os.path.join(HERE, "models", "nllb-tokenizer")
)

# WHAT : FALLBACK mode only - also produce the original text.
# WHY  : that needs a second Whisper pass, which is too slow for live use,
#        so it is off by default.
SHOW_ORIGINAL_TEXT = False


# =================================================================
# 3. LANGUAGES
# =================================================================

# WHAT : languages the app officially supports.
# WHY  : Auto-detect is limited to these, so a short sentence is not
#        mistaken for some language the app cannot translate.
SUPPORTED_LANGUAGES = ["hi", "bn", "ko"]

# WHAT : languages Auto-detect may choose from = supported + English.
# WHY  : English is included so a speaker's own English is recognised as
#        English instead of being forced into Hindi/Bengali/Korean.
DETECTION_CANDIDATES = SUPPORTED_LANGUAGES + ["en"]

# WHAT : True = show nothing when English is heard.
# WHY  : in a meeting where the user's own English should not be
#        translated or read aloud again.
IGNORE_ENGLISH = False

# WHAT : True = if Whisper writes Hindi in Urdu letters, retry/force
#        Devanagari letters.
# WHY  : Hindi and Urdu sound almost the same, and general Whisper often
#        picks the wrong alphabet.
TREAT_URDU_AS_HINDI = True


# =================================================================
# 4. MIXED-LANGUAGE LIVE AUDIO (AUTO-DETECT ONLY)
# =================================================================
# Problem solved: when two languages alternate (Hindi, then Korean), one
# 10-second chunk may hold both. One language per chunk would then read
# everything as the same language.

# WHAT : switch for the whole feature.
# WHY  : True splits an unclear chunk at speech pauses and guesses the
#        language of each piece. False = one language per chunk (faster,
#        simpler; use it if splitting causes trouble).
MIXED_LANGUAGE_MODE = True

# WHAT : how sure the guess must be (0.85 = 85%) to treat a chunk as ONE
#        language.
# WHY  : a confident chunk is processed immediately (cheap, the common
#        case); only unclear chunks pay the cost of splitting.
SPLIT_CONFIDENCE = 0.85

# WHAT : maximum number of pieces a chunk may be split into.
# WHY  : each piece is another language guess + transcription, so too many
#        pieces would make the delay grow.
MAX_PIECES = 4

# WHAT : pieces shorter than MIN_PIECE_SECONDS are ignored (cough, click).
#        Pieces shorter than SHORT_PIECE_SECONDS are too short to guess
#        reliably and copy the language of a neighbouring piece.
# WHY  : language guesses on under ~1.5 seconds of audio are unreliable.
MIN_PIECE_SECONDS = 0.5
SHORT_PIECE_SECONDS = 1.5

# WHAT : a pause this long (milliseconds) separates two pieces.
# WHY  : people take a short breath when they switch speaker or language.
PIECE_PAUSE_MS = 400


# =================================================================
# 5. LIVE AUDIO
# =================================================================

# WHAT : audio sample rate in Hz. The browser sends 16000 Hz, which is also
#        what Whisper expects.
# WHY  : one shared value keeps the page, the backend and the model in step.
SAMPLE_RATE = 16000

# WHAT : chunks quieter than this are treated as silence and skipped.
# WHY  : saves processing time during silence and avoids invented text.
SILENCE_LEVEL = 0.003

# WHAT : how many audio chunks may wait while the models are busy.
# WHY  : audio is NEVER dropped just because processing is slow; chunks
#        queue up instead. 120 chunks x 10 s = 20 minutes of waiting audio.
#        The limit only protects memory.
MAX_WAITING_CHUNKS = 120


# =================================================================
# 6. WEB ACCESS
# =================================================================

# WHAT : web pages allowed to talk to this backend (CORS).
# WHY  : browsers block a page from calling another address unless that
#        address allows it. These are the Vite development server
#        addresses. Add the server's address here when deploying.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174"
]
