"""
speech_models.py
================
PURPOSE
    Loads and manages every SPEECH-TO-TEXT (Whisper) model of the app:

      1. the general live model            (settings.MODEL_SIZE, at start-up)
      2. specialist models per language    (Hindi / Bengali / Korean, at start-up)
      3. the "accurate" model              (only when a file needs it)
      4. the language-detection model      (optional, Auto-detect only)

    It also holds `model_lock`: only ONE job may use the models at a time.

WHY IT EXISTS
    Models are big (hundreds of MB) and slow to load, so they must be
    loaded ONCE and shared. Keeping all loading logic here means the rest
    of the app just asks "give me the model for Hindi".

HOW OTHER FILES USE IT
    import speech_models as sm      ->  sm.model, sm.model_lock, ...
    IMPORTANT: importing this file LOADS THE MODELS (progress lines are
    printed in the backend window). main.py imports it first for that
    reason.

USED BY
    pipeline.py, routes_http.py (health), main.py
"""

import os          # to check that model folders/files exist
import threading   # locks, so two jobs never load or use a model together

from faster_whisper import WhisperModel   # the Whisper engine (CTranslate2)

import settings                      # every adjustable setting
from languages import is_arabic_char  # used to find Urdu-letter tokens


# =================================================================
# 1. THE GENERAL LIVE MODEL
# =================================================================

print("Loading Whisper model...")

# WHAT : the general-purpose Whisper model, loaded once at start-up.
# WHY  : it is the live fallback for any language without a specialist
#        model, and it also guesses the language in Auto-detect.
model = WhisperModel(
    settings.MODEL_SIZE,
    device=settings.DEVICE,
    compute_type=settings.COMPUTE_TYPE,
    cpu_threads=settings.CPU_THREADS
)

# WHAT : a lock that only one job at a time may hold.
# WHY  : the models are not safe to run twice at once and the CPU would be
#        overloaded. Whoever works with a model (live chunk or file) first
#        takes this lock, so jobs queue up instead of colliding.
model_lock = threading.Lock()

print("Whisper model loaded successfully.")


# =================================================================
# 2. HINDI IN DEVANAGARI LETTERS
# =================================================================
# PROBLEM : Whisper often writes Hindi speech in Urdu letters (the two
#           languages sound alike).
# FIX     : forbid every Whisper token that writes Arabic/Urdu letters.
#           The model must then use Devanagari.

def build_arabic_token_ids(whisper=None):
    """
    PURPOSE : find every Whisper token that writes Arabic / Urdu letters.
    WHY     : these ids are later passed as "suppress_tokens" so the model
              cannot choose them (see hindi_options below).
    INPUT   : the model whose vocabulary to scan (default: the general
              model). Each model has its own vocabulary, so the list must
              be built per model.
    RETURNS : a sorted list of token ids; an empty list if anything fails
              (the app then simply works without the filter).
    """
    try:
        # The model's complete word list: {token_text: token_id}
        vocabulary = (whisper or model).hf_tokenizer.get_vocab()

        # Whisper tokens are stored "byte-level": every byte is shown as a
        # special printable character. To read a token as real text we
        # must rebuild that byte <-> character table first.
        byte_values = (
            list(range(33, 127))
            + list(range(161, 173))
            + list(range(174, 256))
        )
        char_codes = byte_values[:]
        extra = 0

        # LOOP 1 - belongs to: build_arabic_token_ids(), step "rebuild the
        #                      byte table"
        # WHY  : bytes that are not printable on their own (0-32, 127-160,
        #        173) are given substitute characters numbered from 256 up.
        # REPEATS: once for each of the 256 possible byte values.
        for byte in range(256):
            if byte not in byte_values:
                byte_values.append(byte)
                char_codes.append(256 + extra)
                extra += 1

        # Table to convert a token's characters back into bytes
        # ONE-LINE LOOP - belongs to: build_arabic_token_ids(), after LOOP 1
        # WHY  : pairs every byte with its stand-in character, giving the lookup
        #        table that LOOP 2 needs to read tokens as bytes.
        # REPEATS: once per byte/character pair (256 times).
        char_to_byte = {
            chr(code): byte
            for byte, code in zip(byte_values, char_codes)
        }

        ids = []

        # LOOP 2 - belongs to: build_arabic_token_ids(), step "scan the
        #                      whole vocabulary"
        # WHY  : the only way to know which tokens write Urdu letters is
        #        to decode every token of the model and look at its text.
        # REPEATS: once per token (about 50,000 times).
        for token, token_id in vocabulary.items():
            try:
                # Characters of the token -> bytes
                # ONE-LINE LOOP - belongs to: LOOP 2, per token
                # WHY  : converts each character of the token back into its byte so the
                #        token can be decoded as real text.
                # REPEATS: once per character of the token.
                raw = bytes(char_to_byte[ch] for ch in token)
            except KeyError:
                # Special tokens that are not byte-level text: skip them
                continue

            # Bytes -> real text (UTF-8)
            text = raw.decode("utf-8", errors="ignore")

            # LOOP 3 (inside the check) - belongs to: LOOP 2, per token
            # WHY  : one Arabic letter anywhere in the token is enough to
            #        forbid the whole token.
            if any(is_arabic_char(ch) for ch in text):
                ids.append(token_id)

        return sorted(ids)

    except Exception as error:
        # Never stop the app because the filter could not be built
        print("Could not prepare the Hindi-letters filter:", error)
        return []


# Built once for the general model at start-up
ARABIC_TOKEN_IDS = build_arabic_token_ids()
print("Hindi-letters filter ready:", len(ARABIC_TOKEN_IDS), "tokens.")

# WHAT : one forbidden-token list per model, keyed by the model's identity.
# WHY  : the Hindi specialist model has a different vocabulary than the
#        general model, so it needs its own list. Caching avoids scanning
#        50,000 tokens again for every chunk.
arabic_cache = {id(model): ARABIC_TOKEN_IDS}


def arabic_ids(whisper):
    """
    PURPOSE : the forbidden-token list for a given model (built once, then
              reused from arabic_cache).
    """
    key = id(whisper)

    # Not built yet for this model: build it now and remember it
    if key not in arabic_cache:
        arabic_cache[key] = build_arabic_token_ids(whisper)

    return arabic_cache[key]


def hindi_options(whisper):
    """
    PURPOSE : extra decoding options for Hindi: block Urdu letters so the
              text comes out in Devanagari.
    WHY     : replaces an earlier "prompt" trick; a text prompt made Whisper
              repeat the prompt words in its output, while blocking tokens
              has no such side effect.
    NOTE    : -1 in the list is Whisper's own "default suppressed tokens"
              marker and must stay.
    """
    return {
        "suppress_tokens": [-1] + arabic_ids(whisper)
    }


# =================================================================
# 3. SPECIALIST MODELS (ONE PER LANGUAGE)
# =================================================================
# WHY : general Whisper "base"/"small" is weak for Hindi (error rate
#       149% / 67% on Google Fleurs). A model trained on Hindi is far
#       better at the same size. Put a converted model in its folder (see
#       convert_model.py) and it is used automatically for that language.

specialist_models = {}        # language code -> loaded model
specialist_failed = set()     # language codes whose model failed to load
specialist_lock = threading.Lock()   # stops two threads loading one model


def specialist_available(code):
    """
    PURPOSE : True if a converted model folder exists for this language.
    HOW     : checks that the folder contains the file "model.bin".
    """
    folder = settings.SPECIALIST_DIRS.get(code)

    return bool(folder) and os.path.exists(
        os.path.join(folder, "model.bin")
    )


def specialist_model(code):
    """
    PURPOSE : the model trained for this language, or None (then the
              general model is used).
    WHY     : loaded on first use and then kept in memory, so later chunks
              pay no loading time.
    """
    # No language, or a language without a specialist folder configured
    if not code or code not in settings.SPECIALIST_DIRS:
        return None

    # Already loaded: reuse it
    if code in specialist_models:
        return specialist_models[code]

    # Failed before, or the folder does not exist: use the general model
    if code in specialist_failed or not specialist_available(code):
        return None

    # Load it (the lock makes sure only one thread does this)
    with specialist_lock:
        # Another thread may have loaded it while we waited for the lock
        if code in specialist_models:
            return specialist_models[code]

        try:
            print("Loading the", code, "specialist model from",
                  settings.SPECIALIST_DIRS[code], "...")

            specialist_models[code] = WhisperModel(
                settings.SPECIALIST_DIRS[code],
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                cpu_threads=settings.CPU_THREADS
            )

            print("Specialist model for", code, "loaded.")

            return specialist_models[code]

        except Exception as error:
            # Remember the failure so we do not retry on every chunk
            specialist_failed.add(code)
            print(
                "Could not load the", code, "specialist model:", error,
                "\n -> Using the general model instead."
            )
            return None


# LOOP - belongs to: module start-up, section 3 (specialist models)
# WHY  : load every available specialist model NOW, so the first live
#        chunk is not delayed by a model download/load, and so the startup
#        text tells you which languages have a specialist model.
# REPEATS: once per configured language (hi, bn, ko).
for _code in settings.SPECIALIST_DIRS:
    if specialist_available(_code):
        specialist_model(_code)
    else:
        print("No specialist model for", _code,
              "(using the general", settings.MODEL_SIZE, "model).")


# =================================================================
# 4. THE ACCURATE MODEL (UPLOADED / RECORDED FILES)
# =================================================================

accurate_model = None                 # loaded on demand
accurate_lock = threading.Lock()      # stops two threads loading it twice


def get_accurate_model():
    """
    PURPOSE : the bigger model for "Best accuracy" files.
    WHY     : loaded only the first time it is needed (the download can be
              large), then kept. If it is the same as the live model, the
              live model is returned instead of loading a duplicate.
    """
    global accurate_model

    # Same model as the live one: nothing extra to load
    if settings.ACCURATE_MODEL == settings.MODEL_SIZE:
        return model

    with accurate_lock:
        if accurate_model is None:
            print(
                "Loading the accurate model:", settings.ACCURATE_MODEL,
                "(first time only; the download may be large)..."
            )

            accurate_model = WhisperModel(
                settings.ACCURATE_MODEL,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                cpu_threads=settings.CPU_THREADS
            )

            print("Accurate model loaded.")

    return accurate_model


# =================================================================
# 5. THE LANGUAGE-DETECTION MODEL (OPTIONAL)
# =================================================================
# WHY : guessing the language is more reliable with "small" than with
#       "base", but costs extra time for every chunk. Enable it in
#       PowerShell with:   $env:WHISPER_DETECT_MODEL="small"

detect_model_cache = None             # loaded on demand
detect_model_lock = threading.Lock()  # stops two threads loading it twice


def detection_model():
    """
    PURPOSE : the model that guesses the language in Auto-detect.
    RETURNS : the live model by default, or the larger detection model when
              settings.DETECT_MODEL_NAME is set.
    """
    global detect_model_cache

    name = settings.DETECT_MODEL_NAME

    # Not configured (or same as the live model): use the live model
    if not name or name == settings.MODEL_SIZE:
        return model

    with detect_model_lock:
        if detect_model_cache is None:
            print("Loading the language-detection model:", name, "...")

            detect_model_cache = WhisperModel(
                name,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                cpu_threads=settings.CPU_THREADS
            )

            print("Language-detection model loaded.")

    return detect_model_cache


# =================================================================
# 6. DECODING OPTIONS
# =================================================================

def decode_options(accurate):
    """
    PURPOSE : how Whisper decodes (the speed vs. accuracy trade-off).
    accurate=False (live)  : fastest, one attempt, predictable delay.
    accurate=True  (files) : wider search + retries for hard passages.
    """
    return {
        # How many alternatives Whisper explores: more = better but slower
        "beam_size": 5 if accurate else 1,
        # Skip silent parts (see settings.VAD_SETTINGS)
        "vad_filter": True,
        "vad_parameters": settings.VAD_SETTINGS,
        # Remembering the previous sentence helps whole recordings but can
        # carry a mistake forward in live chunks, so live turns it off.
        "condition_on_previous_text": accurate,
        # Live: one attempt (0.0). Files: if a passage is garbled, retry
        # with a little randomness (0.2 ... 1.0).
        "temperature": (
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if accurate else 0.0
        )
    }
