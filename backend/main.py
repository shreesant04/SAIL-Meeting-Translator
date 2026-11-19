import os
import re
import json
import time
import asyncio
import tempfile
import threading

import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from faster_whisper import WhisperModel

# Optional packages that add the Roman-letters line
try:
    from indic_transliteration import sanscript
except Exception:
    sanscript = None

try:
    from korean_romanizer.romanizer import Romanizer
except Exception:
    Romanizer = None


# =====================================================
# SETTINGS (safe to change)
# =====================================================

# "base" is fast. "small" / "medium" / "large-v3" are more accurate
# but need a stronger computer. With a GPU, change DEVICE to "cuda"
# and COMPUTE_TYPE to "float16".
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")

CPU_THREADS = max(2, (os.cpu_count() or 4) - 1)

# Bigger, more accurate model used for recorded audio ("best accuracy").
# It is loaded only when first needed. It needs about 3 GB of RAM and is
# several times slower than real time on a CPU.
# Lighter choice: set WHISPER_ACCURATE_MODEL to "medium" (about 1.5 GB).
ACCURATE_MODEL = os.environ.get("WHISPER_ACCURATE_MODEL", "small")

# Browser sends audio at 16000 Hz
SAMPLE_RATE = 16000

# Chunks quieter than this are treated as silence and skipped
SILENCE_LEVEL = 0.003

# Audio is NEVER skipped while the model is busy: chunks wait in line.
# 120 chunks of 10 seconds = 20 minutes of waiting audio.
MAX_WAITING_CHUNKS = 120

# If Whisper hears Urdu (or writes Hindi in Urdu letters),
# try again and write it in Hindi letters (Devanagari)
TREAT_URDU_AS_HINDI = True

# Languages this app supports. Auto-detect chooses ONLY among these
# (English is included so a speaker's own English is not forced into
# another language).
SUPPORTED_LANGUAGES = ["hi", "bn", "ko"]
DETECTION_CANDIDATES = SUPPORTED_LANGUAGES + ["en"]

# --- Mixed-language live audio (Auto-detect only) ---
# True  = if the language guess for a chunk is not clear, split the chunk
#         at speech pauses and guess the language of each piece.
# False = one language per chunk.
MIXED_LANGUAGE_MODE = True

# The chunk is treated as ONE language when the best guess is at least this
# share of the supported languages (0.85 = 85%). Below that, it is split.
SPLIT_CONFIDENCE = 0.85

# At most this many pieces per chunk (each piece costs processing time)
MAX_PIECES = 4

# Pieces shorter than this are ignored; pieces shorter than the second
# value are too short to guess reliably and copy a neighbour's language
MIN_PIECE_SECONDS = 0.5
SHORT_PIECE_SECONDS = 1.5

# A pause this long (milliseconds) separates two pieces
PIECE_PAUSE_MS = 400

# True = do not show anything when English is detected
IGNORE_ENGLISH = False

# --- Translation method ---
# TWO-STEP (preferred): audio -> original text (Whisper)
#                       -> English text (offline text translation model)
# If the translation model is not installed, the app automatically
# falls back to Whisper's own audio -> English translation.
HERE = os.path.dirname(os.path.abspath(__file__))

MT_MODEL_DIR = os.environ.get(
    "MT_MODEL_DIR",
    os.path.join(HERE, "models", "nllb-600m-ct2")
)
MT_TOKENIZER_DIR = os.environ.get(
    "MT_TOKENIZER_DIR",
    os.path.join(HERE, "models", "nllb-tokenizer")
)

# Used only in FALLBACK mode:
# False = one Whisper pass (English only), True = also original text
SHOW_ORIGINAL_TEXT = False

HINDI_PROMPT = (
    "यह हिंदी भाषा में बोला गया है। "
    "कृपया हिंदी को देवनागरी लिपि में लिखें। "
    "उर्दू लिपि का उपयोग न करें।"
)

VAD_SETTINGS = {"min_silence_duration_ms": 500}


# =====================================================
# APP SETUP
# =====================================================

app = FastAPI(title="SAIL Meeting Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading Whisper model...")

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS
)

# Only one job may use the models at a time
model_lock = threading.Lock()

print("Whisper model loaded successfully.")


def is_arabic_char(ch):
    code = ord(ch)

    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def build_arabic_token_ids(whisper=None):
    """
    Finds every Whisper token that writes Arabic/Urdu letters.
    Used to FORCE Hindi speech to be written in Devanagari letters.
    """
    try:
        vocabulary = (whisper or model).hf_tokenizer.get_vocab()

        # Whisper tokens are byte-level; rebuild the byte <-> char map
        byte_values = (
            list(range(33, 127))
            + list(range(161, 173))
            + list(range(174, 256))
        )
        char_codes = byte_values[:]
        extra = 0

        for byte in range(256):
            if byte not in byte_values:
                byte_values.append(byte)
                char_codes.append(256 + extra)
                extra += 1

        char_to_byte = {
            chr(code): byte
            for byte, code in zip(byte_values, char_codes)
        }

        ids = []

        for token, token_id in vocabulary.items():
            try:
                raw = bytes(char_to_byte[ch] for ch in token)
            except KeyError:
                continue

            text = raw.decode("utf-8", errors="ignore")

            if any(is_arabic_char(ch) for ch in text):
                ids.append(token_id)

        return sorted(ids)

    except Exception as error:
        print("Could not prepare the Hindi-letters filter:", error)
        return []


ARABIC_TOKEN_IDS = build_arabic_token_ids()
print("Hindi-letters filter ready:", len(ARABIC_TOKEN_IDS), "tokens.")

arabic_cache = {id(model): ARABIC_TOKEN_IDS}


def arabic_ids(whisper):
    # One list per model (the big model has its own vocabulary)
    key = id(whisper)

    if key not in arabic_cache:
        arabic_cache[key] = build_arabic_token_ids(whisper)

    return arabic_cache[key]


# ---------------------------------------------------
# Language-specific ("specialist") speech models
# ---------------------------------------------------
# The general Whisper "base"/"small" models are very weak for Hindi
# (word error rate 149% / 67% on Google Fleurs). A model trained on
# Hindi is far better at the same size and speed.
# Put a converted model in these folders and it is used automatically
# for that language. See convert_model.py.
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

specialist_models = {}
specialist_failed = set()
specialist_lock = threading.Lock()


def specialist_available(code):
    folder = SPECIALIST_DIRS.get(code)

    return bool(folder) and os.path.exists(
        os.path.join(folder, "model.bin")
    )


def specialist_model(code):
    """The model trained for this language, or None (use the general one)."""
    if not code or code not in SPECIALIST_DIRS:
        return None

    if code in specialist_models:
        return specialist_models[code]

    if code in specialist_failed or not specialist_available(code):
        return None

    with specialist_lock:
        if code in specialist_models:
            return specialist_models[code]

        try:
            print("Loading the", code, "specialist model from",
                  SPECIALIST_DIRS[code], "...")

            specialist_models[code] = WhisperModel(
                SPECIALIST_DIRS[code],
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS
            )

            print("Specialist model for", code, "loaded.")

            return specialist_models[code]

        except Exception as error:
            specialist_failed.add(code)
            print(
                "Could not load the", code, "specialist model:", error,
                "\n -> Using the general model instead."
            )
            return None


print(
    "Roman letters: Hindi/Bengali",
    "ok" if sanscript is not None else "NOT installed (pip install indic-transliteration)",
    "| Korean",
    "ok" if Romanizer is not None else "NOT installed (pip install korean-romanizer)"
)

for _code in SPECIALIST_DIRS:
    if specialist_available(_code):
        specialist_model(_code)
    else:
        print("No specialist model for", _code,
              "(using the general", MODEL_SIZE, "model).")


accurate_model = None
accurate_lock = threading.Lock()


def get_accurate_model():
    """Loads the big model the first time it is needed."""
    global accurate_model

    if ACCURATE_MODEL == MODEL_SIZE:
        return model

    with accurate_lock:
        if accurate_model is None:
            print(
                "Loading the accurate model:", ACCURATE_MODEL,
                "(first time only; the download may be large)..."
            )

            accurate_model = WhisperModel(
                ACCURATE_MODEL,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS
            )

            print("Accurate model loaded.")

    return accurate_model


# =====================================================
# OFFLINE TEXT TRANSLATION MODEL (NLLB via CTranslate2)
# =====================================================

# Whisper language code -> NLLB language code
NLLB_CODES = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ko": "kor_Hang",
    "ur": "urd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "pt": "por_Latn"
}

translator = None
tokenizers = {}
MT_READY = False


def load_translation_model():
    global translator, MT_READY

    model_file = os.path.join(MT_MODEL_DIR, "model.bin")

    if not os.path.exists(model_file):
        print(
            "Translation model not found at:", MT_MODEL_DIR,
            "\n -> Using FALLBACK mode (Whisper audio-to-English)."
            "\n -> Run: python download_models.py"
        )
        return

    if not os.path.isdir(MT_TOKENIZER_DIR):
        print(
            "Tokenizer folder not found at:", MT_TOKENIZER_DIR,
            "\n -> Using FALLBACK mode (Whisper audio-to-English)."
            "\n -> Run: python download_models.py"
        )
        return

    try:
        import ctranslate2
        import transformers  # noqa: F401  (needed for the tokenizer)

        print("Loading text translation model...")

        translator = ctranslate2.Translator(
            MT_MODEL_DIR,
            device="cpu" if DEVICE == "cpu" else "cuda",
            compute_type="int8" if DEVICE == "cpu" else "float16",
            inter_threads=1,
            intra_threads=CPU_THREADS
        )

        # Quick self-test so a broken install is found at startup
        get_tokenizer("hin_Deva")

        MT_READY = True
        print("Text translation model loaded. Mode: TWO-STEP.")

    except Exception as error:
        translator = None
        MT_READY = False
        print(
            "Could not load the translation model:", error,
            "\n -> Using FALLBACK mode (Whisper audio-to-English)."
        )


def get_tokenizer(nllb_code):
    # One tokenizer per source language, created when first needed
    if nllb_code not in tokenizers:
        from transformers import AutoTokenizer

        tokenizers[nllb_code] = AutoTokenizer.from_pretrained(
            MT_TOKENIZER_DIR,
            src_lang=nllb_code
        )

    return tokenizers[nllb_code]


def has_arabic_script(text):
    return any("؀" <= ch <= "ۿ" for ch in text)


# Text that repeats the same short piece many times in a row
# (for example "vuvuvuvuvuvu") is a sign the model got stuck.
REPEAT_PATTERN = re.compile(r"(.{1,12}?)\1{5,}", re.DOTALL)
COLLAPSE_PATTERN = re.compile(r"(.{1,12}?)\1{3,}", re.DOTALL)

UNCLEAR_TEXT = "(unclear audio)"


def has_repetition(text):
    return bool(REPEAT_PATTERN.search(text))


def collapse_repetition(text):
    return COLLAPSE_PATTERN.sub(r"\1", text)


def clean_segments(segments):
    """
    Remove noise and mark garbled speech instead of showing nonsense.
    - Not speech (music, noise)            -> removed
    - Garbled or stuck in a repeating loop  -> shown as "(unclear audio)"
    """
    cleaned = []

    for item in segments:
        logprob = item.get("logprob", 0.0)
        no_speech = item.get("no_speech", 0.0)
        compression = item.get("compression", 1.0)

        # Whisper's own rule for "this is not speech"
        if no_speech > 0.6 and logprob < -1.0:
            continue

        garbled = (
            compression > 2.4
            or logprob < -1.2
            or has_repetition(item["text"])
        )

        if garbled:
            item = dict(item)
            item["text"] = UNCLEAR_TEXT
            item["unclear"] = True

        cleaned.append(item)

    return cleaned


def text_for_translation(item):
    # Garbled segments are not translated
    return "" if item.get("unclear") else item["text"]


def translate_to_english(texts, nllb_code, beam_size=1):
    """
    Translate a list of texts (one per segment) into English.
    Returns a list of the same length. Empty input gives empty output.
    """

    tokenizer = get_tokenizer(nllb_code)

    positions = [i for i, text in enumerate(texts) if text.strip()]
    english = [""] * len(texts)

    if not positions:
        return english

    sources = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(texts[i]))
        for i in positions
    ]

    # The English can't reasonably be much longer than the input
    longest = max(len(source) for source in sources)
    limit = min(200, int(longest * 2.5) + 10)

    results = translator.translate_batch(
        sources,
        target_prefix=[["eng_Latn"]] * len(sources),
        beam_size=beam_size,
        max_decoding_length=limit,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4
    )

    for position, result in zip(positions, results):
        tokens = result.hypotheses[0][1:]  # drop the language tag

        text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokens),
            skip_special_tokens=True
        ).strip()

        # Last safety net against a stuck repeating loop
        english[position] = collapse_repetition(text)

    return english


load_translation_model()


LANGUAGE_NAMES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ko": "Korean",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ja": "Japanese",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
    "ru": "Russian",
    "pt": "Portuguese"
}


def language_label(code):
    return LANGUAGE_NAMES.get(code, code.upper())


# =====================================================
# BASIC ENDPOINTS
# =====================================================

@app.get("/")
def home():
    return {"message": "SAIL Meeting Translator backend is running!"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "method": "two-step" if MT_READY else "whisper-direct",
        "live_model": MODEL_SIZE,
        "specialist_models": sorted(specialist_models),
        "accurate_model": ACCURATE_MODEL
    }


# =====================================================
# AUDIO PROCESSING
# =====================================================

def collect_segments(segments):
    result = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            result.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text,
                "logprob": round(getattr(segment, "avg_logprob", 0.0), 2),
                "no_speech": round(
                    getattr(segment, "no_speech_prob", 0.0), 2
                ),
                "compression": round(
                    getattr(segment, "compression_ratio", 1.0), 2
                )
            })

    return result


def decode_options(accurate):
    return {
        # Best accuracy: look at more alternatives and remember the
        # previous sentence (only sensible for whole recordings)
        "beam_size": 5 if accurate else 1,
        "vad_filter": True,
        "vad_parameters": VAD_SETTINGS,
        "condition_on_previous_text": accurate,
        # Live: one decoding attempt only (fast, predictable delay).
        # Files: allow retries for hard passages.
        "temperature": (
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if accurate else 0.0
        )
    }


def hindi_options(whisper):
    # No text prompt: Whisper can repeat prompt words in its output
    # ("देवनागरी नहीं है", "लिखा है"). Blocking Urdu letters is enough
    # to keep Hindi in Devanagari.
    return {
        "suppress_tokens": [-1] + arabic_ids(whisper)
    }


# Optional: a bigger general model used ONLY to guess the language.
# Language guessing is much more reliable with "small" than with "base",
# but it costs extra time for every chunk. Example (PowerShell):
#   $env:WHISPER_DETECT_MODEL="small"
DETECT_MODEL_NAME = os.environ.get("WHISPER_DETECT_MODEL", "")
detect_model_cache = None
detect_model_lock = threading.Lock()


def detection_model():
    """The model that guesses the language (default: the live model)."""
    global detect_model_cache

    if not DETECT_MODEL_NAME or DETECT_MODEL_NAME == MODEL_SIZE:
        return model

    with detect_model_lock:
        if detect_model_cache is None:
            print("Loading the language-detection model:",
                  DETECT_MODEL_NAME, "...")

            detect_model_cache = WhisperModel(
                DETECT_MODEL_NAME,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS
            )

            print("Language-detection model loaded.")

    return detect_model_cache


def detect_ranked(whisper, audio):
    """
    Language guess among the supported languages.
    Returns [(code, share), ...] best first (shares add up to 1), or None.
    """
    try:
        if isinstance(audio, str):
            from faster_whisper.audio import decode_audio

            audio = decode_audio(audio, sampling_rate=SAMPLE_RATE)

        # First try: look only at the parts that contain speech
        # (silence and music confuse the guess). If that is not
        # possible, look at everything.
        try:
            probabilities = whisper.detect_language(
                audio, vad_filter=True
            )[2]
        except Exception:
            probabilities = whisper.detect_language(audio)[2]

        ranked = [
            (code, probability)
            for code, probability in probabilities
            if code in DETECTION_CANDIDATES
        ]

        total = sum(probability for _, probability in ranked)

        if not ranked or total <= 0:
            return None

        return sorted(
            [(code, probability / total) for code, probability in ranked],
            key=lambda item: -item[1]
        )

    except Exception as error:
        print("Language choice failed, using normal auto-detect:", error)
        return None


def show_ranked(label, ranked):
    print(
        label,
        " | ".join(f"{code} {share:.2f}" for code, share in ranked)
    )


def pick_language(whisper, audio):
    """
    Auto-detect limited to the supported languages.
    Returns a language code, or None if detection is not possible
    (then Whisper's normal auto-detect is used).
    """
    ranked = detect_ranked(whisper, audio)

    if not ranked:
        return None

    show_ranked("Language check:", ranked)

    return ranked[0][0]


def speech_pieces(audio):
    """
    Splits audio at speech pauses. Returns [(start_sample, end_sample)],
    or [] if the voice detector is not available.
    """
    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions

        stamps = get_speech_timestamps(
            audio,
            VadOptions(
                min_silence_duration_ms=PIECE_PAUSE_MS,
                speech_pad_ms=150
            )
        )
    except Exception as error:
        print("Splitting at pauses is not available:", error)
        return []

    minimum = int(MIN_PIECE_SECONDS * SAMPLE_RATE)

    pieces = [
        (int(stamp["start"]), int(stamp["end"]))
        for stamp in stamps
        if int(stamp["end"]) - int(stamp["start"]) >= minimum
    ]

    # Too many pieces cost too much time: join the closest neighbours
    while len(pieces) > MAX_PIECES:
        gaps = [
            pieces[i + 1][0] - pieces[i][1]
            for i in range(len(pieces) - 1)
        ]

        i = gaps.index(min(gaps))
        pieces[i:i + 2] = [(pieces[i][0], pieces[i + 1][1])]

    return pieces


def plan_mixed(audio, detector):
    """
    Decides which language each part of a live chunk is in.
    Returns [(language, start_sample, end_sample), ...], or None when no
    guess is possible.
    - Clear single language -> one part (cheap, the normal case).
    - Unclear -> split at pauses, guess each piece, join neighbours that
      share a language.
    """
    chunk = detect_ranked(detector, audio)

    if chunk is None:
        return None

    show_ranked("Language check (whole chunk):", chunk)

    top, share = chunk[0]
    whole = [(top, 0, len(audio))]

    if share >= SPLIT_CONFIDENCE:
        return whole

    pieces = speech_pieces(audio)

    if len(pieces) < 2:
        return whole

    languages = []

    for start, end in pieces:
        seconds = (end - start) / SAMPLE_RATE

        if seconds < SHORT_PIECE_SECONDS:
            languages.append(None)
            continue

        ranked = detect_ranked(detector, audio[start:end])

        if ranked:
            show_ranked(
                f"  piece {start / SAMPLE_RATE:.1f}-"
                f"{end / SAMPLE_RATE:.1f}s:",
                ranked
            )

        languages.append(ranked[0][0] if ranked else None)

    if not any(languages):
        return whole

    # Pieces too short to guess take the language of a neighbour
    for i, language in enumerate(languages):
        if language is None:
            before = next(
                (languages[j] for j in range(i - 1, -1, -1)
                 if languages[j]),
                None
            )
            after = next(
                (languages[j] for j in range(i + 1, len(languages))
                 if languages[j]),
                None
            )
            languages[i] = before or after

    # Neighbours with the same language are transcribed together
    plan = []

    for (start, end), language in zip(pieces, languages):
        if plan and plan[-1][0] == language:
            plan[-1] = (language, plan[-1][1], end)
        else:
            plan.append((language, start, end))

    return plan


def forced_language(language, whisper=None, audio=None):
    """
    The language the user picked. For auto-detect: the most likely of
    the supported languages (or None if that could not be worked out).
    """
    if language in (None, "", "auto"):
        if whisper is not None and audio is not None:
            return pick_language(whisper, audio)

        return None

    return language


def transcribe_original(audio, whisper=None, language=None, accurate=False):
    """
    Whisper: audio -> text in the language that was spoken.
    Returns: original segments, language code, probability
    """

    whisper = whisper or model
    forced = forced_language(language, whisper, audio)

    # Use the model trained for this language when we have one
    specialist = specialist_model(forced)

    if specialist is not None:
        whisper = specialist

    options = decode_options(accurate)
    extra = {}

    if forced:
        extra["language"] = forced

        # Hindi: write Hindi letters straight away
        if forced == "hi" and TREAT_URDU_AS_HINDI:
            extra.update(hindi_options(whisper))

    segments, info = whisper.transcribe(
        audio,
        task="transcribe",
        **options,
        **extra
    )

    original = collect_segments(segments)
    language_code = info.language
    probability = info.language_probability

    if not original:
        return [], language_code, probability

    joined = " ".join(item["text"] for item in original)

    # Hindi/Urdu written in Urdu letters: try again for Devanagari
    if (
        TREAT_URDU_AS_HINDI
        and forced != "ur"
        and language_code in ("hi", "ur")
        and has_arabic_script(joined)
    ):
        retry_segments, _ = whisper.transcribe(
            audio,
            task="transcribe",
            language="hi",
            **options,
            **hindi_options(whisper)
        )

        retry = collect_segments(retry_segments)

        if retry:
            original = retry
            language_code = "hi"

    original = clean_segments(original)

    return original, language_code, probability


def process_two_step(audio, whisper=None, language=None, accurate=False):
    original, language_code, probability = transcribe_original(
        audio, whisper, language, accurate
    )

    if not original:
        return [], [], language_code, probability

    # English speech needs no translation
    if language_code == "en":
        return original, original, language_code, probability

    joined = " ".join(item["text"] for item in original)

    # Choose the source language by the SCRIPT that was written
    if language_code in ("hi", "ur"):
        nllb_code = (
            "urd_Arab" if has_arabic_script(joined) else "hin_Deva"
        )
    else:
        nllb_code = NLLB_CODES.get(language_code)

    # Language not supported by the text model: use Whisper instead
    if nllb_code is None:
        translated = process_whisper_translate(
            audio, whisper, language, accurate
        )[1]
        return original, translated, language_code, probability

    english_texts = translate_to_english(
        [text_for_translation(item) for item in original],
        nllb_code,
        beam_size=4 if accurate else 1
    )

    translated = []

    for item, text in zip(original, english_texts):
        if text:
            translated.append({
                "start": item["start"],
                "end": item["end"],
                "text": text
            })

    return original, translated, language_code, probability


def process_whisper_translate(
    audio, whisper=None, language=None, accurate=False
):
    """
    FALLBACK: Whisper translates the audio straight to English.
    Returns: original, translated, language, probability
    """

    whisper = whisper or model
    forced = forced_language(language, whisper, audio)
    options = decode_options(accurate)

    if forced:
        options["language"] = forced

    translated_segments, info = whisper.transcribe(
        audio,
        task="translate",
        **options
    )

    translated = collect_segments(translated_segments)

    language_code = info.language
    probability = info.language_probability

    if not translated:
        return [], [], language_code, probability

    if language_code == "en":
        return translated, translated, language_code, probability

    transcribe_language = language_code
    extra = {}

    if language_code == "ur" and TREAT_URDU_AS_HINDI and forced != "ur":
        transcribe_language = "hi"
        language_code = "hi"

    if not SHOW_ORIGINAL_TEXT and not accurate:
        return [], translated, language_code, probability

    if transcribe_language == "hi":
        extra.update(hindi_options(whisper))

    original_options = decode_options(accurate)

    original_segments, _ = whisper.transcribe(
        audio,
        task="transcribe",
        language=transcribe_language,
        **original_options,
        **extra
    )

    original = collect_segments(original_segments)

    return original, translated, language_code, probability


def process_audio(audio, whisper=None, language=None, accurate=False):
    with model_lock:
        if MT_READY:
            return process_two_step(audio, whisper, language, accurate)

        return process_whisper_translate(
            audio, whisper, language, accurate
        )


def process_file(path, language, accuracy):
    """Uploaded file / recording. 'best' uses the big model."""
    accurate = accuracy == "best"
    whisper = get_accurate_model() if accurate else model

    return process_audio(
        path,
        whisper=whisper,
        language=language,
        accurate=accurate
    )


# =====================================================
# ROMAN LETTERS (Hinglish-style) FOR THE ORIGINAL TEXT
# =====================================================
# Shows how the original text sounds in English letters, for example
# Hindi "नमस्ते कैसे हैं" -> "namaste kaise hain".
# It is an approximation (a rule-based transliteration), not a
# translation. Hindi and Bengali use "indic-transliteration",
# Korean uses "korean-romanizer". If a package is missing, the
# Roman line is simply left out.

HINGLISH_MAP = [
    ("RRi", "ri"), ("R^i", "ri"), ("RRI", "ri"),
    ("kSh", "ksh"), ("GY", "gy"), ("j~n", "gy"),
    ("Chh", "chh"), ("Ch", "chh"), ("shh", "sh"), ("Sh", "sh"),
    ("~N", "n"), ("~n", "n"), (".n", "n"), (".N", "n"),
    (".h", ""), (".a", ""),
    ("Th", "th"), ("Dh", "dh"),
    ("A", "aa"), ("I", "ee"), ("U", "oo"),
    ("M", "n"), ("H", "h"), ("N", "n"),
    ("T", "t"), ("D", "d"), ("L", "l"),
    ("K", "kh"), ("G", "g"), ("J", "z"), ("Q", "q"),
    ("||", "."), ("|", ".")
]

_CONSONANT_CLASS = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
_FINAL_SCHWA = re.compile(
    "(?<=[" + _CONSONANT_CLASS + "])a(?![A-Za-z])"
)
_MIDDLE_SCHWA = re.compile(
    "(?<=[aeiouAEIOU][bcdfghjklmnpqrstvwxyz])a"
    "(?=[bcdfghjklmnpqrstvwxyz][aeiouAEIOU])"
)


def hinglishify(text, drop_schwa):
    """ITRANS letters -> simple everyday Roman letters."""
    words = []

    for word in text.split(" "):
        if drop_schwa and len(word) > 2:
            # Hindi does not pronounce the 'a' at the end of a word
            # (सब = "sab") or in the middle of many words (करता = "karta")
            word = _FINAL_SCHWA.sub("", word)
            word = _MIDDLE_SCHWA.sub("", word)

        for old, new in HINGLISH_MAP:
            word = word.replace(old, new)

        words.append(word)

    return " ".join(words)


def romanize(text, code):
    """Original text in Roman letters, or "" if not possible."""
    try:
        if code == "ko" and Romanizer is not None:
            result = Romanizer(text).romanize()
        elif code in ("hi", "bn") and sanscript is not None:
            source = (
                sanscript.DEVANAGARI if code == "hi"
                else sanscript.BENGALI
            )

            itrans = sanscript.transliterate(
                text, source, sanscript.ITRANS
            )

            result = hinglishify(itrans, drop_schwa=(code == "hi"))
        else:
            return ""

        result = result.strip()

        return "" if result == text.strip() else result

    except Exception as error:
        print("Roman letters failed:", error)
        return ""


# =====================================================
# STREAMING RESULTS (used by live audio AND uploaded files)
# =====================================================
# Results are sent segment by segment, as soon as each one is ready:
#   1. the original text (+ Roman letters) appears first,
#   2. then its English meaning appears on the same row.
# So the screen starts filling after the first sentence instead of
# after the whole chunk or whole file.

def nllb_code_for(code, text):
    if code in ("hi", "ur"):
        return "urd_Arab" if has_arabic_script(text) else "hin_Deva"

    return NLLB_CODES.get(code)


def _stream_language(
    audio,
    forced,
    accurate,
    cancel,
    skip_english,
    start_index=0,
    offset=0.0,
    total_duration=None
):
    """
    Transcribes + translates ONE stretch of audio in ONE language and
    yields the events. Must be called while model_lock is held.
    Returns the next free row index.
    """

    if total_duration is None:
        total_duration = len(audio) / SAMPLE_RATE

    # Use the model trained for the language when we have one.
    # The big general model is loaded ONLY when nothing better exists.
    specialist = specialist_model(forced)

    if specialist is not None:
        whisper = specialist
    elif accurate:
        whisper = get_accurate_model()
    else:
        whisper = model

    options = decode_options(accurate)
    extra = {}

    if forced:
        extra["language"] = forced

        if forced == "hi" and TREAT_URDU_AS_HINDI:
            extra.update(hindi_options(whisper))

    segments, info = whisper.transcribe(
        audio,
        task="transcribe",
        **options,
        **extra
    )

    code = info.language
    name = language_label(code)

    yield {
        "kind": "language",
        "language": code,
        "language_name": name,
        "probability": info.language_probability
    }

    if code == "en" and skip_english:
        return start_index

    index = start_index

    for raw in segments:

        if cancel is not None and cancel.is_set():
            return index

        items = clean_segments(collect_segments([raw]))

        if not items:
            continue

        item = items[0]
        unclear = bool(item.get("unclear"))
        text = item["text"]

        begin = round(offset + item["start"], 2)
        finish = round(offset + item["end"], 2)

        # 1) ORIGINAL TEXT FIRST
        yield {
            "kind": "original",
            "index": index,
            "start": begin,
            "end": finish,
            "text": text,
            "roman": "" if unclear or code == "en"
                     else romanize(text, code),
            "language": code,
            "language_name": name
        }

        # 2) THEN ITS ENGLISH MEANING
        if unclear:
            english = ""
        elif code == "en":
            english = text
        else:
            nllb_code = nllb_code_for(code, text)

            english = (
                translate_to_english(
                    [text],
                    nllb_code,
                    beam_size=4 if accurate else 1
                )[0]
                if nllb_code
                else ""
            )

        yield {
            "kind": "translation",
            "index": index,
            "start": begin,
            "end": finish,
            "text": english,
            "language": code,
            "language_name": name
        }

        index += 1

        if total_duration > 0:
            yield {
                "kind": "progress",
                "percent": min(99, int(finish / total_duration * 100))
            }

    return index


def iter_results(
    audio,
    language="auto",
    accurate=False,
    cancel=None,
    skip_english=False,
    mixed=False
):
    """
    Generator (runs in a worker thread). Yields dictionaries:
      {"kind": "language", ...}     first (once per language stretch)
      {"kind": "original", "index": i, ...}
      {"kind": "translation", "index": i, ...}
      {"kind": "progress", "percent": n}
    mixed=True (live Auto-detect): a chunk with several languages is
    split at speech pauses and each piece gets its own language.
    """

    with model_lock:

        if isinstance(audio, str):
            from faster_whisper.audio import decode_audio

            audio = decode_audio(audio, sampling_rate=SAMPLE_RATE)

        duration = len(audio) / SAMPLE_RATE
        auto = language in (None, "", "auto")
        detector = detection_model()

        if auto and mixed and MIXED_LANGUAGE_MODE:
            plan = plan_mixed(audio, detector)

            if plan and len(plan) > 1:
                index = 0

                for part_language, start, end in plan:
                    if cancel is not None and cancel.is_set():
                        return

                    index = yield from _stream_language(
                        audio[start:end],
                        part_language,
                        accurate,
                        cancel,
                        skip_english,
                        start_index=index,
                        offset=start / SAMPLE_RATE,
                        total_duration=duration
                    )

                return

            forced = plan[0][0] if plan else None

        else:
            forced = forced_language(language, detector, audio)

        yield from _stream_language(
            audio,
            forced,
            accurate,
            cancel,
            skip_english,
            total_duration=duration
        )


def iter_fallback(path, language, accuracy):
    """Same events when the text translation model is not installed."""
    original, translated, code, probability = process_file(
        path, language, accuracy
    )

    name = language_label(code)

    yield {
        "kind": "language",
        "language": code,
        "language_name": name,
        "probability": probability
    }

    for i in range(max(len(original), len(translated))):
        base = original[i] if i < len(original) else translated[i]

        if i < len(original):
            yield {
                "kind": "original", "index": i,
                "start": base["start"], "end": base["end"],
                "text": original[i]["text"],
                "roman": romanize(original[i]["text"], code),
                "language": code, "language_name": name
            }

        if i < len(translated):
            yield {
                "kind": "translation", "index": i,
                "start": base["start"], "end": base["end"],
                "text": translated[i]["text"],
                "language": code, "language_name": name
            }


def wire(event, base_id=0, offset=0.0):
    """Turns an internal event into the message the page understands."""
    kind = event["kind"]

    if kind == "original":
        return {
            "type": "original_transcript",
            "id": base_id + event["index"],
            "language": event["language"],
            "language_name": event["language_name"],
            "start": round(offset + event["start"], 2),
            "end": round(offset + event["end"], 2),
            "text": event["text"],
            "roman": event.get("roman", "")
        }

    if kind == "translation":
        return {
            "type": "translation",
            "id": base_id + event["index"],
            "language": "en",
            "language_name": "English",
            "source_language_name": event["language_name"],
            "start": round(offset + event["start"], 2),
            "end": round(offset + event["end"], 2),
            "text": event["text"]
        }

    return None


async def run_stream(make_iterator):
    """
    Runs a generator in a worker thread and hands its events to
    async code one by one, as soon as they are produced.
    """
    loop = asyncio.get_running_loop()
    box = asyncio.Queue()

    def work():
        try:
            for event in make_iterator():
                loop.call_soon_threadsafe(box.put_nowait, event)
        except Exception as error:
            loop.call_soon_threadsafe(
                box.put_nowait,
                {"kind": "error", "message": str(error)}
            )
        finally:
            loop.call_soon_threadsafe(box.put_nowait, None)

    threading.Thread(target=work, daemon=True).start()

    while True:
        event = await box.get()

        if event is None:
            break

        yield event


def run_transcribe(audio, language=None):
    """Live stage 1: audio -> original text (two-step mode)."""
    with model_lock:
        return transcribe_original(audio, language=language)


def run_translate(original, language, audio):
    """
    Live stage 2: original text -> English text (two-step mode).
    Returns a list of English texts, one per original segment,
    in the same order (so rows always match).
    """
    with model_lock:
        joined = " ".join(item["text"] for item in original)

        if language in ("hi", "ur"):
            nllb_code = (
                "urd_Arab" if has_arabic_script(joined) else "hin_Deva"
            )
        else:
            nllb_code = NLLB_CODES.get(language)

        if nllb_code is None:
            # Language not supported by the text model:
            # let Whisper translate this chunk, shown in the first row
            whisper_english = process_whisper_translate(audio)[1]
            text = " ".join(item["text"] for item in whisper_english)
            return [text] + [""] * (len(original) - 1)

        return translate_to_english(
            [text_for_translation(item) for item in original],
            nllb_code
        )


# =====================================================
# UPLOADED FILE TRANSLATION
# =====================================================

# =====================================================
# TALKBACK VOICE (Piper, offline text-to-speech)
# =====================================================
# Looks for a Piper voice file (for example en_US-lessac-medium.onnx).
# To choose one yourself:  $env:PIPER_VOICE="C:\\path\\voice.onnx"

piper_lock = threading.Lock()
piper_cache = {"voice": None, "path": None, "engine": None, "tried": False}


def find_piper_voice():
    wanted = os.environ.get("PIPER_VOICE", "").strip()

    if wanted and os.path.exists(wanted):
        return wanted

    home = os.path.expanduser("~")
    folders = [
        os.path.join(HERE, "models"),
        HERE,
        os.path.join(HERE, ".."),
        os.path.join(home, "piper"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Documents"),
    ]

    found = []

    for folder in folders:
        for root, dirs, files in os.walk(folder):
            # do not wander into huge folders
            depth = root[len(folder):].count(os.sep)

            if depth >= 3:
                dirs[:] = []

            dirs[:] = [d for d in dirs if d not in
                       ("node_modules", ".git", "venv", ".venv",
                        "convert_env", "__pycache__")]

            for name in files:
                if name.lower().endswith(".onnx") and name.lower().startswith("en_"):
                    found.append(os.path.join(root, name))

    # prefer the lessac voice when several exist
    found.sort(key=lambda path: (0 if "lessac" in path.lower() else 1, path))

    return found[0] if found else None


def load_piper():
    """Returns (engine, voice_or_path, path) or None when Piper is not usable."""
    with piper_lock:
        if piper_cache["tried"]:
            if piper_cache["engine"]:
                return (piper_cache["engine"], piper_cache["voice"],
                        piper_cache["path"])
            return None

        piper_cache["tried"] = True
        path = find_piper_voice()

        if not path:
            print("Talkback: no Piper voice file (.onnx) found.")
            return None

        try:
            from piper import PiperVoice

            voice = PiperVoice.load(path)
            piper_cache.update(engine="python", voice=voice, path=path)
            print("Talkback: Piper voice loaded:", path)

        except Exception as error:
            import shutil

            exe = shutil.which("piper") or shutil.which("piper.exe")

            if exe:
                piper_cache.update(engine="exe", voice=exe, path=path)
                print("Talkback: using piper program:", exe, "with", path)
            else:
                print("Talkback: Piper is not usable:", error)
                return None

        return (piper_cache["engine"], piper_cache["voice"],
                piper_cache["path"])


def piper_wav(text):
    """English text -> WAV bytes (offline). Raises if Piper is unavailable."""
    import io
    import wave
    import subprocess

    loaded = load_piper()

    if not loaded:
        raise RuntimeError("Piper voice is not available")

    engine, voice, path = loaded

    if engine == "python":
        buffer = io.BytesIO()

        with piper_lock:
            with wave.open(buffer, "wb") as wav_file:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wav_file)
                else:
                    voice.synthesize(text, wav_file)

        return buffer.getvalue()

    with tempfile.TemporaryDirectory() as folder:
        out = os.path.join(folder, "speech.wav")

        subprocess.run(
            [voice, "--model", path, "--output_file", out],
            input=text.encode("utf-8"),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )

        with open(out, "rb") as handle:
            return handle.read()


@app.get("/speak-status")
async def speak_status():
    loaded = await asyncio.get_event_loop().run_in_executor(
        None, load_piper
    )

    return {
        "available": bool(loaded),
        "voice": os.path.basename(loaded[2]) if loaded else ""
    }


@app.post("/speak")
async def speak(text: str = Form(...)):
    text = (text or "").strip()[:600]

    if not text:
        raise HTTPException(status_code=400, detail="No text")

    try:
        wav = await asyncio.get_event_loop().run_in_executor(
            None, piper_wav, text
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error))

    return Response(content=wav, media_type="audio/wav")


@app.post("/translate-audio")
async def translate_audio(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    accuracy: str = Form("best")
):
    """
    Streams results line by line (JSON lines). Each sentence appears as
    soon as it is ready: original text first, then its English meaning.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file selected."
        )

    suffix = os.path.splitext(file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    cancel = threading.Event()

    def make_iterator():
        try:
            if MT_READY:
                yield from iter_results(
                    temp_path, language, accuracy == "best", cancel
                )
            else:
                yield from iter_fallback(temp_path, language, accuracy)
        finally:
            # The temporary copy of the audio is always deleted
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def line(message):
        return json.dumps(message, ensure_ascii=False) + "\n"

    async def body():
        started = time.perf_counter()

        try:
            yield line({
                "type": "status",
                "message": "Reading the audio file..."
            })

            async for event in run_stream(make_iterator):
                kind = event["kind"]

                if kind == "error":
                    print("File translation error:", event["message"])
                    yield line({
                        "type": "error",
                        "message": event["message"]
                    })
                    return

                if kind == "language":
                    yield line({
                        "type": "language",
                        "language": event["language"],
                        "language_name": event["language_name"]
                    })
                    continue

                if kind == "progress":
                    yield line({
                        "type": "progress",
                        "percent": event["percent"]
                    })
                    continue

                message = wire(event)

                if message:
                    yield line(message)

            print(
                "File processed in",
                round(time.perf_counter() - started, 1), "s"
            )

            yield line({"type": "done"})

        finally:
            # If the page was closed or cancelled, stop the work
            cancel.set()

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson"
    )


# =====================================================
# LIVE BROWSER AUDIO
# =====================================================

@app.websocket("/ws/browser-live")
async def browser_live_translation(websocket: WebSocket):

    await websocket.accept()

    chosen_language = websocket.query_params.get("language", "auto")

    if chosen_language not in DETECTION_CANDIDATES:
        chosen_language = "auto"

    print("Live connection opened. Language:", chosen_language)

    # Audio chunks wait here while the model is busy
    queue = asyncio.Queue(maxsize=MAX_WAITING_CHUNKS)

    next_id = 0

    async def send_status(message):
        await websocket.send_json({"type": "status", "message": message})

    async def send_original(segment, row_id, offset, language, name):
        await websocket.send_json({
            "type": "original_transcript",
            "id": row_id,
            "language": language,
            "language_name": name,
            "start": round(offset + segment["start"], 2),
            "end": round(offset + segment["end"], 2),
            "text": segment["text"],
            "roman": (
                ""
                if language == "en"
                or segment["text"] == UNCLEAR_TEXT
                else romanize(segment["text"], language)
            )
        })

    async def send_english(text, segment, row_id, offset, name):
        await websocket.send_json({
            "type": "translation",
            "id": row_id,
            "language": "en",
            "language_name": "English",
            "source_language_name": name,
            "start": round(offset + segment["start"], 2),
            "end": round(offset + segment["end"], 2),
            "text": text
        })

    async def idle_status():
        waiting = queue.qsize()

        await send_status(
            "Listening to system audio..."
            if waiting == 0
            else f"Catching up: {waiting} audio chunk(s) "
                 "waiting. Nothing is skipped."
        )

    async def worker():
        # Takes chunks from the queue, one at a time.
        # An error in one chunk never closes the connection.
        nonlocal next_id

        try:
            while True:
                audio, offset = await queue.get()
                seconds = round(len(audio) / SAMPLE_RATE, 1)

                try:
                    started = time.perf_counter()

                    if MT_READY:
                        # Results are sent sentence by sentence:
                        # original text first, then its English.
                        await send_status("Transcribing speech...")

                        base_id = next_id
                        count = 0
                        name = ""
                        probability = 0.0

                        async for event in run_stream(
                            lambda: iter_results(
                                audio,
                                chosen_language,
                                False,
                                None,
                                IGNORE_ENGLISH,
                                True
                            )
                        ):
                            kind = event["kind"]

                            if kind == "error":
                                raise RuntimeError(event["message"])

                            if kind == "language":
                                name = event["language_name"]
                                probability = event["probability"]
                                continue

                            message = wire(event, base_id, offset)

                            if message:
                                await websocket.send_json(message)
                                count = max(count, event["index"] + 1)

                        next_id += count

                        print(
                            "Processed", seconds, "s of audio in",
                            round(time.perf_counter() - started, 1), "s",
                            "| Detected:", name or "-",
                            "| Confidence:", round(probability, 2),
                            "| Segments:", count
                        )

                    else:
                        # ---- FALLBACK: Whisper audio -> English ----
                        original, translated, language, probability = (
                            await asyncio.to_thread(process_audio, audio)
                        )

                        print(
                            "Processed", seconds, "s of audio in",
                            round(time.perf_counter() - started, 1), "s"
                        )

                        if not original and not translated:
                            await idle_status()
                            continue

                        if language == "en" and IGNORE_ENGLISH:
                            await idle_status()
                            continue

                        name = language_label(language)
                        base_id = next_id
                        next_id += max(len(original), len(translated))

                        for index, segment in enumerate(original):
                            await send_original(
                                segment, base_id + index,
                                offset, language, name
                            )

                        for index, segment in enumerate(translated):
                            await send_english(
                                segment["text"], segment,
                                base_id + index, offset, name
                            )

                except Exception as error:
                    print("Chunk processing error:", error)
                    await send_status("Processing error: " + str(error))
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {error}"
                    })
                    continue

                await idle_status()

        except asyncio.CancelledError:
            raise
        except Exception as error:
            print("Worker stopped:", error)

    worker_task = asyncio.create_task(worker())
    audio_offset = 0.0

    try:
        await websocket.send_json({
            "type": "status",
            "message": (
                "Browser audio connection established. Method: "
                + ("two-step (text translation)." if MT_READY
                   else "Whisper direct (fallback).")
            )
        })

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                print("Browser closed the connection.")
                break

            audio_bytes = message.get("bytes")

            # Float32 audio needs complete 4-byte samples
            if not audio_bytes or len(audio_bytes) % 4 != 0:
                continue

            audio = np.frombuffer(
                audio_bytes,
                dtype=np.float32
            ).copy()

            if len(audio) == 0:
                continue

            duration = len(audio) / SAMPLE_RATE
            offset = audio_offset
            audio_offset += duration

            level = float(np.sqrt(np.mean(audio ** 2)))

            print(
                "Audio received:",
                round(duration, 2), "seconds",
                "| Level:", round(level, 4)
            )

            # Skip silence to save processing time
            if level < SILENCE_LEVEL:
                continue

            # Only if the line is extremely long (memory protection)
            if queue.full():
                try:
                    queue.get_nowait()
                    print("Line too long: dropped one old audio chunk.")
                    await websocket.send_json({
                        "type": "status",
                        "message": (
                            "Translation is far behind. "
                            "The oldest audio was skipped."
                        )
                    })
                except asyncio.QueueEmpty:
                    pass

            queue.put_nowait((audio, offset))

            if queue.qsize() >= 2:
                await websocket.send_json({
                    "type": "status",
                    "message": (
                        f"Catching up: {queue.qsize()} audio chunk(s) "
                        "waiting. Nothing is skipped."
                    )
                })

    except WebSocketDisconnect:
        print("Browser audio connection disconnected.")

    except Exception as error:
        print("Live connection error:", error)

    finally:
        worker_task.cancel()
        print("Live connection closed.")