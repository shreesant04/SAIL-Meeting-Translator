"""
pipeline.py
===========
PURPOSE
    The CORE job of the app:  AUDIO -> ORIGINAL TEXT -> ENGLISH TEXT,
    produced as a stream of small events, so the screen fills sentence by
    sentence instead of waiting for a whole chunk or file.

THE EVENTS (plain dictionaries) produced by iter_results():
    {"kind": "language",    ...}   which language was heard
    {"kind": "original",    ...}   original text (+ Roman letters)  - FIRST
    {"kind": "translation", ...}   its English meaning - SECOND, same index
    {"kind": "progress",    ...}   percent finished (files only)
    The "original" and "translation" events of one sentence share the same
    `index`, so the page puts them on the SAME row, always matching.

WHO USES THIS FILE
    routes_live.py  - live audio, chunk by chunk (through run_stream)
    routes_http.py  - uploaded / recorded files (through run_stream)

FILE MAP
    1. stream_language   one stretch of audio in ONE language
    2. iter_results      a whole chunk/file, with mixed-language handling
    3. fallback          used only when the translation model is missing
    4. wire, run_stream  messages for the web page and the thread bridge
"""

import asyncio     # run_stream hands events from a thread to async code
import threading   # run_stream starts the worker thread

import settings                                          # all settings
from languages import language_label, nllb_code_for      # names / NLLB codes
from language_detect import plan_mixed, forced_language  # which language?
from romanize import romanize                            # Roman letters line
from speech_models import (
    model,              # the general live model
    model_lock,         # only one job may use the models at a time
    specialist_model,   # model trained for one language (or None)
    get_accurate_model, # bigger model for files
    detection_model,    # model that guesses the language
    decode_options,     # speed vs. accuracy options
    hindi_options       # forbids Urdu letters for Hindi
)
from text_cleanup import collect_segments, clean_segments   # quality filter
from translation import translate_to_english                # NLLB translation

# Short local name, used several times below
SAMPLE_RATE = settings.SAMPLE_RATE


# =====================================================
# 1. ONE STRETCH OF AUDIO IN ONE LANGUAGE
# =====================================================

def stream_language(
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
    PURPOSE : transcribe and translate ONE stretch of audio that is in ONE
              language, yielding events as soon as each sentence is ready.
    WHY     : this is the building block. A live chunk with one language
              calls it once; a mixed chunk calls it once per piece.
    NOTE    : must be called while model_lock is held (iter_results does).
    RETURNS : the next free row index (so the next piece continues the
              numbering and rows never clash).

    INPUT   : audio           samples of this stretch
              forced          language code to use (None = Whisper decides)
              accurate        True for files (slower, more careful)
              cancel          threading.Event; when set, work stops early
              skip_english    True = say nothing when English is heard
              start_index     first row number of this stretch
              offset          seconds added to every timestamp (so a piece
                              in the middle of a chunk shows its true time)
              total_duration  seconds of the WHOLE audio (progress percent)
    """

    # Duration of the whole audio, if the caller did not say
    if total_duration is None:
        total_duration = len(audio) / SAMPLE_RATE

    # ---- STEP 1: choose the model (specialist > accurate > general) ----
    # WHY: a model trained for the language is the most accurate; the big
    # "accurate" model is only used for files without a specialist; the
    # general live model is the last resort.
    specialist = specialist_model(forced)

    if specialist is not None:
        whisper = specialist
    elif accurate:
        whisper = get_accurate_model()
    else:
        whisper = model

    # ---- STEP 2: decoding options ----
    options = decode_options(accurate)
    extra = {}

    if forced:
        # Tell Whisper the language instead of letting it guess again
        extra["language"] = forced

        # Hindi: forbid Urdu letters so the text comes out in Devanagari
        if forced == "hi" and settings.TREAT_URDU_AS_HINDI:
            extra.update(hindi_options(whisper))

    # ---- STEP 3: speech to text ----
    # `segments` is LAZY: the model produces one sentence at a time while
    # the loop below consumes it. That is what makes the first sentence
    # appear before the whole chunk is finished.
    segments, info = whisper.transcribe(
        audio,
        task="transcribe",
        **options,
        **extra
    )

    code = info.language            # language that was actually used
    name = language_label(code)     # readable name ("Hindi")

    # Tell the page which language was heard (for the "Detected language"
    # label)
    yield {
        "kind": "language",
        "language": code,
        "language_name": name,
        "probability": info.language_probability
    }

    # English speech can be hidden (settings.IGNORE_ENGLISH)
    if code == "en" and skip_english:
        return start_index

    index = start_index   # row number of the next sentence

    # ---- STEP 4: loop over the sentences as they are produced ----
    # LOOP - belongs to: stream_language(), step 4
    # WHY  : each sentence must be cleaned, shown in its original
    #        language and then translated, ONE AT A TIME, so the screen
    #        updates sentence by sentence.
    # REPEATS: once per sentence (Whisper segment) of this stretch.
    for raw in segments:

        # Stop early if the user pressed Stop or closed the page
        if cancel is not None and cancel.is_set():
            return index

        # Quality filter: noise removed, garbled text marked "unclear"
        items = clean_segments(collect_segments([raw]))

        # Nothing left after the filter (it was noise): next sentence
        if not items:
            continue

        item = items[0]
        unclear = bool(item.get("unclear"))
        text = item["text"]

        # Real position in the audio (piece offset + position in the piece)
        begin = round(offset + item["start"], 2)
        finish = round(offset + item["end"], 2)

        # ---- 4a) ORIGINAL TEXT FIRST (with Roman letters) ----
        # WHY first: the user sees what was said immediately, and the
        # English follows a moment later on the same row.
        yield {
            "kind": "original",
            "index": index,
            "start": begin,
            "end": finish,
            "text": text,
            # No Roman line for garbled text or English
            "roman": "" if unclear or code == "en"
                     else romanize(text, code),
            "language": code,
            "language_name": name
        }

        # ---- 4b) THEN ITS ENGLISH MEANING ----
        if unclear:
            english = ""                      # garbled: nothing to translate
        elif code == "en":
            english = text                    # already English
        else:
            # Language code the translation model understands
            nllb_code = nllb_code_for(code, text)

            # Translate just this sentence (None code = language the model
            # does not know, so no translation)
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
            "index": index,               # same index = same row
            "start": begin,
            "end": finish,
            "text": english,
            "language": code,
            "language_name": name
        }

        index += 1   # next sentence gets the next row

        # Progress percent (used by file translation)
        if total_duration > 0:
            yield {
                "kind": "progress",
                "percent": min(99, int(finish / total_duration * 100))
            }

    return index


# =====================================================
# 2. WHOLE AUDIO (LIVE CHUNK OR FILE), WITH MIXED-LANGUAGE HANDLING
# =====================================================

def iter_results(
    audio,
    language="auto",
    accurate=False,
    cancel=None,
    skip_english=False,
    mixed=False
):
    """
    PURPOSE : generator (runs in a worker thread) that yields the events
              described at the top of this file for a whole chunk or file.
    WHY     : decides HOW the audio is split by language and calls
              stream_language() for each part.

    INPUT   : audio     samples (numpy array) or a file path
              language  "hi", "bn", "ko", "en" or "auto"
              mixed     True (live Auto-detect only): a chunk that contains
                        several languages is split at speech pauses and
                        each piece handled in its own language
    """

    # Take the model lock: only one job at a time may use the models.
    # Others wait here, which is what makes the live queue "never skip".
    with model_lock:

        # A file path is decoded into samples first
        if isinstance(audio, str):
            from faster_whisper.audio import decode_audio

            audio = decode_audio(audio, sampling_rate=SAMPLE_RATE)

        duration = len(audio) / SAMPLE_RATE
        auto = language in (None, "", "auto")
        detector = detection_model()

        # ---- CASE 1: live Auto-detect with mixed-language support ----
        if auto and mixed and settings.MIXED_LANGUAGE_MODE:
            # Decide which language each part of the chunk is in
            plan = plan_mixed(audio, detector)

            # More than one part: several languages in this chunk
            if plan and len(plan) > 1:
                index = 0   # row numbers continue from part to part

                # LOOP - belongs to: iter_results(), CASE 1
                # WHY  : each part may be in a different language, so each
                #        is transcribed with the model of ITS language.
                # REPEATS: once per part of the plan (language switch).
                for part_language, start, end in plan:
                    if cancel is not None and cancel.is_set():
                        return

                    # `yield from` passes the events straight through and
                    # gives back the next free row number
                    index = yield from stream_language(
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

            # One part (or no plan): use that language for the whole chunk
            forced = plan[0][0] if plan else None

        # ---- CASE 2: a fixed language, a file, or plain one-language ----
        else:
            forced = forced_language(language, detector, audio)

        # Single stretch in a single language
        yield from stream_language(
            audio,
            forced,
            accurate,
            cancel,
            skip_english,
            total_duration=duration
        )


# =====================================================
# 3. FALLBACK (TRANSLATION MODEL NOT INSTALLED)
# =====================================================
# WHY : if the NLLB files are missing the app must still work. Whisper then
# translates the audio straight to English. It is less accurate and the
# original text may be missing, but the meeting is not left without help.

def process_whisper_translate(
    audio, whisper=None, language=None, accurate=False
):
    """
    PURPOSE : Whisper turns audio directly into English (and, if wanted,
              also gives the original text).
    RETURNS : (original segments, translated segments, language,
              probability)
    """

    whisper = whisper or model
    forced = forced_language(language, whisper, audio)
    options = decode_options(accurate)

    if forced:
        options["language"] = forced

    # Pass 1: audio -> English
    translated_segments, info = whisper.transcribe(
        audio,
        task="translate",
        **options
    )

    translated = collect_segments(translated_segments)

    language_code = info.language
    probability = info.language_probability

    # Nothing was said
    if not translated:
        return [], [], language_code, probability

    # English speech: original and translation are the same
    if language_code == "en":
        return translated, translated, language_code, probability

    transcribe_language = language_code

    # Hindi that Whisper took for Urdu: treat it as Hindi
    if (
        language_code == "ur"
        and settings.TREAT_URDU_AS_HINDI
        and forced != "ur"
    ):
        transcribe_language = "hi"
        language_code = "hi"

    # Live fallback shows English only (a second pass would be too slow)
    if not settings.SHOW_ORIGINAL_TEXT and not accurate:
        return [], translated, language_code, probability

    extra = {}

    if transcribe_language == "hi":
        extra.update(hindi_options(whisper))

    # Pass 2 (files only): audio -> original text
    original_segments, _ = whisper.transcribe(
        audio,
        task="transcribe",
        language=transcribe_language,
        **decode_options(accurate),
        **extra
    )

    original = collect_segments(original_segments)

    return original, translated, language_code, probability


def process_audio(audio, whisper=None, language=None, accurate=False):
    """
    PURPOSE : fallback processing of one chunk or file.
    WHY     : takes the model lock first, like iter_results does, so the
              fallback never runs at the same time as another job.
    """
    with model_lock:
        return process_whisper_translate(
            audio, whisper, language, accurate
        )


def process_file(path, language, accuracy):
    """
    PURPOSE : fallback processing of an uploaded file.
    NOTE    : accuracy "best" uses the bigger model.
    """
    accurate = accuracy == "best"
    whisper = get_accurate_model() if accurate else model

    return process_audio(
        path,
        whisper=whisper,
        language=language,
        accurate=accurate
    )


def iter_fallback(path, language, accuracy):
    """
    PURPOSE : turn the fallback result of a file into the same events the
              normal path produces.
    WHY     : the web page and routes_http.py then do not need to know
              which mode ran.
    """
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

    # LOOP - belongs to: iter_fallback()
    # WHY  : the fallback result is two plain lists (original, English).
    #        They are walked together so each sentence's original and
    #        English share the same index (the same row on the page).
    # REPEATS: once per sentence of the longer list.
    for i in range(max(len(original), len(translated))):
        # Timing comes from the original text if there is one
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


# =====================================================
# 4. MESSAGES FOR THE WEB PAGE AND THE THREAD BRIDGE
# =====================================================

def wire(event, base_id=0, offset=0.0):
    """
    PURPOSE : turn an internal event into the JSON message the web page
              understands.
    WHY     : the page expects "original_transcript" / "translation"
              messages with an `id`. Row id = base_id + index, so the
              English fills the row that already shows its original.
    INPUT   : base_id  first row number of the current chunk
              offset   seconds to add (the chunk's position in the meeting)
    RETURNS : a dictionary, or None for events the page does not need
              (language and progress events are handled by the callers).
    """
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
    PURPOSE : run a generator (iter_results) in a worker THREAD and hand its
              events to async code one by one, as soon as they appear.
    WHY     : the models are slow and block while they work. Running them in
              a thread keeps the web server responsive (it can still read
              audio, answer pings and send messages), which is what keeps
              the live connection from dropping.
    HOW     : a queue connects the two sides. The thread puts events in,
              the async code takes them out.
    """
    loop = asyncio.get_running_loop()
    box = asyncio.Queue()   # events travel from the thread to the caller

    def work():
        """Runs in the worker thread: pushes every event into the queue."""
        try:
            # LOOP - belongs to: run_stream(), inner function work()
            # WHY  : drive the generator step by step; each yielded event
            #        is passed to the async side immediately.
            # REPEATS: once per event until the generator is finished.
            for event in make_iterator():
                loop.call_soon_threadsafe(box.put_nowait, event)
        except Exception as error:
            # Send the error to the caller instead of losing it in the
            # thread
            loop.call_soon_threadsafe(
                box.put_nowait,
                {"kind": "error", "message": str(error)}
            )
        finally:
            # None tells the reader that the work is finished
            loop.call_soon_threadsafe(box.put_nowait, None)

    # daemon=True: the thread never blocks the program from closing
    threading.Thread(target=work, daemon=True).start()

    # LOOP - belongs to: run_stream(), reader side
    # WHY  : give every event to the caller the moment it arrives.
    # REPEATS: until the None "finished" marker comes out of the queue.
    while True:
        event = await box.get()

        if event is None:
            break

        yield event
