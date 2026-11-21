"""
text_cleanup.py
===============
PURPOSE
    Turns the raw pieces ("segments") that Whisper produces into clean
    rows for the screen. It removes noise and replaces garbled output with
    "(unclear audio)", so the screen never shows nonsense.

WHY IT EXISTS
    A speech model does not always know when it is wrong. On music,
    background noise or very fast speech it may print random syllables or
    repeat one sound endlessly ("vuvuvuvu"). The quality numbers Whisper
    attaches to each segment let us detect that and hide it.

USED BY
    pipeline.py (collect_segments + clean_segments for every segment),
    translation.py (collapse_repetition on the English output)
"""

import re  # regular expressions, used to detect repeating text

# WHAT : the text shown in place of garbled speech.
# WHY  : honest "I could not understand this" is better than wrong text.
#        pipeline.py also compares against it to skip Roman letters and
#        translation for such rows.
UNCLEAR_TEXT = "(unclear audio)"

# -----------------------------------------------------------------
# Patterns that find "stuck" text
# -----------------------------------------------------------------
# A model that gets stuck repeats a short piece many times in a row.
#   REPEAT_PATTERN   : a 1-12 character piece repeated 6+ times in a row
#                      -> the whole segment is garbled
#   COLLAPSE_PATTERN : repeated 4+ times -> squeezed to one copy
# (.{1,12}?) = a short piece; \1{5,} = the same piece again 5+ times.
REPEAT_PATTERN = re.compile(r"(.{1,12}?)\1{5,}", re.DOTALL)
COLLAPSE_PATTERN = re.compile(r"(.{1,12}?)\1{3,}", re.DOTALL)


def has_repetition(text):
    """
    PURPOSE : True if the text is stuck in a repeating loop.
    WHY     : one of the three signs of garbled speech used by
              clean_segments().
    """
    return bool(REPEAT_PATTERN.search(text))


def collapse_repetition(text):
    """
    PURPOSE : squeeze a repeating loop down to a single copy.
    WHY     : last safety net on the ENGLISH output of the translator, so
              an endless "the the the the" never reaches the screen.
    """
    return COLLAPSE_PATTERN.sub(r"\1", text)


def collect_segments(segments):
    """
    PURPOSE : convert Whisper's segment objects into plain dictionaries
              that carry the numbers needed to judge quality.
    WHY     : the rest of the app should not depend on Whisper's own
              classes, and the quality numbers (log-probability,
              no-speech chance, compression ratio) are needed by
              clean_segments().
    INPUT   : an iterable of Whisper segments (one in live use, many for
              a whole file).
    RETURNS : list of {start, end, text, logprob, no_speech, compression};
              segments without text are dropped.
    """
    result = []

    # LOOP - belongs to: collect_segments()
    # WHY  : the input may contain several segments (a file) and each one
    #        must be converted and checked separately.
    # REPEATS: once per Whisper segment.
    for segment in segments:
        # Remove spaces at the start/end of the recognised text
        text = segment.text.strip()

        # Keep only segments that actually contain text
        if text:
            result.append({
                "start": round(segment.start, 2),    # seconds
                "end": round(segment.end, 2),        # seconds
                "text": text,
                # How sure the model was (closer to 0 = more sure)
                "logprob": round(getattr(segment, "avg_logprob", 0.0), 2),
                # Chance this segment is not speech at all
                "no_speech": round(
                    getattr(segment, "no_speech_prob", 0.0), 2
                ),
                # Very high = repetitive, a sign of a stuck model
                "compression": round(
                    getattr(segment, "compression_ratio", 1.0), 2
                )
            })

    return result


def clean_segments(segments):
    """
    PURPOSE : quality filter applied to every segment before it is shown.
    RULES   :
        - Not speech (music, noise)             -> removed completely
        - Garbled or stuck in a repeating loop  -> kept but its text is
                                                   replaced by
                                                   "(unclear audio)"
        - Everything else                       -> unchanged
    WHY     : hides nonsense while still showing that something was said.
    """
    cleaned = []

    # LOOP - belongs to: clean_segments()
    # WHY  : each segment is judged on its own numbers; one bad segment
    #        must not remove the good ones next to it.
    # REPEATS: once per segment.
    for item in segments:
        # Quality numbers prepared by collect_segments()
        logprob = item.get("logprob", 0.0)
        no_speech = item.get("no_speech", 0.0)
        compression = item.get("compression", 1.0)

        # RULE 1 - Whisper's own test for "this is not speech":
        # high no-speech chance AND low confidence -> drop the segment.
        if no_speech > 0.6 and logprob < -1.0:
            continue

        # RULE 2 - signs of garbled text:
        #   compression > 2.4 : very repetitive text
        #   logprob < -1.2    : the model itself was very unsure
        #   has_repetition    : stuck in a loop ("vuvuvu...")
        garbled = (
            compression > 2.4
            or logprob < -1.2
            or has_repetition(item["text"])
        )

        if garbled:
            # Copy first so the caller's original dictionary is untouched
            item = dict(item)
            item["text"] = UNCLEAR_TEXT
            item["unclear"] = True   # lets pipeline.py skip translation

        cleaned.append(item)

    return cleaned
