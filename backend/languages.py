"""
languages.py
============
PURPOSE
    Small helpers about languages: readable names, the codes the text
    translation model uses, and checks of which alphabet a text is
    written in.

WHY IT EXISTS
    Three different parts of the app (speech models, translation, live
    screen) all need to turn a language code like "hi" into a name or a
    translation-model code. One shared file avoids three copies drifting
    apart.

USED BY
    speech_models.py (is_arabic_char), pipeline.py (language_label,
    nllb_code_for), routes_live.py (language_label)
"""

# -----------------------------------------------------------------
# TABLE 1: Whisper language code -> NLLB language code
# -----------------------------------------------------------------
# WHAT : Whisper names languages "hi", "ko" ...; the translation model
#        (NLLB) uses "hin_Deva", "kor_Hang" ... (language + alphabet).
# WHY  : the translator needs its own code to know the source language.
#        Languages not in this table cannot be translated by NLLB.
# USED : nllb_code_for() below.
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

# -----------------------------------------------------------------
# TABLE 2: Whisper language code -> name shown on screen
# -----------------------------------------------------------------
# WHAT : "hi" -> "Hindi" etc.
# WHY  : the page shows "Detected language: Hindi", not "hi".
# USED : language_label() below.
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
    """
    PURPOSE : turn a language code into the name shown on screen.
    WHY     : used for the "Detected language" label and every row.
    RETURNS : "Hindi" for "hi"; for an unknown code, the code in capitals
              (so the screen still shows something sensible).
    """
    return LANGUAGE_NAMES.get(code, code.upper())


def is_arabic_char(ch):
    """
    PURPOSE : tell whether ONE character belongs to the Arabic / Urdu
              alphabet.
    WHY     : Whisper sometimes writes Hindi speech in Urdu letters.
              speech_models.py uses this to find every Whisper token that
              writes Urdu letters, so those tokens can be forbidden and
              Hindi comes out in Devanagari.
    HOW     : the Arabic alphabet is spread over several Unicode ranges
              (basic, supplement, two presentation-form blocks); the
              character is checked against all four.
    """
    # Unicode number of the character (for example 0x0627 for an Arabic
    # letter). The ranges below are fixed blocks defined by Unicode.
    code = ord(ch)

    return (
        0x0600 <= code <= 0x06FF      # basic Arabic block
        or 0x0750 <= code <= 0x077F   # Arabic supplement
        or 0xFB50 <= code <= 0xFDFF   # presentation forms A
        or 0xFE70 <= code <= 0xFEFF   # presentation forms B
    )


def has_arabic_script(text):
    """
    PURPOSE : tell whether a whole text contains any Arabic / Urdu letters.
    WHY     : decides whether Hindi/Urdu text must be translated as Urdu
              ("urd_Arab") or Hindi ("hin_Deva") by nllb_code_for().
    HOW     : the generator below is a loop that looks at the characters
              one by one and stops at the first Arabic letter (any()).
    """
    # LOOP - belongs to: has_arabic_script()
    # WHY  : we must look at each character of the text, because one Urdu
    #        letter anywhere means the text is (partly) Urdu script.
    # REPEATS: once per character; stops early at the first match.
    return any("\u0600" <= ch <= "\u06ff" for ch in text)


def nllb_code_for(code, text):
    """
    PURPOSE : pick the translation model's code for the language that was
              spoken.
    WHY     : the translator must be told the source language and alphabet.
              Hindi that was written in Urdu letters has to be translated
              as Urdu or the result is nonsense.
    RETURNS : an NLLB code such as "hin_Deva", or None when the translation
              model does not know the language (then nothing is translated).
    """
    # Hindi and Urdu speech: choose the alphabet that was actually written
    if code in ("hi", "ur"):
        return "urd_Arab" if has_arabic_script(text) else "hin_Deva"

    # Every other language: look it up in TABLE 1
    return NLLB_CODES.get(code)
