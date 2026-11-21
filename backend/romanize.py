"""
romanize.py
===========
PURPOSE
    Shows the original text in ROMAN (English) letters under the original,
    for example Hindi "नमस्ते कैसे हैं" -> "namaste kaise hain".

WHY IT EXISTS
    People who cannot read Devanagari, Bengali or Hangul scripts can still
    read how the sentence sounds. It is an APPROXIMATION made with rules,
    NOT a translation.

HOW IT WORKS
    Hindi / Bengali : "indic-transliteration" package turns the script into
                      ITRANS letters, then our own rules (HINGLISH_MAP and
                      the silent-"a" rules) turn that into everyday spelling.
    Korean          : "korean-romanizer" package.
    If a package is missing, the Roman line is simply left out and the app
    keeps working.

USED BY
    pipeline.py (stream_language, iter_fallback), routes_live.py (fallback)
"""

import re  # regular expressions for the silent-"a" rules below

# -----------------------------------------------------------------
# Optional packages
# -----------------------------------------------------------------
# WHY try/except: the packages are optional. If one is not installed we
# store None and romanize() quietly returns "" instead of crashing.
try:
    from indic_transliteration import sanscript   # Hindi / Bengali
except Exception:
    sanscript = None

try:
    from korean_romanizer.romanizer import Romanizer   # Korean
except Exception:
    Romanizer = None

# Start-up report, so a missing package is noticed immediately in the
# backend window.
print(
    "Roman letters: Hindi/Bengali",
    "ok" if sanscript is not None
    else "NOT installed (pip install indic-transliteration)",
    "| Korean",
    "ok" if Romanizer is not None
    else "NOT installed (pip install korean-romanizer)"
)


# -----------------------------------------------------------------
# TABLE: ITRANS letters -> everyday Roman letters
# -----------------------------------------------------------------
# WHAT : (old, new) pairs applied in order by hinglishify().
# WHY  : ITRANS is a scholarly spelling ("aa" capital A, "~N" ...). People
#        write "kh", "aa", "n" instead. ORDER MATTERS: longer patterns come
#        first so "kSh" is replaced before "K" or "S" alone.
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

# Letters that count as consonants in the rules below
_CONSONANT_CLASS = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"

# RULE A - final silent "a":
# Hindi does not pronounce the "a" at the end of a word (सब = "sab").
# Matches an "a" that follows a consonant and is not followed by a letter.
_FINAL_SCHWA = re.compile(
    "(?<=[" + _CONSONANT_CLASS + "])a(?![A-Za-z])"
)

# RULE B - middle silent "a":
# Also silent inside many words (करता = "karta"). Matches an "a" between
# "vowel+consonant" before it and "consonant+vowel" after it.
_MIDDLE_SCHWA = re.compile(
    "(?<=[aeiouAEIOU][bcdfghjklmnpqrstvwxyz])a"
    "(?=[bcdfghjklmnpqrstvwxyz][aeiouAEIOU])"
)


def hinglishify(text, drop_schwa):
    """
    PURPOSE : ITRANS letters -> simple everyday Roman letters.
    WHY     : makes the Roman line readable ("namaste kaise hain").
    INPUT   : text in ITRANS letters; drop_schwa=True for Hindi, which
              also removes the silent "a" sounds (rules A and B).
    """
    words = []

    # LOOP 1 - belongs to: hinglishify()
    # WHY  : the silent-"a" rules must be applied word by word (a word
    #        boundary decides whether an "a" is final), so the sentence is
    #        cut into words first.
    # REPEATS: once per word of the sentence.
    for word in text.split(" "):
        # Silent-"a" rules only for Hindi and only for words longer than
        # two characters (short words like "ka" keep their "a").
        if drop_schwa and len(word) > 2:
            word = _FINAL_SCHWA.sub("", word)
            word = _MIDDLE_SCHWA.sub("", word)

        # LOOP 2 - belongs to: hinglishify(), inside LOOP 1
        # WHY  : apply every replacement of HINGLISH_MAP to this word, in
        #        the table's order (longest patterns first).
        # REPEATS: once per (old, new) pair of the table.
        for old, new in HINGLISH_MAP:
            word = word.replace(old, new)

        words.append(word)

    # Join the converted words back into a sentence
    return " ".join(words)


def romanize(text, code):
    """
    PURPOSE : original text -> Roman letters for language `code`
              ("hi", "bn" or "ko").
    WHY     : fills the italic Roman line under every original sentence.
    RETURNS : the Roman text, or "" when it is not possible (package
              missing, unsupported language, or the text did not change).
              "" means the page simply shows no Roman line.
    """
    try:
        # ---- Korean ----
        if code == "ko" and Romanizer is not None:
            result = Romanizer(text).romanize()

        # ---- Hindi and Bengali ----
        elif code in ("hi", "bn") and sanscript is not None:
            # Which script is the text in?
            source = (
                sanscript.DEVANAGARI if code == "hi"
                else sanscript.BENGALI
            )

            # Step 1: script -> ITRANS letters (library does this)
            itrans = sanscript.transliterate(
                text, source, sanscript.ITRANS
            )

            # Step 2: ITRANS -> everyday spelling (our rules above).
            # Only Hindi drops the silent "a" sounds.
            result = hinglishify(itrans, drop_schwa=(code == "hi"))

        # ---- Any other language: no Roman line ----
        else:
            return ""

        result = result.strip()

        # If nothing changed the text was already Roman letters; showing
        # the same text twice would be pointless.
        return "" if result == text.strip() else result

    except Exception as error:
        # Never let a Roman-letter problem break the translation itself
        print("Roman letters failed:", error)
        return ""
