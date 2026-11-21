"""
translation.py
==============
PURPOSE
    The OFFLINE text-translation model (NLLB-200 through CTranslate2).
    It turns the original text (Hindi / Bengali / Korean ...) into English.

WHY IT EXISTS
    The two-step method (audio -> original text -> English text) is more
    accurate than letting Whisper translate audio directly, and it lets
    the screen show the ORIGINAL text first, matched row by row with its
    English. This file owns that second step.

IMPORTANT FOR OTHER FILES
    The state of this module changes at start-up (MT_READY becomes True
    once the model loaded). Other files must read it as
        translation.MT_READY
    and NOT as `from translation import MT_READY`, which would keep an
    old copy that never updates.

USED BY
    pipeline.py (translate_to_english), routes_http.py and routes_live.py
    (translation.MT_READY chooses normal or fallback mode)
"""

import os  # to check that the model files exist

import settings                              # model folders, device, threads
from languages import NLLB_CODES  # noqa: F401  (kept for other modules)
from text_cleanup import collapse_repetition  # last-resort loop remover

# -----------------------------------------------------------------
# Module state (filled in by load_translation_model() at the bottom)
# -----------------------------------------------------------------
translator = None     # the CTranslate2 translator once loaded
tokenizers = {}       # one tokenizer per source language (created lazily)
MT_READY = False      # True only when two-step translation is available


def get_tokenizer(nllb_code):
    """
    PURPOSE : the tokenizer for one source language.
    WHY     : the model reads "pieces" of words, not plain text; the
              tokenizer cuts text into those pieces and needs to know the
              source language. One is created per language, on first use,
              and then kept.
    """
    if nllb_code not in tokenizers:
        # Imported here (not at the top) so the app starts even if
        # `transformers` is missing; the failure is then caught in
        # load_translation_model() and fallback mode is used.
        from transformers import AutoTokenizer

        tokenizers[nllb_code] = AutoTokenizer.from_pretrained(
            settings.MT_TOKENIZER_DIR,
            src_lang=nllb_code
        )

    return tokenizers[nllb_code]


def load_translation_model():
    """
    PURPOSE : load the translation model at start-up.
    WHY     : if anything is missing or broken, the app must NOT crash; it
              switches to FALLBACK mode (Whisper translates audio straight
              to English) and says so in the backend window.
    EFFECT  : sets `translator` and `MT_READY`.
    """
    global translator, MT_READY

    model_file = os.path.join(settings.MT_MODEL_DIR, "model.bin")

    # CHECK 1 - the model file must exist
    if not os.path.exists(model_file):
        print(
            "Translation model not found at:", settings.MT_MODEL_DIR,
            "\n -> Using FALLBACK mode (Whisper audio-to-English)."
            "\n -> Run: python download_models.py"
        )
        return

    # CHECK 2 - the tokenizer folder must exist
    if not os.path.isdir(settings.MT_TOKENIZER_DIR):
        print(
            "Tokenizer folder not found at:", settings.MT_TOKENIZER_DIR,
            "\n -> Using FALLBACK mode (Whisper audio-to-English)."
            "\n -> Run: python download_models.py"
        )
        return

    try:
        # Imported here so a missing package is caught by the except below
        import ctranslate2
        import transformers  # noqa: F401  (needed for the tokenizer)

        print("Loading text translation model...")

        translator = ctranslate2.Translator(
            settings.MT_MODEL_DIR,
            device="cpu" if settings.DEVICE == "cpu" else "cuda",
            compute_type=(
                "int8" if settings.DEVICE == "cpu" else "float16"
            ),
            inter_threads=1,                  # one translation at a time
            intra_threads=settings.CPU_THREADS
        )

        # Quick self-test: build one tokenizer now so a broken install is
        # discovered at start-up, not in the middle of a meeting.
        get_tokenizer("hin_Deva")

        MT_READY = True
        print("Text translation model loaded. Mode: TWO-STEP.")

    except Exception as error:
        # Anything went wrong: stay in fallback mode
        translator = None
        MT_READY = False
        print(
            "Could not load the translation model:", error,
            "\n -> Using FALLBACK mode (Whisper audio-to-English)."
        )


def translate_to_english(texts, nllb_code, beam_size=1):
    """
    PURPOSE : translate a list of texts (one per segment) into English.
    WHY     : returns a list of the SAME LENGTH and ORDER, so each English
              sentence always lands on the row of its original text.
    INPUT   : texts     list of original sentences
              nllb_code source language code (see languages.py)
              beam_size 1 = fast (live), 4 = more accurate (files)
    RETURNS : list of English sentences; empty input text gives "".
    """

    tokenizer = get_tokenizer(nllb_code)

    # Only non-empty texts are translated; remember where they were
    # ONE-LINE LOOP - belongs to: translate_to_english(), step "select"
    # WHY  : the translator must not be asked to translate blank text, and we
    #        must remember each text's slot to put the answer back.
    # REPEATS: once per input text.
    positions = [i for i, text in enumerate(texts) if text.strip()]
    english = [""] * len(texts)

    # Nothing to translate (all texts empty)
    if not positions:
        return english

    # Text -> list of model "pieces" (tokens)
    # ONE-LINE LOOP - belongs to: translate_to_english(), step "tokenize"
    # WHY  : the model reads tokens, not text, so each selected text is cut
    #        into tokens.
    # REPEATS: once per text that will be translated.
    sources = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(texts[i]))
        for i in positions
    ]

    # The English cannot reasonably be much longer than the input; this
    # cap stops the model from rambling or looping endlessly.
    # ONE-LINE LOOP - belongs to: translate_to_english(), step "length cap"
    # WHY  : the cap is based on the longest input, so every input's length is
    #        looked at.
    # REPEATS: once per tokenized text.
    longest = max(len(source) for source in sources)
    limit = min(200, int(longest * 2.5) + 10)

    # The actual translation.
    #   target_prefix        : force the output language to English
    #   repetition_penalty   : discourage repeating words
    #   no_repeat_ngram_size : never repeat a 4-word phrase
    results = translator.translate_batch(
        sources,
        target_prefix=[["eng_Latn"]] * len(sources),
        beam_size=beam_size,
        max_decoding_length=limit,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4
    )

    # LOOP - belongs to: translate_to_english(), final step "put each
    #        result back on its own row"
    # WHY  : results come back in the order of `positions`; each English
    #        text must be written to the slot of the original text it came
    #        from, otherwise rows would be mismatched.
    # REPEATS: once per translated text.
    for position, result in zip(positions, results):
        tokens = result.hypotheses[0][1:]  # drop the language tag

        # Model pieces -> readable English text
        text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokens),
            skip_special_tokens=True
        ).strip()

        # Last safety net against a stuck repeating loop
        english[position] = collapse_repetition(text)

    return english


# Load at start-up, so the first translation is not delayed and the
# startup text shows "Mode: TWO-STEP" or the fallback warning.
load_translation_model()
