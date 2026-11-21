"""
language_detect.py
==================
PURPOSE
    Decides WHICH language is being spoken. Used by "Auto-detect".

FUNCTIONS (in the order they are used)
    detect_ranked  - guess the language of a stretch of audio (best first)
    show_ranked    - print that guess in the backend window (diagnosis)
    pick_language  - the single best language of a stretch of audio
    forced_language- the language the user picked, or a guess for "auto"
    speech_pieces  - cut audio at pauses in speech
    plan_mixed     - for LIVE audio that may switch language (for example
                     Hindi, then Korean) decide the language of every part
                     of a 10-second chunk

WHY IT EXISTS
    If the user picks a language by hand none of this is needed. Auto-detect
    is for when the language is unknown or changes during the meeting. It is
    kept apart from pipeline.py so the "which language?" decision can be
    tuned or replaced without touching the transcription logic.

USED BY
    pipeline.py (iter_results, process_whisper_translate)
"""

import settings  # candidate languages, thresholds for splitting

# Local short name: used many times below
SAMPLE_RATE = settings.SAMPLE_RATE


def detect_ranked(whisper, audio):
    """
    PURPOSE : guess the language among the SUPPORTED languages only.
    WHY     : Whisper knows ~100 languages; limiting the choice to Hindi,
              Bengali, Korean and English removes most wrong guesses.
    INPUT   : whisper  the model that makes the guess
              audio    audio samples, or a file path
    RETURNS : [(code, share), ...] best first, shares adding up to 1;
              or None if no guess is possible.
    """
    try:
        # A file path is first decoded into audio samples
        if isinstance(audio, str):
            from faster_whisper.audio import decode_audio

            audio = decode_audio(audio, sampling_rate=SAMPLE_RATE)

        # First try: look only at the parts that contain speech (silence
        # and music confuse the guess). If this version of the library does
        # not support that option, look at everything.
        try:
            probabilities = whisper.detect_language(
                audio, vad_filter=True
            )[2]
        except Exception:
            probabilities = whisper.detect_language(audio)[2]

        # Keep only the languages this app supports
        # ONE-LINE LOOP - belongs to: detect_ranked(), step "filter"
        # WHY  : Whisper returns a probability for ~100 languages; the loop goes
        #        through all of them and keeps only those in settings.DETECTION_CANDIDATES.
        # REPEATS: once per language Whisper reported.
        ranked = [
            (code, probability)
            for code, probability in probabilities
            if code in settings.DETECTION_CANDIDATES
        ]

        # Total probability of the supported languages
        # ONE-LINE LOOP - belongs to: detect_ranked(), step "re-scale"
        # WHY  : adds up the kept probabilities so they can be turned into shares
        #        of 100% below.
        # REPEATS: once per kept language (at most 4).
        total = sum(probability for _, probability in ranked)

        # None of our languages appeared: no usable guess
        if not ranked or total <= 0:
            return None

        # Re-scale so the supported languages add up to 100%, then sort
        # best first. Example: hi 0.30, ko 0.10 -> hi 0.75, ko 0.25.
        # ONE-LINE LOOP - belongs to: detect_ranked(), step "re-scale"
        # WHY  : divides every kept probability by the total.
        # REPEATS: once per kept language.
        return sorted(
            [(code, probability / total) for code, probability in ranked],
            key=lambda item: -item[1]
        )

    except Exception as error:
        # A failed guess must never stop the translation: the caller then
        # lets Whisper decide the language itself.
        print("Language choice failed, using normal auto-detect:", error)
        return None


def show_ranked(label, ranked):
    """
    PURPOSE : print a language guess in the backend window.
    WHY     : lets you SEE how sure the model was (for example
              "hi 0.62 | ko 0.38"), which is the key to fixing wrong
              Auto-detect results.
    """
    # ONE-LINE LOOP - belongs to: show_ranked()
    # WHY  : writes each language and its share as "hi 0.62" and joins them
    #        with " | " so the whole guess fits on one line.
    # REPEATS: once per language in the guess.
    print(
        label,
        " | ".join(f"{code} {share:.2f}" for code, share in ranked)
    )


def pick_language(whisper, audio):
    """
    PURPOSE : Auto-detect limited to the supported languages; returns the
              single best language code.
    RETURNS : a code such as "hi", or None if detection is not possible
              (then Whisper's own auto-detect is used).
    """
    ranked = detect_ranked(whisper, audio)

    if not ranked:
        return None

    show_ranked("Language check:", ranked)

    # The first entry is the most likely language
    return ranked[0][0]


def forced_language(language, whisper=None, audio=None):
    """
    PURPOSE : the language to transcribe in.
    CASES   : - a language the user picked        -> that language
              - "auto"                            -> the most likely
                supported language (None if it could not be worked out)
    WHY     : one place that turns the menu choice into a decision, used by
              both live audio and files.
    """
    if language in (None, "", "auto"):
        # Auto: we need the model and the audio to make a guess
        if whisper is not None and audio is not None:
            return pick_language(whisper, audio)

        return None

    # A fixed language chosen on the page
    return language


def speech_pieces(audio):
    """
    PURPOSE : cut audio at pauses in speech.
    WHY     : when two languages alternate, the pause between them is the
              natural place to cut, so each piece holds ONE language.
    RETURNS : [(start_sample, end_sample), ...], or [] if the voice
              detector is not available in this installation.
    """
    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions

        # Voice detector: returns where speech starts and ends.
        #   min_silence_duration_ms : a pause this long starts a new piece
        #   speech_pad_ms           : keep a little audio around each piece
        #                             so word edges are not cut off
        stamps = get_speech_timestamps(
            audio,
            VadOptions(
                min_silence_duration_ms=settings.PIECE_PAUSE_MS,
                speech_pad_ms=150
            )
        )
    except Exception as error:
        # Splitting is a bonus feature: if it fails the chunk is simply
        # handled as one language.
        print("Splitting at pauses is not available:", error)
        return []

    # Ignore pieces too short to contain real speech (a cough, a click)
    minimum = int(settings.MIN_PIECE_SECONDS * SAMPLE_RATE)

    # ONE-LINE LOOP - belongs to: speech_pieces(), step "drop tiny pieces"
    # WHY  : the voice detector may report many short blips; this goes through
    #        every reported piece, keeps the long enough ones and converts the
    #        positions to whole numbers.
    # REPEATS: once per piece reported by the voice detector.
    pieces = [
        (int(stamp["start"]), int(stamp["end"]))
        for stamp in stamps
        if int(stamp["end"]) - int(stamp["start"]) >= minimum
    ]

    # LOOP - belongs to: speech_pieces(), step "limit the number of pieces"
    # WHY  : every piece costs a language guess + a transcription, so a
    #        chunk with many tiny pauses would make the delay grow. While
    #        there are too many pieces, the two pieces with the SHORTEST
    #        gap between them are joined into one.
    # REPEATS: until at most settings.MAX_PIECES pieces remain.
    while len(pieces) > settings.MAX_PIECES:
        # Size of the gap between each piece and the next
        # ONE-LINE LOOP - belongs to: the `while` loop above
        # WHY  : the pair of pieces to join is the one with the smallest gap, so
        #        every gap must be measured first.
        # REPEATS: once per neighbouring pair of pieces.
        gaps = [
            pieces[i + 1][0] - pieces[i][1]
            for i in range(len(pieces) - 1)
        ]

        # Join the pair with the smallest gap
        i = gaps.index(min(gaps))
        pieces[i:i + 2] = [(pieces[i][0], pieces[i + 1][1])]

    return pieces


def plan_mixed(audio, detector):
    """
    PURPOSE : decide which language each part of a LIVE chunk is in.
    RETURNS : [(language, start_sample, end_sample), ...], or None when no
              guess is possible at all.
    WHY     : fixes "everything is read as Hindi" when Hindi and Korean
              alternate inside one 10-second chunk.

    THE IDEA (five steps):
      1. Guess the language of the WHOLE chunk.
      2. If the guess is confident -> one part, done (cheap, the usual case).
      3. Otherwise cut the chunk at pauses and guess every piece.
      4. Pieces too short to guess take the language of a neighbour.
      5. Neighbouring pieces with the same language are joined.
    """
    # ---- Step 1: guess the language of the whole chunk ----
    chunk = detect_ranked(detector, audio)

    if chunk is None:
        return None

    show_ranked("Language check (whole chunk):", chunk)

    top, share = chunk[0]
    whole = [(top, 0, len(audio))]   # the "one language" answer

    # ---- Step 2: confident -> no splitting needed ----
    if share >= settings.SPLIT_CONFIDENCE:
        return whole

    # ---- Step 3a: unclear -> cut at pauses ----
    pieces = speech_pieces(audio)

    # Fewer than two pieces: nothing to split, use the whole-chunk answer
    if len(pieces) < 2:
        return whole

    # ---- Step 3b: guess the language of each piece ----
    languages = []

    # LOOP 1 - belongs to: plan_mixed(), step 3b
    # WHY  : every piece may be in a different language, so each one needs
    #        its own guess.
    # REPEATS: once per piece (at most settings.MAX_PIECES times).
    for start, end in pieces:
        seconds = (end - start) / SAMPLE_RATE

        # Too short to guess reliably: leave it empty, decided in step 4
        if seconds < settings.SHORT_PIECE_SECONDS:
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

    # No piece could be guessed at all: fall back to the whole-chunk answer
    if not any(languages):
        return whole

    # ---- Step 4: short pieces take the language of a neighbour ----
    # LOOP 2 - belongs to: plan_mixed(), step 4
    # WHY  : the pieces left empty in loop 1 still need a language. The
    #        closest guessed piece (looking left first, then right) is the
    #        best available evidence.
    # REPEATS: once per piece; only empty ones are changed.
    for i, language in enumerate(languages):
        if language is None:
            # Nearest guessed piece on the LEFT (searching backwards)
            # ONE-LINE LOOP - belongs to: LOOP 2 above, per empty piece
            # WHY  : walks from the piece just before this one back to the first piece
            #        and takes the first one that has a language.
            # REPEATS: until a guessed piece is found or the start is reached.
            before = next(
                (languages[j] for j in range(i - 1, -1, -1)
                 if languages[j]),
                None
            )
            # Nearest guessed piece on the RIGHT (searching forwards)
            # ONE-LINE LOOP - belongs to: LOOP 2 above, per empty piece
            # WHY  : same idea, looking forward; used only when nothing was found on
            #        the left.
            # REPEATS: until a guessed piece is found or the end is reached.
            after = next(
                (languages[j] for j in range(i + 1, len(languages))
                 if languages[j]),
                None
            )
            languages[i] = before or after

    # ---- Step 5: join neighbours that share a language ----
    plan = []

    # LOOP 3 - belongs to: plan_mixed(), step 5
    # WHY  : two neighbouring pieces in the same language should be
    #        transcribed together; it is faster and gives the model more
    #        context for better accuracy.
    # REPEATS: once per piece, in time order.
    for (start, end), language in zip(pieces, languages):
        if plan and plan[-1][0] == language:
            # Same language as the previous part: stretch that part
            plan[-1] = (language, plan[-1][1], end)
        else:
            # A new language starts here
            plan.append((language, start, end))

    return plan
