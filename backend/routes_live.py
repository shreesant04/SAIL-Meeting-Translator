"""
routes_live.py
==============
PURPOSE
    LIVE audio over a WebSocket:
        /ws/browser-live?language=hi|bn|ko|auto
    The page captures the meeting audio, cuts it into 10-second chunks and
    sends them here; this file sends back the original text, the Roman
    letters and the English text, sentence by sentence.

THREE PARTS WORK TOGETHER (they all live inside one function below)
    1. RECEIVER (the loop at the bottom)
         Receives chunks from the browser and puts them in a waiting line
         (a queue). It NEVER waits for the models, so the connection stays
         alive and no audio is lost.
    2. WORKER (a background task)
         Takes chunks from the line one at a time, runs them through the
         models and sends the results to the page.
    3. SENDERS (small helper functions)
         Send status text, original text (+ Roman letters) and English text.

WHY A QUEUE
    The first version processed each chunk before reading the next one. When
    the models were slow the browser's connection starved and closed. The
    queue separates "receiving" from "processing" so slow models only cause
    a delay, never a disconnect or skipped audio.
"""

import asyncio   # queue and background task
import time      # measure how long each chunk takes

import numpy as np                                    # audio sample arrays
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import settings                                       # limits and thresholds
import translation                                    # translation.MT_READY
from languages import language_label                  # "hi" -> "Hindi"
from pipeline import iter_results, process_audio, run_stream, wire
from romanize import romanize                         # Roman letters (fallback)
from text_cleanup import UNCLEAR_TEXT                 # "(unclear audio)"

# WHAT : the group of addresses defined in this file (attached in main.py)
router = APIRouter()

# Short local name, used several times below
SAMPLE_RATE = settings.SAMPLE_RATE


@router.websocket("/ws/browser-live")
async def browser_live_translation(websocket: WebSocket):
    """
    PURPOSE : one live meeting session (one browser connection). Runs from
              the moment the page connects until it disconnects.
    """

    # Accept the connection that the page opened
    await websocket.accept()

    # Language chosen on the page ("auto" = detect). It comes in the
    # address: /ws/browser-live?language=hi
    chosen_language = websocket.query_params.get("language", "auto")

    # Anything unknown is treated as "auto" (protects against bad input)
    if chosen_language not in settings.DETECTION_CANDIDATES:
        chosen_language = "auto"

    print("Live connection opened. Language:", chosen_language)

    # ---- The waiting line ----
    # WHAT : audio chunks wait here while the models are busy.
    # WHY  : see "WHY A QUEUE" at the top. maxsize only protects memory.
    queue = asyncio.Queue(maxsize=settings.MAX_WAITING_CHUNKS)

    # WHAT : the next free row number on the page.
    # WHY  : rows are matched by id; every chunk continues the numbering of
    #        the one before so ids never clash.
    next_id = 0

    # -------------------------------------------------
    # PART 3 - SENDERS: messages from the backend to the page
    # -------------------------------------------------

    async def send_status(message):
        """PURPOSE : the status line shown under the buttons on the page."""
        await websocket.send_json({"type": "status", "message": message})

    async def send_original(segment, row_id, offset, language, name):
        """
        PURPOSE : send original text + Roman letters (FALLBACK mode only;
                  normal mode sends the same through wire()).
        WHY     : the fallback result arrives as plain lists, so it needs
                  its own sender. The Roman line is added here too, so the
                  page shows it in both modes.
        """
        await websocket.send_json({
            "type": "original_transcript",
            "id": row_id,
            "language": language,
            "language_name": name,
            "start": round(offset + segment["start"], 2),
            "end": round(offset + segment["end"], 2),
            "text": segment["text"],
            # No Roman line for English or garbled text
            "roman": (
                ""
                if language == "en"
                or segment["text"] == UNCLEAR_TEXT
                else romanize(segment["text"], language)
            )
        })

    async def send_english(text, segment, row_id, offset, name):
        """
        PURPOSE : send English text for the row with the same id (FALLBACK
                  mode only).
        """
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
        """
        PURPOSE : status text when nothing is being processed right now.
        WHY     : tells the user whether the app is simply listening or is
                  catching up on a backlog of chunks.
        """
        waiting = queue.qsize()

        await send_status(
            "Listening to system audio..."
            if waiting == 0
            else f"Catching up: {waiting} audio chunk(s) "
                 "waiting. Nothing is skipped."
        )

    # -------------------------------------------------
    # PART 2 - WORKER: processes the waiting chunks one by one
    # -------------------------------------------------

    async def process_two_step_chunk(audio, offset, seconds, started):
        """
        PURPOSE : NORMAL mode. Sends the original text first, then the
                  English, sentence by sentence, as soon as each is ready.
        INPUT   : audio    the chunk's samples
                  offset   where this chunk starts in the meeting (seconds)
                  seconds  chunk length (for the log line)
                  started  time the chunk began processing (for the log)
        """
        nonlocal next_id

        await send_status("Transcribing speech...")

        base_id = next_id          # rows of this chunk start here
        count = 0                  # how many rows this chunk produced
        name = ""                  # detected language (for the log)
        probability = 0.0          # how sure the guess was (for the log)

        # LOOP - belongs to: process_two_step_chunk(), PART 2 (worker)
        # WHY  : the models produce events one by one in a worker thread;
        #        each must be forwarded to the page the moment it is ready
        #        so the screen updates sentence by sentence.
        # REPEATS: once per event (language, original, translation,
        #          progress) of this chunk.
        async for event in run_stream(
            lambda: iter_results(
                audio,
                chosen_language,
                False,                  # live = fast settings
                None,                   # nothing cancels a live chunk
                settings.IGNORE_ENGLISH,
                True                    # allow mixed languages in a chunk
            )
        ):
            kind = event["kind"]

            # A problem inside the models: handled by worker()
            if kind == "error":
                raise RuntimeError(event["message"])

            # Remember the language for the log line; the page learns it
            # from the original/translation messages themselves
            if kind == "language":
                name = event["language_name"]
                probability = event["probability"]
                continue

            # Turn the event into a page message and send it
            message = wire(event, base_id, offset)

            if message:
                await websocket.send_json(message)
                count = max(count, event["index"] + 1)

        # The next chunk continues the row numbering
        next_id += count

        # Log line: how long this chunk took (key for diagnosing delay)
        print(
            "Processed", seconds, "s of audio in",
            round(time.perf_counter() - started, 1), "s",
            "| Detected:", name or "-",
            "| Confidence:", round(probability, 2),
            "| Segments:", count
        )

    async def process_fallback_chunk(audio, offset, seconds, started):
        """
        PURPOSE : FALLBACK mode (translation model not installed). Whisper
                  translates the whole chunk, then the rows are sent.
        RETURNS : False when there is nothing to show, True otherwise.
        """
        nonlocal next_id

        # The models are blocking, so they run in a thread
        original, translated, language, probability = (
            await asyncio.to_thread(process_audio, audio)
        )

        print(
            "Processed", seconds, "s of audio in",
            round(time.perf_counter() - started, 1), "s"
        )

        # Nothing was said
        if not original and not translated:
            return False

        # English speech is hidden by the setting
        if language == "en" and settings.IGNORE_ENGLISH:
            return False

        name = language_label(language)
        base_id = next_id
        next_id += max(len(original), len(translated))

        # LOOP 1 - belongs to: process_fallback_chunk()
        # WHY  : send the ORIGINAL text of every sentence first.
        # REPEATS: once per original sentence.
        for index, segment in enumerate(original):
            await send_original(
                segment, base_id + index, offset, language, name
            )

        # LOOP 2 - belongs to: process_fallback_chunk()
        # WHY  : then send the ENGLISH of every sentence; the shared row id
        #        puts each English next to its original.
        # REPEATS: once per English sentence.
        for index, segment in enumerate(translated):
            await send_english(
                segment["text"], segment, base_id + index, offset, name
            )

        return True

    async def worker():
        """
        PURPOSE : the background task of PART 2. Takes chunks from the line,
                  one at a time, and processes them.
        WHY     : an error in one chunk must never close the connection, so
                  every chunk is processed inside its own try/except.
        """
        try:
            # LOOP - belongs to: worker(), PART 2
            # WHY  : the worker must keep taking chunks for as long as the
            #        meeting lasts.
            # REPEATS: forever; it stops when worker_task.cancel() is called
            #          in the `finally` block at the bottom of this function.
            while True:
                # Wait for the next chunk (sleeps while the line is empty)
                audio, offset = await queue.get()
                seconds = round(len(audio) / SAMPLE_RATE, 1)

                try:
                    started = time.perf_counter()

                    if translation.MT_READY:
                        # NORMAL two-step mode
                        await process_two_step_chunk(
                            audio, offset, seconds, started
                        )
                    else:
                        # FALLBACK mode
                        shown = await process_fallback_chunk(
                            audio, offset, seconds, started
                        )

                        # Nothing to show: back to "Listening..."
                        if not shown:
                            await idle_status()
                            continue

                except Exception as error:
                    # Tell the page about the problem but KEEP RUNNING
                    print("Chunk processing error:", error)
                    await send_status("Processing error: " + str(error))
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {error}"
                    })
                    continue

                # Chunk done: update the status line
                await idle_status()

        except asyncio.CancelledError:
            # Normal end of the meeting
            raise
        except Exception as error:
            print("Worker stopped:", error)

    # Start the worker in the background
    worker_task = asyncio.create_task(worker())

    # WHAT : seconds of audio received so far.
    # WHY  : gives every chunk its time position in the meeting (the
    #        "00:24" label on each row).
    audio_offset = 0.0

    # -------------------------------------------------
    # PART 1 - RECEIVER: takes audio from the browser
    # -------------------------------------------------

    try:
        # Greeting that also tells which method is active
        await websocket.send_json({
            "type": "status",
            "message": (
                "Browser audio connection established. Method: "
                + ("two-step (text translation)." if translation.MT_READY
                   else "Whisper direct (fallback).")
            )
        })

        # LOOP - belongs to: browser_live_translation(), PART 1 (receiver)
        # WHY  : keep receiving audio for the whole meeting. It only waits
        #        for the browser, never for the models, so the connection
        #        stays alive.
        # REPEATS: once per message from the browser, until the browser
        #          disconnects (break) or an error ends it.
        while True:
            message = await websocket.receive()

            # The page was closed or Stop was pressed
            if message["type"] == "websocket.disconnect":
                print("Browser closed the connection.")
                break

            audio_bytes = message.get("bytes")

            # Float32 audio needs complete 4-byte samples; ignore damaged
            # or text messages
            if not audio_bytes or len(audio_bytes) % 4 != 0:
                continue

            # Raw bytes -> audio samples (copy: the buffer is read-only)
            audio = np.frombuffer(
                audio_bytes,
                dtype=np.float32
            ).copy()

            if len(audio) == 0:
                continue

            # This chunk's position in the meeting
            duration = len(audio) / SAMPLE_RATE
            offset = audio_offset
            audio_offset += duration

            # Loudness of the chunk (for diagnosis and silence skipping)
            level = float(np.sqrt(np.mean(audio ** 2)))

            print(
                "Audio received:",
                round(duration, 2), "seconds",
                "| Level:", round(level, 4)
            )

            # Skip silence to save processing time
            if level < settings.SILENCE_LEVEL:
                continue

            # Memory protection: only if the line is extremely long
            # (20 minutes of waiting audio) is the oldest chunk dropped.
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

            # Put the chunk in the waiting line for the worker
            queue.put_nowait((audio, offset))

            # More than one chunk waiting: tell the user we are catching up
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
        # Always stop the worker when the session ends
        worker_task.cancel()
        print("Live connection closed.")
