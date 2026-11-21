"""
routes_http.py
==============
PURPOSE
    The normal web (HTTP) addresses of the backend. The web page calls
    these with ordinary requests (the LIVE audio has its own file,
    routes_live.py, because it uses a WebSocket).

ADDRESSES
    GET  /                 "backend is running" check
    GET  /health           which models and method are in use
    GET  /speak-status     is the Piper talkback voice available?
    POST /speak            English text -> WAV sound (talkback)
    POST /translate-audio  translate an uploaded / recorded audio file

WHY IT IS SEPARATE
    main.py only assembles the app. Keeping the addresses in their own
    "router" files means each group of addresses can be read, tested and
    changed alone.
"""

import asyncio      # run slow work (Piper) without blocking the server
import json         # messages are sent as JSON text lines
import os           # temporary file handling
import tempfile     # temporary copy of the uploaded audio
import threading    # cancel flag shared with the worker thread
import time         # measure how long a file took

from fastapi import (
    APIRouter,      # a group of addresses, attached to the app in main.py
    UploadFile,     # an uploaded file
    File,           # marks a parameter as "a file upload"
    Form,           # marks a parameter as "a form field"
    HTTPException   # an error answer with a status code
)
from fastapi.responses import StreamingResponse, Response

import settings                  # model names for /health
import speech_models             # which specialist models are loaded
import translation               # translation.MT_READY (normal/fallback)
from pipeline import iter_results, iter_fallback, run_stream, wire
from talkback import load_piper, piper_wav

# WHAT : the group of addresses defined in this file.
# WHY  : main.py attaches it with app.include_router(router).
router = APIRouter()


# =====================================================
# STATUS ADDRESSES
# =====================================================

@router.get("/")
def home():
    """
    PURPOSE : simple check that the backend is running (open the address in
              a browser; any answer means the server is up).
    """
    return {"message": "SAIL ISP backend is running!"}


@router.get("/health")
def health():
    """
    PURPOSE : show which translation method and models are active.
    WHY     : the first thing to open when something seems wrong:
              "method" must be "two-step" and "specialist_models" should
              list bn, hi and ko.
    """
    return {
        "status": "healthy",
        "method": "two-step" if translation.MT_READY else "whisper-direct",
        "live_model": settings.MODEL_SIZE,
        "specialist_models": sorted(speech_models.specialist_models),
        "accurate_model": settings.ACCURATE_MODEL
    }


# =====================================================
# TALKBACK VOICE
# =====================================================

@router.get("/speak-status")
async def speak_status():
    """
    PURPOSE : tell the page whether the Piper voice is available.
    WHY     : when the user ticks "talkback", the page asks this first and
              offers Piper only if it works; otherwise the Windows voice.
    """
    # Loading can be slow the first time, so it runs outside the server's
    # main loop (run_in_executor) to keep everything else responsive.
    loaded = await asyncio.get_event_loop().run_in_executor(
        None, load_piper
    )

    return {
        "available": bool(loaded),
        "voice": os.path.basename(loaded[2]) if loaded else ""
    }


@router.post("/speak")
async def speak(text: str = Form(...)):
    """
    PURPOSE : English text -> WAV sound made by the Piper voice.
    WHY     : called once per translated sentence while talkback is on.
    ERRORS  : 400 = no text; 503 = Piper not available (the page then
              switches to the Windows voice).
    """
    # Limit the length: a runaway text must not block the voice engine
    text = (text or "").strip()[:600]

    if not text:
        raise HTTPException(status_code=400, detail="No text")

    try:
        # Sound generation takes CPU time: run it outside the main loop
        wav = await asyncio.get_event_loop().run_in_executor(
            None, piper_wav, text
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error))

    return Response(content=wav, media_type="audio/wav")


# =====================================================
# UPLOADED / RECORDED FILE
# =====================================================

@router.post("/translate-audio")
async def translate_audio(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    accuracy: str = Form("best")
):
    """
    PURPOSE : translate an audio file and STREAM the result as JSON lines.
    WHY     : each sentence appears as soon as it is ready (original text
              first, then its English meaning), instead of the page
              waiting minutes for the whole file.
    INPUT   : file      the audio (also used for "Translate this recording")
              language  "hi", "bn", "ko" or "auto"
              accuracy  "best" (bigger model, slower) or "fast"
    """

    # No file chosen on the page
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file selected."
        )

    # ---- Save the upload to a temporary file ----
    # WHY: the audio decoder reads from a file path. The copy is deleted
    # when the job ends (see make_iterator below), so nothing is kept.
    suffix = os.path.splitext(file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    # WHAT : a flag that tells the worker thread to stop.
    # WHY  : set when the page is closed or Stop is pressed, so the models
    #        do not keep working for nobody.
    cancel = threading.Event()

    def make_iterator():
        """
        PURPOSE : choose the normal two-step path or the fallback, and
                  always delete the temporary audio at the end.
        """
        try:
            if translation.MT_READY:
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
        """PURPOSE : one JSON message on its own line (the stream format)."""
        return json.dumps(message, ensure_ascii=False) + "\n"

    async def body():
        """
        PURPOSE : produces the streamed answer, line by line, while the
                  worker thread transcribes and translates.
        """
        started = time.perf_counter()

        try:
            # Tell the page right away that the file was received
            yield line({
                "type": "status",
                "message": "Reading the audio file..."
            })

            # LOOP - belongs to: body() inside translate_audio()
            # WHY  : every event from the worker thread must be turned into
            #        a JSON line and sent to the page immediately; that is
            #        what makes the screen fill sentence by sentence.
            # REPEATS: once per event (language, original, translation,
            #          progress) until the file is finished.
            async for event in run_stream(make_iterator):
                kind = event["kind"]

                # A problem inside the worker: report it and stop
                if kind == "error":
                    print("File translation error:", event["message"])
                    yield line({
                        "type": "error",
                        "message": event["message"]
                    })
                    return

                # Which language was heard (for the "Detected" label)
                if kind == "language":
                    yield line({
                        "type": "language",
                        "language": event["language"],
                        "language_name": event["language_name"]
                    })
                    continue

                # Percent finished
                if kind == "progress":
                    yield line({
                        "type": "progress",
                        "percent": event["percent"]
                    })
                    continue

                # Original text / English text rows
                message = wire(event)

                if message:
                    yield line(message)

            print(
                "File processed in",
                round(time.perf_counter() - started, 1), "s"
            )

            # Tells the page the file is completely finished
            yield line({"type": "done"})

        finally:
            # If the page was closed or cancelled, stop the work
            cancel.set()

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson"
    )
