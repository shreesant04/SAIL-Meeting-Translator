import os
import sys
import time
import asyncio
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

# Workaround for soundcard library bug on Linux where it inspects sys.argv[1]
if len(sys.argv) < 2:
    sys.argv.append("sail")

import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Request,
    Response,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None

from audio_utils import (
    resample_audio,
    calculate_rms_level,
    pcm_bytes_to_float32,
    ensure_mono_float32
)
from tts import tts_manager

# =====================================================
# SETTINGS & CONFIGURATION
# =====================================================

MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
CPU_THREADS = int(os.getenv("CPU_THREADS", max(2, (os.cpu_count() or 4) - 1)))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
SILENCE_LEVEL = float(os.getenv("SILENCE_LEVEL", "0.003"))
MAX_WAITING_CHUNKS = int(os.getenv("MAX_WAITING_CHUNKS", "120"))
SHOW_ORIGINAL_TEXT = os.getenv("SHOW_ORIGINAL_TEXT", "True").lower() in ("true", "1", "yes")
TREAT_URDU_AS_HINDI = os.getenv("TREAT_URDU_AS_HINDI", "True").lower() in ("true", "1", "yes")
IGNORE_ENGLISH = os.getenv("IGNORE_ENGLISH", "False").lower() in ("true", "1", "yes")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

HINDI_PROMPT = (
    "यह हिंदी भाषा में बोला गया है। "
    "कृपया हिंदी को देवनागरी लिपि में लिखें। "
    "उर्दू लिपि का उपयोग न करें।"
)

MEETING_PROMPT = os.getenv(
    "MEETING_PROMPT",
    "This is an official business meeting at Steel Authority of India Limited (SAIL) discussing steel production, plant operations, safety compliance, and financial targets."
)

VAD_SETTINGS = {"min_silence_duration_ms": 500}

LANGUAGE_NAMES: Dict[str, str] = {
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
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "tr": "Turkish",
    "vi": "Vietnamese"
}

def language_label(code: Optional[str]) -> str:
    if not code:
        return "Unknown"
    return LANGUAGE_NAMES.get(code.lower(), code.upper())

# Model singleton and lock
model: Optional[Any] = None
model_lock = threading.Lock()

def get_whisper_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                if not WHISPER_AVAILABLE:
                    raise RuntimeError("faster-whisper is not installed.")
                print(f"Loading Whisper model ('{MODEL_SIZE}', device={DEVICE}, compute_type={COMPUTE_TYPE})...")
                model = WhisperModel(
                    MODEL_SIZE,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    cpu_threads=CPU_THREADS
                )
                print("Whisper model loaded successfully.")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: attempt to warm up models or initialize background components
    print("SAIL Meeting Translator backend starting up...")
    try:
        threading.Thread(target=get_whisper_model, daemon=True).start()
    except Exception as e:
        print(f"Background model initialization warning: {e}")
    try:
        threading.Thread(target=tts_manager.load_voice, daemon=True).start()
    except Exception as e:
        print(f"Background TTS initialization warning: {e}")
    yield
    print("SAIL Meeting Translator backend shutting down...")

# =====================================================
# APP SETUP
# =====================================================

app = FastAPI(
    title="SAIL Meeting Translator",
    description="Multilingual meeting translation & talkback system for SAIL internal infrastructure",
    version="1.0.0",
    lifespan=lifespan
)

# Flexible CORS to allow localhost, Codespaces (*.app.github.dev), and configurable origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]
custom_origins = os.getenv("CORS_ORIGINS", "")
if custom_origins:
    ALLOWED_ORIGINS.extend([o.strip() for o in custom_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("CORS_ALLOW_ALL", "true").lower() in ("true", "1") else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static files / frontend serving paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
DIST_DIR = os.path.join(PROJECT_DIR, "frontend", "dist")
SAMPLES_DIR = os.path.join(BACKEND_DIR, "samples")

os.makedirs(SAMPLES_DIR, exist_ok=True)

# Mount samples directory
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

# Mount frontend assets if dist exists
if os.path.exists(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# =====================================================
# REQUEST SCHEMAS
# =====================================================

class SynthesizeRequest(BaseModel):
    text: str

# =====================================================
# BASIC ENDPOINTS
# =====================================================

def get_status_dict():
    return {
        "app": "SAIL Meeting Translator",
        "status": "running",
        "model_size": MODEL_SIZE,
        "sample_rate": SAMPLE_RATE,
        "show_original_text": SHOW_ORIGINAL_TEXT,
        "tts_available": tts_manager.is_model_available()
    }

@app.get("/")
def home(request: Request):
    """
    Serves the React frontend application to web browsers (Accept: text/html),
    or returns JSON system status to API clients.
    """
    accept = request.headers.get("accept", "")
    index_path = os.path.join(DIST_DIR, "index.html")

    if "text/html" in accept and os.path.exists(index_path):
        return FileResponse(index_path)

    return get_status_dict()

@app.get("/api")
@app.get("/api/info")
def api_info():
    return get_status_dict()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "whisper_ready": model is not None or WHISPER_AVAILABLE,
        "tts_ready": tts_manager.is_model_available()
    }

@app.get("/languages")
def get_supported_languages():
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in sorted(LANGUAGE_NAMES.items(), key=lambda x: x[1])
        ]
    }

@app.get("/sample-files")
def get_sample_files():
    """
    Returns available sample meeting audio files for quick testing in UI.
    """
    files = []
    if os.path.exists(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if f.endswith(".wav"):
                files.append({
                    "filename": f,
                    "url": f"/samples/{f}",
                    "name": "Sample Hindi Meeting" if "hindi" in f else "Sample English Meeting" if "english" in f else f
                })
    return {"samples": files}

# =====================================================
# AUDIO PROCESSING CORE
# =====================================================

def collect_segments(segments) -> List[Dict[str, Any]]:
    result = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            result.append({
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": text
            })
    return result

def process_audio(audio: Any) -> tuple:
    """
    Pass 1: translate speech to English and detect language.
    Pass 2: transcribe original language speech (if not English and SHOW_ORIGINAL_TEXT is True).
    Returns: (original_segments, translated_segments, detected_language, probability)
    """
    # Ensure float32 mono if numpy array
    if isinstance(audio, np.ndarray):
        audio = ensure_mono_float32(audio)

    active_model = get_whisper_model()

    with model_lock:
        # Pass 1: English translation + language detection
        translated_segments, info = active_model.transcribe(
            audio,
            task="translate",
            beam_size=2,
            vad_filter=True,
            vad_parameters=VAD_SETTINGS,
            initial_prompt=MEETING_PROMPT,
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        translated = collect_segments(translated_segments)
        language = info.language
        probability = float(info.language_probability)

        if not translated:
            return [], [], language, probability

        # If detected language is already English, original & translation are identical
        if language == "en":
            return translated, translated, language, probability

        transcribe_language = language
        extra = {}

        if language == "ur" and TREAT_URDU_AS_HINDI:
            transcribe_language = "hi"
            language = "hi"

        if not SHOW_ORIGINAL_TEXT:
            return [], translated, language, probability

        # Pass 2: original text transcription in native script
        if transcribe_language == "hi":
            extra["initial_prompt"] = HINDI_PROMPT

        try:
            original_segments, _ = active_model.transcribe(
                audio,
                task="transcribe",
                language=transcribe_language,
                beam_size=1,
                vad_filter=True,
                vad_parameters=VAD_SETTINGS,
                condition_on_previous_text=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                **extra
            )
            original = collect_segments(original_segments)
        except Exception as transcribe_err:
            print(f"Pass 2 native script transcription error: {transcribe_err}")
            original = []

    return original, translated, language, probability

# =====================================================
# UPLOADED FILE TRANSLATION
# =====================================================

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac", ".aac", ".opus"
}

@app.post("/translate-audio")
async def translate_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file selected."
        )

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix and suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )

    suffix = suffix or ".wav"
    temp_path = None

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(content)

        original, translated, language, probability = await asyncio.to_thread(
            process_audio, temp_path
        )

        return {
            "language": language,
            "language_name": language_label(language),
            "language_probability": round(probability, 3),
            "original_transcript": original,
            "translation": translated
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"File translation error: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio translation failed: {str(error)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

# =====================================================
# TEXT-TO-SPEECH (TALKBACK) ENDPOINT
# =====================================================

@app.post("/synthesize")
async def synthesize_speech(req: SynthesizeRequest):
    """
    Generate spoken English WAV audio for translated text using Piper TTS.
    """
    if not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty."
        )

    try:
        wav_bytes = await asyncio.to_thread(tts_manager.synthesize_wav, req.text)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="talkback.wav"'
            }
        )
    except Exception as e:
        print(f"TTS synthesis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Talkback synthesis failed: {str(e)}"
        )

# =====================================================
# LIVE BROWSER AUDIO WEBSOCKET
# =====================================================

@app.websocket("/ws/browser-live")
async def browser_live_translation(
    websocket: WebSocket,
    sample_rate: int = Query(default=16000),
    talkback: bool = Query(default=False)
):
    # Validate client sample rate
    if sample_rate <= 0 or sample_rate > 192000:
        sample_rate = 16000

    await websocket.accept()
    print(f"Live WebSocket connection accepted (client sample_rate: {sample_rate} Hz, talkback={talkback}).")

    queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_WAITING_CHUNKS)

    async def worker():
        try:
            while True:
                audio, offset = await queue.get()

                try:
                    started = time.perf_counter()
                    original, translated, language, probability = await asyncio.to_thread(
                        process_audio, audio
                    )
                    duration_s = round(len(audio) / SAMPLE_RATE, 2)
                    elapsed_s = round(time.perf_counter() - started, 2)
                    print(f"Processed {duration_s}s audio in {elapsed_s}s.")
                except Exception as error:
                    print(f"Chunk processing error: {error}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {error}"
                    })
                    continue

                if not original and not translated:
                    continue

                if language == "en" and IGNORE_ENGLISH:
                    continue

                name = language_label(language)

                for segment in original:
                    await websocket.send_json({
                        "type": "original_transcript",
                        "language": language,
                        "language_name": name,
                        "start": round(offset + segment["start"], 2),
                        "end": round(offset + segment["end"], 2),
                        "text": segment["text"]
                    })

                for segment in translated:
                    await websocket.send_json({
                        "type": "translation",
                        "language": "en",
                        "language_name": "English",
                        "source_language_name": name,
                        "start": round(offset + segment["start"], 2),
                        "end": round(offset + segment["end"], 2),
                        "text": segment["text"]
                    })

                waiting = queue.qsize()
                status_msg = (
                    "Listening to meeting audio..."
                    if waiting == 0
                    else f"Catching up: {waiting} audio chunk(s) queued."
                )
                await websocket.send_json({
                    "type": "status",
                    "message": status_msg
                })

        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"WebSocket worker stopped: {error}")

    worker_task = asyncio.create_task(worker())
    audio_offset = 0.0

    try:
        await websocket.send_json({
            "type": "status",
            "message": "Live audio translation connection established."
        })

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Handle text messages (e.g. heartbeat ping or config change)
            if "text" in message and message["text"]:
                if message["text"] == "ping":
                    await websocket.send_text("pong")
                continue

            audio_bytes = message.get("bytes")
            if not audio_bytes:
                continue

            audio = pcm_bytes_to_float32(audio_bytes)
            if len(audio) == 0:
                continue

            # Downsample if client sample rate differs from 16000
            if sample_rate != SAMPLE_RATE:
                audio = resample_audio(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

            if len(audio) == 0:
                continue

            duration = len(audio) / SAMPLE_RATE
            offset = audio_offset
            audio_offset += duration

            level = calculate_rms_level(audio)

            # Skip silent chunks
            if level < SILENCE_LEVEL:
                continue

            # Queue full handling: drop oldest chunk if line exceeds capacity
            if queue.full():
                try:
                    queue.get_nowait()
                    await websocket.send_json({
                        "type": "status",
                        "message": "Translation line full. Oldest chunk skipped."
                    })
                except asyncio.QueueEmpty:
                    pass

            queue.put_nowait((audio, offset))

            if queue.qsize() >= 2:
                await websocket.send_json({
                    "type": "status",
                    "message": f"Catching up: {queue.qsize()} chunks waiting."
                })

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as error:
        print(f"WebSocket connection error: {error}")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        print("Live connection cleaned up.")
