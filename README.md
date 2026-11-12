# SAIL Meeting Translator

> **Production-grade multilingual meeting translation and auditory talkback system.**  
> Designed for internal enterprise infrastructure, multi-regional committees, and international delegations.

---

## 1. Project Motive & Overview

In multi-regional enterprises like the Steel Authority of India Limited (SAIL), high-stakes meetings frequently involve participants speaking different languages—including Hindi, Bengali, Korean, Japanese, and various Indian and international languages. 

**SAIL Meeting Translator** solves language barriers in real-time meetings by:
1. **Live Speech Capture & Streaming**: Capturing system audio (from Google Meet, Zoom, MS Teams, video streams) or direct microphone input.
2. **On-Premise / Edge Machine Learning**: Utilizing `faster-whisper` (CTranslate2-optimized Whisper models) for low-latency automatic speech recognition (ASR) and automatic translation directly to English.
3. **Dual Transcript Stream**: Displaying both the original speaker transcript in its native script and the real-time English translation side-by-side with segment-level timestamps.
4. **Neural Talkback (Text-To-Speech)**: Synthesizing natural English speech using Piper TTS so English listeners can hear translated audio in real time or on demand.
5. **Session Recording & Export**: Archiving audio sessions securely and exporting meeting transcripts in `.txt` or `.json` format.

---

## 2. Architecture & Components

```
                ┌────────────────────────────────────────┐
                │          Browser / Client UI           │
                │        (React 19 + Vite + CSS)         │
                └──────┬──────────────────────────┬──────┘
                       │                          │
           Float32 PCM │ (WebSocket)    Multipart │ (REST API)
            Audio Stream (16kHz)       File Upload│ & TTS
                       │                          │
                       ▼                          ▼
    ┌────────────────────────────────────────────────────────────┐
    │                     FastAPI Backend                        │
    │  - CORS & Connection Manager                               │
    │  - Audio Resampling & Signal Processing (audio_utils.py)   │
    ├─────────────────────────────┬──────────────────────────────┤
    │       Whisper Engine        │       Talkback Engine        │
    │      (faster-whisper)       │         (piper-tts)          │
    │  - Multilingual Detection   │  - Neural Voice Synthesis    │
    │  - Speech-to-English        │  - Low Latency Audio (WAV)   │
    │  - Native Script Transcribe │                              │
    └─────────────────────────────┴──────────────────────────────┘
```

### Components:
- **Backend (`/backend`)**:
  - `main.py`: FastAPI server exposing WebSocket stream `/ws/browser-live`, file upload `/translate-audio`, TTS `/synthesize`, and health checks `/health`.
  - `audio_utils.py`: Robust audio resampling, RMS silence detection, PCM conversions, synthetic wave generation.
  - `tts.py`: Thread-safe Piper neural text-to-speech engine with automated voice model retrieval and graceful fallbacks.
  - `download_voices.py`: Standalone CLI helper to download neural ONNX voice models.
  - `tests/`: Automated unit and integration test suite (`test_api.py`, `test_file_translation.py`, `test_websocket.py`, `test_audio_utils.py`, `test_tts.py`).
- **Frontend (`/frontend`)**:
  - React 19 single-page application built with Vite.
  - Web Audio API pipeline with real-time browser downsampling (converts native browser audio rates like 44.1kHz or 48kHz cleanly to 16kHz).
  - Audio source selection (System Audio / Screen Share vs. Microphone).
  - Interactive transcript viewer with search, timestamp sync, audio meter, and auto-talkback.

---

## 3. Supported Languages

Automatic language detection supports 20+ languages out of the box, with native script handling:
- **Indian Regional**: Hindi (`hi`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Marathi (`mr`), Gujarati (`gu`), Punjabi (`pa`), Urdu (`ur`).
- **International / Partner Delegations**: Korean (`ko`), Japanese (`ja`), Chinese (`zh`), French (`fr`), German (`de`), Spanish (`es`), Russian (`ru`), Portuguese (`pt`), and more.

---

## 4. Quick Start & Setup

### A. Developing in GitHub Codespaces (Recommended)
This repository includes a `.devcontainer/devcontainer.json` configuration. When opening in GitHub Codespaces:
1. The container automatically installs Python 3.12, `ffmpeg`, and required audio packages.
2. Ports `8000` (Backend) and `5173` (Frontend) are forwarded automatically.
3. Start the entire application with:
   ```bash
   ./start.sh
   ```

### B. Local Setup

#### 1. System Requirements
- Linux / macOS / Windows WSL2
- Python 3.10+ (Python 3.12 recommended)
- `ffmpeg` installed on your system
- Node.js 18+ and `npm`

#### 2. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download Piper voice model for Talkback (one-time)
python download_voices.py

# Run FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 5. Running Tests

Run the complete test suite across backend and frontend:
```bash
./test.sh
```

Or individually:
```bash
# Backend pytest suite
cd backend
pytest tests/ -v

# Frontend lint & production build
cd frontend
npm run lint
npm run build
```

---

## 6. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | `GET` | Service overview, model parameters, and status |
| `GET /health` | `GET` | Health check for container probes & frontend connection |
| `GET /languages` | `GET` | List of supported translation languages |
| `POST /translate-audio` | `POST` | Upload audio file (`.wav`, `.mp3`, `.webm`, etc.) for translation |
| `POST /synthesize` | `POST` | Synthesize English text into speech (`audio/wav`) |
| `WS /ws/browser-live` | `WebSocket` | Real-time PCM audio streaming & translation events |

---

## 7. License

Internal Enterprise Software · Steel Authority of India Limited (SAIL).
