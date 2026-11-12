import io
import os
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app, WHISPER_AVAILABLE
from audio_utils import create_synthetic_wav
from tts import tts_manager

client = TestClient(app)

def test_translate_audio_empty_file():
    response = client.post(
        "/translate-audio",
        files={"file": ("test.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_translate_audio_invalid_extension():
    response = client.post(
        "/translate-audio",
        files={"file": ("malicious.exe", b"binarydata", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()

@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="faster-whisper is not installed")
def test_translate_audio_valid_wav():
    wav_content = create_synthetic_wav(duration_s=1.0, sample_rate=16000)
    response = client.post(
        "/translate-audio",
        files={"file": ("meeting_sample.wav", wav_content, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "language" in data
    assert "language_name" in data
    assert "language_probability" in data
    assert "original_transcript" in data
    assert "translation" in data
    assert isinstance(data["original_transcript"], list)
    assert isinstance(data["translation"], list)

def test_translate_audio_real_speech():
    """
    Test uploading a synthesized meeting speech audio file and checking transcript.
    """
    if not tts_manager.is_model_available():
        pytest.skip("TTS model not available for speech synthesis")

    wav_content = tts_manager.synthesize_wav("The committee meeting has commenced.")
    response = client.post(
        "/translate-audio",
        files={"file": ("committee_speech.wav", wav_content, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en"
    assert len(data["translation"]) > 0
    transcript_text = " ".join(s["text"] for s in data["translation"]).lower()
    assert "meeting" in transcript_text or "committee" in transcript_text or "commenced" in transcript_text

def test_translate_multilingual_hindi_speech():
    """
    Test translating native Hindi speech from sample if present.
    """
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
    hindi_sample = os.path.join(samples_dir, "sample_hindi.wav")

    if not os.path.exists(hindi_sample):
        pytest.skip("sample_hindi.wav not generated yet")

    with open(hindi_sample, "rb") as f:
        wav_content = f.read()

    response = client.post(
        "/translate-audio",
        files={"file": ("sample_hindi.wav", wav_content, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language"] in ("hi", "ur")
    assert len(data["translation"]) > 0
