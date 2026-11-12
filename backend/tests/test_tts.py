import wave
import io
import pytest
from fastapi.testclient import TestClient
from main import app
from tts import tts_manager

client = TestClient(app)

def test_tts_status():
    status = tts_manager.get_status()
    assert "available" in status
    assert "model_present" in status
    assert "voice_path" in status

def test_synthesize_endpoint_success():
    # If model is present in voices/, verify endpoint outputs valid wav
    if tts_manager.is_model_available():
        response = client.post("/synthesize", json={"text": "SAIL Meeting Translator live check."})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        content = response.content
        assert len(content) > 44
        with wave.open(io.BytesIO(content), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() > 0
