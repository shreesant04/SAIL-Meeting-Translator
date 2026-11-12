import time
import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app
from tts import tts_manager
from audio_utils import resample_audio

client = TestClient(app)

def test_websocket_ping_pong():
    with client.websocket_connect("/ws/browser-live?sample_rate=16000") as websocket:
        init_msg = websocket.receive_json()
        assert init_msg["type"] == "status"
        assert "established" in init_msg["message"].lower()

        # Send heartbeat ping
        websocket.send_text("ping")
        reply = websocket.receive_text()
        assert reply == "pong"

def test_websocket_silence_streaming():
    with client.websocket_connect("/ws/browser-live?sample_rate=16000") as websocket:
        init_msg = websocket.receive_json()
        assert init_msg["type"] == "status"

        # Send 1 second of complete silence (zeros)
        silent_audio = np.zeros(16000, dtype=np.float32)
        websocket.send_bytes(silent_audio.tobytes())

        # Give it a fraction of a second to ensure it does not emit an error
        time.sleep(0.2)
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"

def test_websocket_invalid_sample_rate():
    # Negative or extreme sample rates should not crash the server
    with client.websocket_connect("/ws/browser-live?sample_rate=-1000") as websocket:
        init_msg = websocket.receive_json()
        assert init_msg["type"] == "status"
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"

def test_websocket_real_speech_streaming():
    """
    Synthesize realistic meeting speech, convert to 16kHz float32 PCM,
    stream to WebSocket, and verify real translation message is delivered.
    """
    if not tts_manager.is_model_available():
        pytest.skip("Piper TTS voice model not available")

    import soundfile as sf
    wav_bytes = tts_manager.synthesize_wav("Welcome to the meeting.")
    audio_data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if sr != 16000:
        audio_data = resample_audio(audio_data, sr, 16000)

    with client.websocket_connect("/ws/browser-live?sample_rate=16000") as websocket:
        init_msg = websocket.receive_json()
        assert init_msg["type"] == "status"

        websocket.send_bytes(audio_data.tobytes())

        received_translation = False
        start_time = time.time()
        while time.time() - start_time < 12:
            try:
                msg = websocket.receive_json()
                if msg.get("type") == "translation":
                    assert len(msg.get("text", "")) > 0
                    received_translation = True
                    break
            except Exception:
                break

        assert received_translation, "Expected translation segment from live speech WebSocket stream"
