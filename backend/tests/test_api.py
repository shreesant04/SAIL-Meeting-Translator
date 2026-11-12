import io
import wave
import pytest
from fastapi.testclient import TestClient
from main import app, language_label

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "SAIL Meeting Translator"
    assert data["status"] == "running"
    assert "model_size" in data
    assert "sample_rate" in data

def test_api_info_endpoint():
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "SAIL Meeting Translator"
    assert data["status"] == "running"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "whisper_ready" in data

def test_languages_endpoint():
    response = client.get("/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    codes = [l["code"] for l in data["languages"]]
    assert "hi" in codes
    assert "bn" in codes
    assert "ko" in codes
    assert "en" in codes

def test_sample_files_endpoint():
    response = client.get("/sample-files")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)

def test_language_label_helper():
    assert language_label("hi") == "Hindi"
    assert language_label("bn") == "Bengali"
    assert language_label("ko") == "Korean"
    assert language_label("en") == "English"
    assert language_label(None) == "Unknown"
    assert language_label("xyz") == "XYZ"

def test_synthesize_empty():
    response = client.post("/synthesize", json={"text": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
