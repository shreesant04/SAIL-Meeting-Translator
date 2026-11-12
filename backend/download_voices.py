#!/usr/bin/env python3
"""
Utility script to download Piper TTS voices and prepare bundled sample meeting audio.
Supports English (en_US-lessac-medium) and Hindi (hi_IN-pratham-medium).
"""

import os
import sys
import wave
import urllib.request

VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

VOICES = {
    "en_US-lessac-medium": {
        "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium",
        "files": ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"]
    },
    "hi_IN-pratham-medium": {
        "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/pratham/medium",
        "files": ["hi_IN-pratham-medium.onnx", "hi_IN-pratham-medium.onnx.json"]
    }
}

def download_file(url: str, dest_path: str):
    print(f"Downloading {os.path.basename(dest_path)}...")
    
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            sys.stdout.write(f"\r  Progress: {percent}% ({block_num * block_size // 1024} KB / {total_size // 1024} KB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook)
    sys.stdout.write("\n")
    sys.stdout.flush()

def ensure_piper_voice(target_dir: str = VOICE_DIR, voice_name: str = "en_US-lessac-medium") -> bool:
    os.makedirs(target_dir, exist_ok=True)
    voice_info = VOICES.get(voice_name)
    if not voice_info:
        print(f"Unknown voice: {voice_name}", file=sys.stderr)
        return False

    all_success = True
    for filename in voice_info["files"]:
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"✓ Found existing: {filename} ({os.path.getsize(dest_path)} bytes)")
            continue

        url = f"{voice_info['url_base']}/{filename}"
        try:
            download_file(url, dest_path)
            print(f"✓ Successfully downloaded: {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}", file=sys.stderr)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            all_success = False

    return all_success

def generate_sample_meeting_audio(voices_dir: str = VOICE_DIR, samples_dir: str = SAMPLES_DIR):
    """
    Generate realistic meeting audio samples for Hindi and English.
    """
    os.makedirs(samples_dir, exist_ok=True)
    try:
        from piper import PiperVoice
    except ImportError:
        print("piper-tts is not available for sample audio generation.")
        return

    # 1. English sample
    en_voice_path = os.path.join(voices_dir, "en_US-lessac-medium.onnx")
    en_sample_path = os.path.join(samples_dir, "sample_english.wav")
    if os.path.exists(en_voice_path) and not os.path.exists(en_sample_path):
        try:
            print("Generating sample_english.wav...")
            voice = PiperVoice.load(en_voice_path)
            with wave.open(en_sample_path, "wb") as wf:
                voice.synthesize_wav("Welcome to the Steel Authority of India meeting. We will now review the quarterly production results.", wf)
            print(f"✓ Created {en_sample_path} ({os.path.getsize(en_sample_path)} bytes)")
        except Exception as e:
            print(f"Failed to generate English sample: {e}")

    # 2. Hindi sample
    hi_voice_path = os.path.join(voices_dir, "hi_IN-pratham-medium.onnx")
    hi_sample_path = os.path.join(samples_dir, "sample_hindi.wav")
    if os.path.exists(hi_voice_path) and not os.path.exists(hi_sample_path):
        try:
            print("Generating sample_hindi.wav...")
            voice = PiperVoice.load(hi_voice_path)
            with wave.open(hi_sample_path, "wb") as wf:
                voice.synthesize_wav("सेल बैठक में आप सभी का स्वागत है। आज हम इस्पात उत्पादन की प्रगति पर चर्चा करेंगे।", wf)
            print(f"✓ Created {hi_sample_path} ({os.path.getsize(hi_sample_path)} bytes)")
        except Exception as e:
            print(f"Failed to generate Hindi sample: {e}")

if __name__ == "__main__":
    success_en = ensure_piper_voice(VOICE_DIR, "en_US-lessac-medium")
    # Attempt to download Hindi voice as well if requested or network permits
    if "--all" in sys.argv or os.getenv("DOWNLOAD_ALL_VOICES", "false").lower() in ("true", "1"):
        ensure_piper_voice(VOICE_DIR, "hi_IN-pratham-medium")
    generate_sample_meeting_audio()
    sys.exit(0 if success_en else 1)
