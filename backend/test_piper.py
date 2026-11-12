import os
import wave
from piper import PiperVoice

VOICE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "voices",
    "en_US-lessac-medium.onnx"
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "talkback_output.wav"
)

# Load the local English voice
voice = PiperVoice.load(VOICE_PATH)

# Generate English speech
text = "Hello! This is the SAIL Meeting Translator. English talkback is working successfully."

with wave.open(OUTPUT_PATH, "wb") as audio_file:
    voice.synthesize_wav(text, audio_file)

print(f"SUCCESS: English talkback audio generated at {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes)!")
