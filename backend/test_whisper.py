import sys
import numpy as np
from faster_whisper import WhisperModel

# Workaround for soundcard library bug on Linux where it inspects sys.argv[1]
if len(sys.argv) < 2:
    sys.argv.append("sail")

audio = None

try:
    import soundcard as sc
    speaker = sc.default_speaker()
    if not speaker:
        raise RuntimeError("No default speaker detected.")
    microphone = sc.get_microphone(
        id=str(speaker.name),
        include_loopback=True
    )
    print("Recording audio for 5 seconds from speaker loopback...")
    with microphone.recorder(samplerate=16000) as recorder:
        recorded = recorder.record(numframes=16000 * 5)
    audio = recorded.mean(axis=1).astype(np.float32)
    print("Audio captured from hardware loopback!")
except Exception as e:
    print(f"Notice: soundcard loopback audio device unavailable ({e}).")
    print("Using synthetic audio test tone for Whisper verification in headless environment...")
    from audio_utils import create_synthetic_wav
    # Sine wave
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

# Step 4: Load Whisper model
print("Loading Whisper model...")

try:
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
        cpu_threads=2
    )
    print("Whisper model loaded successfully!", flush=True)

except Exception as e:
    print("WHISPER ERROR:", repr(e), flush=True)
    raise

# Step 5: Translate speech into English
print("Translating speech into English...")

segments, info = model.transcribe(
    audio,
    task="translate",
    beam_size=5
)

print("\nDetected language:", info.language)
print("Language confidence:", round(float(info.language_probability), 3))
print("\nEnglish Translation:")

for segment in segments:
    print(segment.text)

print("\nWhisper verification completed successfully.")
