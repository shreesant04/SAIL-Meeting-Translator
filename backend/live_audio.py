import sys
import numpy as np
import time

# Workaround for soundcard library bug on Linux where it inspects sys.argv[1]
if len(sys.argv) < 2:
    sys.argv.append("sail")

try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except Exception:
    SOUNDCARD_AVAILABLE = False
    sc = None

def capture_audio(duration=5, sample_rate=16000):
    if not SOUNDCARD_AVAILABLE or sc is None:
        raise RuntimeError("soundcard device is not available in this environment. If running headlessly, use browser-based streaming instead.")

    try:
        speaker = sc.default_speaker()
        if not speaker:
            raise RuntimeError("No default speaker found. Verify audio device configuration.")

        microphone = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True
        )
    except Exception as e:
        raise RuntimeError(f"Audio device initialization failed: {e}. If running headlessly, use browser-based streaming instead.")

    print(f"Capturing {duration}s meeting audio from speaker loopback ({speaker.name})...")

    with microphone.recorder(samplerate=sample_rate) as recorder:
        audio = recorder.record(numframes=sample_rate * duration)

    audio = audio.mean(axis=1).astype(np.float32)

    print("Audio captured successfully!")
    print(f"Duration: {len(audio) / sample_rate:.2f}s | Peak level: {np.max(np.abs(audio)):.4f}")

    return audio

if __name__ == "__main__":
    print("=== SAIL Live Audio Capture Test ===")
    print("Play speech or audio through your computer speakers.")
    try:
        time.sleep(1)
        captured = capture_audio(duration=5)
        print("Capture test completed successfully.")
    except Exception as err:
        print(f"Notice: {err}", file=sys.stderr)
        sys.exit(0)
