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

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None

SAMPLE_RATE = 16000
CHUNK_DURATION = 5

def run_live_translate(model_size="tiny", device="cpu", compute_type="int8"):
    if not SOUNDCARD_AVAILABLE or sc is None:
        print("Notice: soundcard loopback audio device is not available in this environment.", file=sys.stderr)
        print("On headless cloud environments (Codespaces / Docker), use the web / WebSocket interface.", file=sys.stderr)
        return

    if not WHISPER_AVAILABLE:
        print("Error: faster-whisper is not installed.", file=sys.stderr)
        return

    print(f"Loading Whisper '{model_size}' model...", flush=True)
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=2
    )
    print("Whisper model loaded successfully!", flush=True)

    try:
        speaker = sc.default_speaker()
        if not speaker:
            print("Error: No default speaker found.", file=sys.stderr)
            return

        microphone = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True
        )
    except Exception as e:
        print(f"Loopback microphone initialization error: {e}", file=sys.stderr)
        print("Note: On headless machines or remote servers, use the WebSocket/web interface.", file=sys.stderr)
        return

    print("\n===========================================")
    print("      SAIL LIVE MEETING TRANSLATOR")
    print("===========================================")
    print(f"Recording device: {speaker.name} (Loopback)")
    print(f"Chunk size: {CHUNK_DURATION}s | Sample rate: {SAMPLE_RATE}Hz")
    print("Play speech in any language through your speakers.")
    print("Press Ctrl+C to terminate.\n")

    try:
        with microphone.recorder(samplerate=SAMPLE_RATE) as recorder:
            while True:
                audio = recorder.record(numframes=SAMPLE_RATE * CHUNK_DURATION)
                audio = audio.mean(axis=1).astype(np.float32)

                peak = np.max(np.abs(audio))
                if peak < 0.005:
                    print(f"[{time.strftime('%H:%M:%S')}] Silence / no speech (peak: {peak:.4f}). Waiting...")
                    continue

                print(f"[{time.strftime('%H:%M:%S')}] Speech detected (peak: {peak:.4f}). Translating...")

                segments, info = model.transcribe(
                    audio,
                    task="translate",
                    beam_size=1
                )

                print(f"  Detected language: {info.language} (confidence: {info.language_probability:.2f})")
                has_text = False
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        print(f"  English: {text}")
                        has_text = True
                if not has_text:
                    print("  (Unclear speech / no text)")

    except KeyboardInterrupt:
        print("\nLive translation terminated by user.")
    except Exception as e:
        print(f"\nLive translation runtime error: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_live_translate()
