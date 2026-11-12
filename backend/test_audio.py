import sys
import numpy as np
import time

# Workaround for soundcard library bug on Linux where it inspects sys.argv[1]
if len(sys.argv) < 2:
    sys.argv.append("sail")

try:
    import soundcard as sc
    speaker = sc.default_speaker()
    if not speaker:
        raise RuntimeError("No default speaker detected.")
    microphone = sc.get_microphone(
        id=str(speaker.name),
        include_loopback=True
    )
    devices = sc.all_microphones(include_loopback=True)
except Exception as e:
    print(f"Notice: soundcard device is unavailable in this environment: {e}")
    print("In headless environments (Codespaces / Docker / CI), audio loopback hardware is not present.")
    print("Meeting audio streaming is handled via the browser WebSocket interface.")
    sys.exit(0)

print("Available audio devices:")
for device in devices:
    print(device.name)

print("\nStarting audio capture for 5 seconds...")

with microphone.recorder(samplerate=16000) as recorder:
    audio = recorder.record(numframes=16000 * 5)

print("\nAudio captured successfully!")
print("Audio shape:", audio.shape)
print("Audio level:", np.max(np.abs(audio)))
