import io
import math
import wave
import numpy as np
from scipy import signal

TARGET_SAMPLE_RATE = 16000

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Resample 1D float32 audio array to the target sample rate.
    Safely handles edge cases like zero/negative sample rates or empty arrays.
    """
    if orig_sr <= 0 or target_sr <= 0 or len(audio) == 0:
        return np.empty(0, dtype=np.float32)

    if orig_sr == target_sr:
        return audio.astype(np.float32)

    # Convert multi-channel audio to mono if necessary
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # Polyphase resampling (FIR filter) avoids Gibbs ringing/spectral smearing
    g = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)

def calculate_rms_level(audio: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) energy level of an audio chunk.
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))

def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Safely convert raw binary float32 samples to numpy array.
    """
    if not pcm_bytes or len(pcm_bytes) % 4 != 0:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(pcm_bytes, dtype=np.float32).copy()

def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    """
    Ensures input audio is 1D float32 mono.
    """
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio.astype(np.float32)

def create_synthetic_wav(duration_s: float = 1.0, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    """
    Create a valid in-memory WAV file with a pure sine wave tone (useful for tests).
    """
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())

    return buffer.getvalue()
