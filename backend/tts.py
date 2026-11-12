import io
import os
import wave
import threading
from typing import Optional

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None

DEFAULT_VOICE_REL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "voices",
    "en_US-lessac-medium.onnx"
)

class TTSManager:
    """
    Thread-safe TTS manager for synthesizing speech using Piper.
    Falls back gracefully if piper-tts or voice models are unavailable.
    """
    def __init__(self, voice_path: str = DEFAULT_VOICE_REL_PATH):
        self.voice_path = voice_path
        self._voice: Optional[PiperVoice] = None
        self._lock = threading.Lock()
        self._initialized = False

    def is_model_available(self) -> bool:
        return os.path.exists(self.voice_path) and os.path.getsize(self.voice_path) > 0

    def load_voice(self) -> bool:
        if not PIPER_AVAILABLE:
            return False

        if not self.is_model_available():
            # Attempt to auto-download voice if missing
            try:
                from download_voices import ensure_piper_voice
                voices_dir = os.path.dirname(self.voice_path)
                ensure_piper_voice(voices_dir)
            except Exception as e:
                print(f"[TTS] Voice auto-download failed: {e}")
                return False

        with self._lock:
            if self._voice is not None:
                return True
            try:
                print(f"[TTS] Loading Piper voice from {self.voice_path}...")
                self._voice = PiperVoice.load(self.voice_path)
                self._initialized = True
                print("[TTS] Piper voice loaded successfully.")
                return True
            except Exception as e:
                print(f"[TTS] Failed to load Piper voice: {e}")
                self._initialized = False
                return False

    def synthesize_wav(self, text: str) -> bytes:
        """
        Synthesize text to WAV bytes.
        """
        text = text.strip()
        if not text:
            raise ValueError("Text to synthesize cannot be empty.")

        if not self._initialized:
            success = self.load_voice()
            if not success or self._voice is None:
                raise RuntimeError(
                    "Piper TTS voice is not available. Please ensure the voice model is downloaded."
                )

        with self._lock:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            return buffer.getvalue()

    def get_status(self) -> dict:
        return {
            "available": PIPER_AVAILABLE,
            "model_present": self.is_model_available(),
            "model_loaded": self._initialized and self._voice is not None,
            "voice_path": self.voice_path
        }

# Global singleton instance
tts_manager = TTSManager()
