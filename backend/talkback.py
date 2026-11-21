"""
talkback.py
===========
PURPOSE
    TALKBACK voice: reads the English translation aloud, OFFLINE, with a
    PIPER voice (for example en_US-lessac-medium.onnx).

WHY IT EXISTS
    The browser's own voices sound robotic. Piper is a neural voice that
    runs on this computer, so no text ever leaves SAIL. The page asks this
    module for sound through routes_http.py (/speak).

FUNCTIONS
    find_piper_voice : looks for the voice file on this computer
    load_piper       : loads the voice once (Python package or piper.exe)
    piper_wav        : English text -> WAV sound bytes

TO CHOOSE THE VOICE FILE YOURSELF (PowerShell, before starting):
    $env:PIPER_VOICE="C:\\full\\path\\en_US-lessac-medium.onnx"
"""

import os          # file paths and searching folders
import tempfile    # temporary folder for the piper.exe fallback
import threading   # lock, so two requests never load/use Piper together

import settings    # settings.HERE = the backend folder

# WHAT : one lock for loading AND using Piper.
# WHY  : the voice engine is not safe to use from two threads at once.
piper_lock = threading.Lock()

# WHAT : remembers what load_piper() found, so the (slow) search and load
#        happen only once.
#   engine : "python" (piper package) or "exe" (piper.exe program)
#   voice  : the loaded voice object, or the path of piper.exe
#   path   : the .onnx voice file
#   tried  : True once a load was attempted (success OR failure)
piper_cache = {"voice": None, "path": None, "engine": None, "tried": False}


def find_piper_voice():
    """
    PURPOSE : find a Piper English voice file (.onnx) on this computer.
    ORDER   : 1. the PIPER_VOICE environment variable (always wins)
              2. common folders, searched a few levels deep
    PREFERS : a "lessac" voice when several are found.
    RETURNS : the file path, or None.
    """
    # 1) The user pointed to the file explicitly
    wanted = os.environ.get("PIPER_VOICE", "").strip()

    if wanted and os.path.exists(wanted):
        return wanted

    # 2) Folders where a voice is likely to be
    home = os.path.expanduser("~")
    folders = [
        os.path.join(settings.HERE, "models"),
        settings.HERE,
        os.path.join(settings.HERE, ".."),
        os.path.join(home, "piper"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Documents"),
    ]

    found = []

    # LOOP 1 - belongs to: find_piper_voice(), step 2
    # WHY  : the voice file could be in any of the likely folders.
    # REPEATS: once per folder in the list above.
    for folder in folders:
        # LOOP 2 - belongs to: find_piper_voice(), inside LOOP 1
        # WHY  : os.walk visits the folder and every sub-folder in turn.
        # REPEATS: once per sub-folder found.
        for root, dirs, files in os.walk(folder):
            # Do not wander into deep folders (could take very long)
            depth = root[len(folder):].count(os.sep)

            if depth >= 3:
                dirs[:] = []   # tells os.walk not to go deeper

            # Skip folders that can never hold the voice and are huge
            # ONE-LINE LOOP - belongs to: LOOP 2, per visited folder
            # WHY  : removes the unwanted sub-folders from the list that os.walk will
            #        visit next (this is how os.walk is told to skip them).
            # REPEATS: once per sub-folder of the visited folder.
            dirs[:] = [d for d in dirs if d not in
                       ("node_modules", ".git", "venv", ".venv",
                        "convert_env", "__pycache__")]

            # LOOP 3 - belongs to: find_piper_voice(), inside LOOP 2
            # WHY  : check every file of this sub-folder against the
            #        voice-file pattern (starts with "en_", ends ".onnx").
            # REPEATS: once per file.
            for name in files:
                lower = name.lower()

                if lower.endswith(".onnx") and lower.startswith("en_"):
                    found.append(os.path.join(root, name))

    # Lessac voices first (0), everything else after (1)
    found.sort(key=lambda path: (0 if "lessac" in path.lower() else 1, path))

    return found[0] if found else None


def load_piper():
    """
    PURPOSE : load the Piper voice the first time it is needed.
    RETURNS : (engine, voice_or_program, voice_path), or None when Piper
              cannot be used.
    WHY     : the result (even a failure) is remembered, so the page can
              ask "is talkback available?" cheaply and often.
    """
    with piper_lock:
        # Already tried once: return the remembered answer
        if piper_cache["tried"]:
            if piper_cache["engine"]:
                return (piper_cache["engine"], piper_cache["voice"],
                        piper_cache["path"])
            return None

        piper_cache["tried"] = True
        path = find_piper_voice()

        # No voice file anywhere: talkback is not available
        if not path:
            print("Talkback: no Piper voice file (.onnx) found.")
            return None

        try:
            # PREFERRED: the piper Python package (fastest, no extra files)
            from piper import PiperVoice

            voice = PiperVoice.load(path)
            piper_cache.update(engine="python", voice=voice, path=path)
            print("Talkback: Piper voice loaded:", path)

        except Exception as error:
            # SECOND CHOICE: the piper.exe program, if it is on the PATH
            import shutil

            exe = shutil.which("piper") or shutil.which("piper.exe")

            if exe:
                piper_cache.update(engine="exe", voice=exe, path=path)
                print("Talkback: using piper program:", exe, "with", path)
            else:
                print("Talkback: Piper is not usable:", error)
                return None

        return (piper_cache["engine"], piper_cache["voice"],
                piper_cache["path"])


def piper_wav(text):
    """
    PURPOSE : English text -> WAV sound bytes (offline).
    RAISES  : RuntimeError when Piper is not available (the web route
              turns that into an error answer, and the page then falls
              back to the Windows voice).
    """
    import io
    import wave
    import subprocess

    loaded = load_piper()

    if not loaded:
        raise RuntimeError("Piper voice is not available")

    engine, voice, path = loaded

    # ---- Method 1: Python package - write the sound into memory ----
    if engine == "python":
        buffer = io.BytesIO()

        # Only one sound is made at a time
        with piper_lock:
            with wave.open(buffer, "wb") as wav_file:
                # Newer package versions have synthesize_wav, older ones
                # synthesize; support both so an update does not break us.
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wav_file)
                else:
                    voice.synthesize(text, wav_file)

        return buffer.getvalue()

    # ---- Method 2: piper.exe - write a temporary file, then read it ----
    with tempfile.TemporaryDirectory() as folder:
        out = os.path.join(folder, "speech.wav")

        subprocess.run(
            [voice, "--model", path, "--output_file", out],
            input=text.encode("utf-8"),   # the text is fed to the program
            check=True,                   # raise if the program fails
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60                    # never hang the server
        )

        with open(out, "rb") as handle:
            return handle.read()
