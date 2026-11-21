import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

// Seconds of audio sent to the backend at a time.
// Longer chunks = fewer pauses and less skipped audio on a slow CPU.
const CHUNK_SECONDS = 10;

// Spoken language the user can pick (auto = let the AI guess)
const LANGUAGE_OPTIONS = [
  ["hi", "Hindi"],
  ["bn", "Bengali"],
  ["ko", "Korean"],
  ["auto", "Auto-detect (Hindi / Bengali / Korean)"]
];

// 125.5 seconds -> "02:05"
function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const rest = String(total % 60).padStart(2, "0");

  return minutes + ":" + rest;
}

function App() {
  const [status, setStatus] = useState("Ready");
  const [audioFile, setAudioFile] = useState(null);
  // One row = what was said + its English meaning
  const [rows, setRows] = useState([]);
  const [detectedLanguage, setDetectedLanguage] = useState("");
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [error, setError] = useState("");

  const [sourceLanguage, setSourceLanguage] = useState("hi");
  const [fileAccuracy, setFileAccuracy] = useState("best");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);
  const fileAbortRef = useRef(null);
  const [progress, setProgress] = useState(0);

  const [recordEnabled, setRecordEnabled] = useState(false);
  const [recordingUrl, setRecordingUrl] = useState("");
  const [recordingName, setRecordingName] = useState("");
  const [filesOpen, setFilesOpen] = useState(false);

  // The SAIL logo is loaded from frontend/public/sail-logo.png.
  // If that file is missing, the letter "S" is shown instead.
  const [logoOk, setLogoOk] = useState(true);

  // Talkback: read the English sentences aloud (voices on this PC only)
  const [talkback, setTalkback] = useState(false);
  const [voices, setVoices] = useState([]);
  const [voiceName, setVoiceName] = useState("");
  const [talkbackVolume, setTalkbackVolume] = useState(1);
  const talkbackRef = useRef(false);
  const voiceNameRef = useRef("");
  const talkbackVolumeRef = useRef(1);
  const speakQueueRef = useRef([]);
  const speakingRef = useRef(false);
  const [piperName, setPiperName] = useState("");
  const [voiceEngine, setVoiceEngine] = useState("windows");
  const voiceEngineRef = useRef("windows");
  const piperAudioRef = useRef(null);

  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const contextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const gainRef = useRef(null);

  const audioBufferRef = useRef([]);
  const sampleCountRef = useRef(0);

  const startingRef = useRef(false);
  const stoppingRef = useRef(false);

  const conversationRef = useRef(null);

  const recorderRef = useRef(null);
  const recordChunksRef = useRef([]);
  const recordEnabledRef = useRef(false);
  const recordingUrlRef = useRef("");

  // Keep the newest row in view
  useEffect(() => {
    const box = conversationRef.current;

    if (box) {
      box.scrollTop = box.scrollHeight;
    }
  }, [rows]);

  // Release the microphone/screen capture if the page is closed
  useEffect(() => {
    return () => {
      cleanupAudio();
    };
  }, []);

  const startRecording = stream => {
    try {
      // Free the previous recording from memory
      if (recordingUrlRef.current) {
        URL.revokeObjectURL(recordingUrlRef.current);
        recordingUrlRef.current = "";
      }

      setRecordingUrl("");
      setRecordingName("");

      const audioTracks = stream.getAudioTracks();

      if (audioTracks.length === 0) return;

      const audioOnly = new MediaStream(audioTracks);

      const mimeType = MediaRecorder.isTypeSupported(
        "audio/webm;codecs=opus"
      )
        ? "audio/webm;codecs=opus"
        : "";

      const recorder = mimeType
        ? new MediaRecorder(audioOnly, { mimeType })
        : new MediaRecorder(audioOnly);

      recordChunksRef.current = [];

      recorder.ondataavailable = event => {
        if (event.data && event.data.size > 0) {
          recordChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const chunks = recordChunksRef.current;
        recordChunksRef.current = [];

        if (chunks.length === 0) return;

        const blob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm"
        });

        const url = URL.createObjectURL(blob);
        recordingUrlRef.current = url;

        const now = new Date();
        const pad = number => String(number).padStart(2, "0");

        const name =
          "meeting-audio-" +
          now.getFullYear() +
          pad(now.getMonth() + 1) +
          pad(now.getDate()) +
          "-" +
          pad(now.getHours()) +
          pad(now.getMinutes()) +
          pad(now.getSeconds()) +
          ".webm";

        setRecordingUrl(url);
        setRecordingName(name);
        setFilesOpen(true);
      };

      recorder.start(1000);
      recorderRef.current = recorder;
    } catch (err) {
      console.error("Recording could not start:", err);
      setError("Recording could not start: " + err.message);
    }
  };

  const stopRecording = () => {
    const recorder = recorderRef.current;
    recorderRef.current = null;

    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch (err) {
        console.error("Recording stop error:", err);
      }
    }
  };

  const cleanupAudio = () => {
    // Stop recording first so the last seconds are saved
    stopRecording();

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }

    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    if (gainRef.current) {
      gainRef.current.disconnect();
      gainRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (contextRef.current) {
      contextRef.current.close().catch(() => {});
      contextRef.current = null;
    }

    audioBufferRef.current = [];
    sampleCountRef.current = 0;
  };

  // ---------- Talkback (text-to-speech) ----------
  // Uses the voices installed in Windows. Only voices that run on this
  // computer are offered, so no text is ever sent to the internet.
  const loadVoices = () => {
    if (!("speechSynthesis" in window)) return;

    const local = window.speechSynthesis
      .getVoices()
      .filter(voice => voice.localService && /^en/i.test(voice.lang));

    setVoices(local);

    if (!voiceNameRef.current && local.length > 0) {
      voiceNameRef.current = local[0].name;
      setVoiceName(local[0].name);
    }
  };

  useEffect(() => {
    if (!("speechSynthesis" in window)) return undefined;

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      window.speechSynthesis.onvoiceschanged = null;
      window.speechSynthesis.cancel();
    };
  }, []);

  const speakNext = () => {
    if (speakingRef.current) return;

    const text = speakQueueRef.current.shift();
    if (!text) return;

    const waitingNow = speakQueueRef.current.length;

    if (voiceEngineRef.current === "piper") {
      speakingRef.current = true;

      const finish = () => {
        piperAudioRef.current = null;
        speakingRef.current = false;
        speakNext();
      };

      const form = new FormData();
      form.append("text", text);

      fetch(`${API}/speak`, { method: "POST", body: form })
        .then(response => {
          if (!response.ok) throw new Error("Piper voice failed");
          return response.blob();
        })
        .then(blob => {
          // Stop was pressed while the voice was being made
          if (!speakingRef.current) return;

          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);

          audio.volume = talkbackVolumeRef.current;
          audio.playbackRate = waitingNow >= 3 ? 1.3 : waitingNow >= 1 ? 1.12 : 1;

          const clean = () => {
            URL.revokeObjectURL(url);
            finish();
          };

          audio.onended = clean;
          audio.onerror = clean;
          piperAudioRef.current = audio;
          audio.play().catch(clean);
        })
        .catch(() => {
          // Piper not working: use the Windows voice from now on
          voiceEngineRef.current = "windows";
          setVoiceEngine("windows");
          speakingRef.current = false;
          speakQueueRef.current.unshift(text);
          speakNext();
        });

      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);

    const voice = window.speechSynthesis
      .getVoices()
      .find(item => item.name === voiceNameRef.current && item.localService);

    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    } else {
      utterance.lang = "en-US";
    }

    utterance.volume = talkbackVolumeRef.current;

    // Falling behind? Speak a little faster so it catches up.
    const waiting = speakQueueRef.current.length;
    utterance.rate = waiting >= 3 ? 1.35 : waiting >= 1 ? 1.15 : 1;

    const done = () => {
      speakingRef.current = false;
      speakNext();
    };

    utterance.onend = done;
    utterance.onerror = done;

    speakingRef.current = true;
    window.speechSynthesis.speak(utterance);
  };

  const speakEnglish = text => {
    if (!talkbackRef.current || !text || !text.trim()) return;
    if (!("speechSynthesis" in window)) return;

    speakQueueRef.current.push(text.trim());

    // Very far behind: drop the oldest SPOKEN sentences (the text rows
    // on screen are never dropped) so the voice stays near the present.
    while (speakQueueRef.current.length > 6) {
      speakQueueRef.current.shift();
    }

    speakNext();
  };

  const stopSpeaking = () => {
    speakQueueRef.current = [];
    speakingRef.current = false;

    if (piperAudioRef.current) {
      piperAudioRef.current.pause();
      piperAudioRef.current = null;
    }

    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  };

  // Handles one message from the backend (live audio AND file jobs).
  // Each row has an id: the original text arrives first, then its
  // English meaning is filled in on the same row.
  const applyEvent = data => {
    if (data.type === "status") {
      setStatus(data.message);
    }

    if (data.type === "error") {
      setError(data.message);
    }

    if (data.type === "language") {
      setDetectedLanguage(data.language_name || data.language);
    }

    if (data.type === "progress") {
      setProgress(data.percent);
    }

    if (data.type === "original_transcript") {
      setDetectedLanguage(data.language_name || data.language);

      setRows(previous => {
        const found = previous.findIndex(row => row.id === data.id);

        if (found >= 0) {
          const copy = [...previous];

          copy[found] = {
            ...copy[found],
            original: data.text,
            roman: data.roman || ""
          };

          return copy;
        }

        return [
          ...previous,
          {
            id: data.id,
            start: data.start,
            end: data.end,
            original: data.text,
            roman: data.roman || "",
            english: null
          }
        ].slice(-300);
      });
    }

    if (data.type === "translation") {
      // Read the English aloud (live meetings only, not file jobs)
      if (socketRef.current) {
        speakEnglish(data.text);
      }

      if (data.source_language_name) {
        setDetectedLanguage(data.source_language_name);
      }

      setRows(previous => {
        const found = previous.findIndex(row => row.id === data.id);

        if (found >= 0) {
          const copy = [...previous];
          copy[found] = { ...copy[found], english: data.text };
          return copy;
        }

        return [
          ...previous,
          {
            id: data.id,
            start: data.start,
            end: data.end,
            original: "",
            roman: "",
            english: data.text
          }
        ].slice(-300);
      });
    }
  };

  const stopTranslation = () => {
    // Also cancels a running file translation
    if (fileAbortRef.current) {
      fileAbortRef.current.abort();
      fileAbortRef.current = null;
    }

    stoppingRef.current = true;
    startingRef.current = false;

    stopSpeaking();

    const socket = socketRef.current;
    socketRef.current = null;

    cleanupAudio();

    if (
      socket &&
      (socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING)
    ) {
      socket.close(1000, "Stopped by user");
    }

    setIsLive(false);
    setStatus("Stopped");
  };

  const startFileTranslation = () => {
    if (!audioFile) {
      setError("Please select an audio file first.");
      return;
    }

    runFileTranslation(audioFile);
  };

  // Sends the recorded meeting to the backend for translation
  const translateRecording = async () => {
    try {
      const response = await fetch(recordingUrl);
      const blob = await response.blob();

      const file = new File(
        [blob],
        recordingName || "meeting-audio.webm",
        { type: blob.type || "audio/webm" }
      );

      runFileTranslation(file);
    } catch (err) {
      setError("Could not open the recording: " + err.message);
    }
  };

  const runFileTranslation = async file => {
    setLoading(true);
    setError("");
    setRows([]);
    setDetectedLanguage("");
    setElapsed(0);
    setProgress(0);
    setStatus("Uploading the audio...");

    if (timerRef.current) clearInterval(timerRef.current);

    timerRef.current = setInterval(
      () => setElapsed(seconds => seconds + 1),
      1000
    );

    const controller = new AbortController();
    fileAbortRef.current = controller;

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("language", sourceLanguage);
      formData.append("accuracy", fileAccuracy);

      const response = await fetch(`${API}/translate-audio`, {
        method: "POST",
        body: formData,
        signal: controller.signal
      });

      if (!response.ok || !response.body) {
        let message = "Translation failed.";

        try {
          const data = await response.json();
          message = data.detail || message;
        } catch {
          /* keep the default message */
        }

        throw new Error(message);
      }

      // The backend sends one JSON message per line, as soon as each
      // sentence is ready.
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let pending = "";
      let finished = false;

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        pending += decoder.decode(value, { stream: true });

        const lines = pending.split("\n");
        pending = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;

          const data = JSON.parse(line);

          if (data.type === "done") {
            finished = true;
            continue;
          }

          applyEvent(data);
        }
      }

      if (finished) {
        setProgress(100);
        setStatus("Translation completed");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        setStatus("Stopped");
      } else {
        setError(err.message);
        setStatus("Translation failed");
      }
    } finally {
      fileAbortRef.current = null;

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      setLoading(false);
    }
  };

  const startLiveTranslation = async () => {
    // Prevent multiple simultaneous start requests
    if (startingRef.current || socketRef.current) {
      return;
    }

    startingRef.current = true;
    stoppingRef.current = false;

    setError("");
    setRows([]);
    setDetectedLanguage("");
    setStatus("Requesting screen audio permission...");

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
        systemAudio: "include",
        selfBrowserSurface: "exclude"
      });

      if (stoppingRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      streamRef.current = stream;

      if (stream.getAudioTracks().length === 0) {
        throw new Error(
          "No system audio received. Select Entire Screen and enable system audio."
        );
      }

      if (recordEnabledRef.current) {
        startRecording(stream);
      }

      const socket = new WebSocket(
        `ws://127.0.0.1:8000/ws/browser-live?language=${sourceLanguage}`
      );

      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onmessage = event => {
        try {
          applyEvent(JSON.parse(event.data));
        } catch (err) {
          console.error("Message error:", err);
          setError("Could not read a backend message.");
        }
      };

      socket.onerror = event => {
        console.error("WebSocket error:", event);
        setError("WebSocket connection error. Check the backend.");
      };

      socket.onclose = event => {
        console.log(
          "WebSocket closed:",
          "Code:", event.code,
          "Reason:", event.reason,
          "Clean:", event.wasClean
        );

        if (socketRef.current !== socket) {
          return;
        }

        socketRef.current = null;
        startingRef.current = false;

        cleanupAudio();
        setIsLive(false);

        if (!stoppingRef.current) {
          setStatus("Connection lost");

          setError(
            `Audio connection closed. Code: ${event.code}. Reason: ${
              event.reason || "No reason supplied"
            }.`
          );
        }
      };

      socket.onopen = async () => {
        try {
          if (stoppingRef.current) {
            socket.close(1000, "Stopped");
            return;
          }

          const context = new AudioContext({
            sampleRate: 16000
          });

          contextRef.current = context;
          await context.resume();

          if (context.sampleRate !== 16000) {
            throw new Error(
              `Audio sample rate is ${context.sampleRate} Hz, not 16000 Hz.`
            );
          }

          const source = context.createMediaStreamSource(stream);
          sourceRef.current = source;

          const processor = context.createScriptProcessor(
            4096,
            1,
            1
          );

          processorRef.current = processor;

          const gain = context.createGain();
          gain.gain.value = 0;
          gainRef.current = gain;

          audioBufferRef.current = [];
          sampleCountRef.current = 0;

          processor.onaudioprocess = event => {
            if (
              socket.readyState !== WebSocket.OPEN ||
              stoppingRef.current
            ) {
              return;
            }

            const samples = event.inputBuffer.getChannelData(0);
            const copy = new Float32Array(samples);

            audioBufferRef.current.push(copy);
            sampleCountRef.current += copy.length;

            const targetSamples = 16000 * CHUNK_SECONDS;

            if (sampleCountRef.current >= targetSamples) {
              const combined = new Float32Array(
                sampleCountRef.current
              );

              let offset = 0;

              for (const part of audioBufferRef.current) {
                combined.set(part, offset);
                offset += part.length;
              }

              // Avoid sending if the connection is no longer open
              if (socket.readyState === WebSocket.OPEN) {
                socket.send(combined.buffer);
              }

              audioBufferRef.current = [];
              sampleCountRef.current = 0;
            }
          };

          source.connect(processor);
          processor.connect(gain);
          gain.connect(context.destination);

          startingRef.current = false;
          setIsLive(true);
          setStatus("Listening to system audio...");
        } catch (err) {
          console.error("Audio setup error:", err);

          setError("Audio setup failed: " + err.message);
          startingRef.current = false;

          cleanupAudio();

          if (socketRef.current === socket) {
            socketRef.current = null;
          }

          socket.close(1011, "Audio setup failed");
          setIsLive(false);
        }
      };

      stream.getVideoTracks()[0]?.addEventListener(
        "ended",
        () => {
          if (socketRef.current === socket) {
            stopTranslation();
          }
        },
        { once: true }
      );

    } catch (err) {
      console.error("Live translation start error:", err);

      cleanupAudio();
      startingRef.current = false;

      setError(err.message);
      setStatus("Audio capture failed");
      setIsLive(false);
    }
  };

  const latestEnglish = [...rows]
    .reverse()
    .find(row => row.english);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          {logoOk ? (
            <img
              className="brand-logo-img"
              src="/sail-logo.png"
              alt="SAIL"
              onError={() => setLogoOk(false)}
            />
          ) : (
            <div className="brand-logo">S</div>
          )}
          <h2>SAIL-ISP</h2>
        </div>

        <div className="header-status">
          <span className={isLive ? "live-dot on" : "live-dot"}></span>
          <strong>{isLive ? "LIVE" : "READY"}</strong>
        </div>
      </header>

      <main className="workspace">
        {/* ---------- LEFT: menu ---------- */}
        <aside className="sidebar">
          <section className="panel">
            <h3>Live meeting</h3>

            <label className="field">
              <span>Spoken language</span>
              <select
                value={sourceLanguage}
                disabled={isLive || loading}
                onChange={event => setSourceLanguage(event.target.value)}
              >
                {LANGUAGE_OPTIONS.map(([code, name]) => (
                  <option key={code} value={code}>
                    {name}
                  </option>
                ))}
              </select>
            </label>

            <div className="button-row">
              <button
                className="start-button"
                onClick={startLiveTranslation}
                disabled={isLive || loading || startingRef.current}
              >
                {isLive ? "Listening..." : "Start Live Translation"}
              </button>

              <button
                className="stop-button"
                onClick={stopTranslation}
                disabled={!isLive && !loading && !startingRef.current}
              >
                Stop
              </button>
            </div>

            <p className="status-line">
              {status}
              {loading && (
                <span>
                  {" "}
                  ({formatTime(elapsed)}
                  {progress > 0 ? ` · ${progress}%` : ""})
                </span>
              )}
            </p>

            {error && <p className="error-line">{error}</p>}
          </section>

          <section className="panel">
            <h3>Options</h3>

            <label className="check">
              <input
                type="checkbox"
                checked={recordEnabled}
                disabled={isLive}
                onChange={event => {
                  setRecordEnabled(event.target.checked);
                  recordEnabledRef.current = event.target.checked;
                }}
              />
              Record meeting audio (saved on this computer only)
            </label>

            <label className="check">
              <input
                type="checkbox"
                checked={talkback}
                disabled={!("speechSynthesis" in window)}
                onChange={event => {
                  setTalkback(event.target.checked);
                  talkbackRef.current = event.target.checked;

                  if (!event.target.checked) {
                    stopSpeaking();
                  } else {
                    // Is the better offline Piper voice available?
                    fetch(`${API}/speak-status`)
                      .then(response => response.json())
                      .then(info => {
                        if (info.available) {
                          setPiperName(info.voice);
                          voiceEngineRef.current = "piper";
                          setVoiceEngine("piper");
                        }
                      })
                      .catch(() => {});
                  }
                }}
              />
              Read the English aloud (talkback)
            </label>

            {talkback && (
              <div className="talkback-settings">
                {piperName && (
                  <select
                    value={voiceEngine}
                    onChange={event => {
                      stopSpeaking();
                      setVoiceEngine(event.target.value);
                      voiceEngineRef.current = event.target.value;
                    }}
                  >
                    <option value="piper">Piper: {piperName}</option>
                    <option value="windows">Windows voice</option>
                  </select>
                )}

                {voiceEngine === "windows" && voices.length === 0 ? (
                  <p className="hint">
                    No offline English voice found. In Windows: Settings,
                    Time and language, Speech, Add voices.
                  </p>
                ) : (
                  <>
                    {voiceEngine === "windows" && (
                      <select
                        value={voiceName}
                        onChange={event => {
                          setVoiceName(event.target.value);
                          voiceNameRef.current = event.target.value;
                        }}
                      >
                        {voices.map(voice => (
                          <option key={voice.name} value={voice.name}>
                            {voice.name}
                          </option>
                        ))}
                      </select>
                    )}

                    <div className="talkback-row">
                      <input
                        type="range"
                        min="0.2"
                        max="1"
                        step="0.1"
                        value={talkbackVolume}
                        title="Talkback volume"
                        onChange={event => {
                          const value = Number(event.target.value);
                          setTalkbackVolume(value);
                          talkbackVolumeRef.current = value;
                        }}
                      />

                      <button
                        type="button"
                        className="small-button"
                        onClick={() => {
                          stopSpeaking();
                          speakEnglish("This is the translation voice test.");
                        }}
                      >
                        Test voice
                      </button>
                    </div>
                  </>
                )}

                <p className="hint">
                  Share only the meeting tab, or use headphones, so the
                  translator does not hear its own voice.
                </p>
              </div>
            )}
          </section>

          <details
            className="panel"
            open={filesOpen}
            onToggle={event => setFilesOpen(event.currentTarget.open)}
          >
            <summary>Recorded audio</summary>

            <label className="field">
              <span>Accuracy</span>
              <select
                value={fileAccuracy}
                disabled={isLive || loading}
                onChange={event => setFileAccuracy(event.target.value)}
              >
                <option value="best">Best accuracy (slower)</option>
                <option value="fast">Fast</option>
              </select>
            </label>

            <input
              className="file-input"
              type="file"
              accept="audio/*"
              onChange={event => {
                setAudioFile(event.target.files[0] || null);
                setError("");
              }}
            />

            <button
              className="start-button wide"
              onClick={startFileTranslation}
              disabled={loading || isLive || !audioFile}
            >
              {loading ? "Translating..." : "Translate File"}
            </button>

            {recordingUrl && (
              <div className="recording-box">
                <p className="hint">
                  Recording saved: <strong>{recordingName}</strong>
                </p>
                <audio controls src={recordingUrl}></audio>
                <div className="button-row">
                  <a
                    className="download-link"
                    href={recordingUrl}
                    download={recordingName}
                  >
                    Download
                  </a>
                  <button
                    className="small-button"
                    onClick={translateRecording}
                    disabled={loading || isLive}
                  >
                    {loading ? "Translating..." : "Translate this recording"}
                  </button>
                </div>
              </div>
            )}
          </details>
        </aside>

        {/* ---------- RIGHT: subtitles and conversation ---------- */}
        <section className="stage">
          <div className="subtitle-card">
            <div className="subtitle-head">
              <span className="muted">
                Detected language:{" "}
                <strong>
                  {detectedLanguage || "Waiting for speech..."}
                </strong>
              </span>
            </div>

            <div className="subtitle-english">
              {latestEnglish
                ? latestEnglish.english
                : isLive
                ? "Listening... speak now"
                : "English subtitles will appear here"}
            </div>

            {latestEnglish && latestEnglish.original && (
              <div className="subtitle-original">
                {latestEnglish.original}
                {latestEnglish.roman && (
                  <span className="subtitle-roman">
                    {" "}
                    · {latestEnglish.roman}
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="conversation">
            <div className="conversation-top">
              <h3>Conversation</h3>
              <span className="live-label">
                {rows.length} {rows.length === 1 ? "LINE" : "LINES"}
              </span>
            </div>

            <div className="conversation-head">
              <span>Original</span>
              <span>English</span>
            </div>

            <div className="conversation-body" ref={conversationRef}>
              {rows.length === 0 ? (
                <div className="conversation-empty">
                  <div className="empty-icon">Aa</div>
                  <p>
                    The original speech shows first, and the English
                    follows a moment later on the same row.
                  </p>
                </div>
              ) : (
                rows.map((row, index) => (
                  <div
                    className="conversation-row"
                    key={row.id ?? index}
                  >
                    <div className="cell original-cell">
                      <span className="time">
                        {formatTime(row.start)}
                      </span>
                      <span>{row.original || "—"}</span>
                      {row.roman && (
                        <div className="roman-line">{row.roman}</div>
                      )}
                    </div>

                    <div className="cell english-cell">
                      {row.english === null ? (
                        <span className="pending">
                          Translating<span className="dots"></span>
                        </span>
                      ) : (
                        row.english || "—"
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;