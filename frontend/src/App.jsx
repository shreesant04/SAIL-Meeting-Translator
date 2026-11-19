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
  const [backendStatus, setBackendStatus] = useState("Checking...");
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

  useEffect(() => {
    let mounted = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${API}/health`);

        if (!response.ok) throw new Error();

        if (mounted) setBackendStatus("Connected");
      } catch {
        if (mounted) setBackendStatus("Disconnected");
      }
    };

    checkBackend();
    const timer = setInterval(checkBackend, 5000);

    return () => {
      mounted = false;
      clearInterval(timer);
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
          <div className="brand-logo">S</div>
          <div>
            <h2>SAIL Meeting Translator</h2>
            <p>Internal Meeting Assistance System</p>
          </div>
        </div>

        <div className="connection">
          <span className="status-dot"></span>
          Backend: {backendStatus}
        </div>
      </header>

      <main className="main">
        <section className="welcome">
          <span className="eyebrow">MEETING WORKSPACE</span>
          <h1>Understand every meeting.</h1>
          <p>
            Translate meeting speech into English using locally
            processed speech recognition.
          </p>
        </section>

        <section className="controls card">
          <h3>Meeting controls</h3>
          <p className="muted">
            Supported languages: Hindi, Bengali and Korean. Pick the
            spoken language for the best accuracy, or use Auto-detect
            if you are not sure.
          </p>

          <div className="option-row">
            <label>
              Spoken language{" "}
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

            <label>
              Recorded audio{" "}
              <select
                value={fileAccuracy}
                disabled={isLive || loading}
                onChange={event => setFileAccuracy(event.target.value)}
              >
                <option value="best">Best accuracy (slower)</option>
                <option value="fast">Fast</option>
              </select>
            </label>
          </div>

          <div className="control-row">
            <input
              type="file"
              accept="audio/*"
              onChange={event => {
                setAudioFile(event.target.files[0] || null);
                setError("");
              }}
            />

            <button
              className="start-button"
              onClick={startFileTranslation}
              disabled={loading || isLive}
            >
              {loading ? "Translating..." : "Translate File"}
            </button>

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

          <label className="record-option">
            <input
              type="checkbox"
              checked={recordEnabled}
              disabled={isLive}
              onChange={event => {
                setRecordEnabled(event.target.checked);
                recordEnabledRef.current = event.target.checked;
              }}
            />{" "}
            Also record meeting audio (saved on this computer only)
          </label>

          <div className="talkback-box">
            <label className="record-option">
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
              />{" "}
              Read the English translation aloud (talkback)
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
                  <p className="status-text">
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
                      onClick={() => {
                        stopSpeaking();
                        speakEnglish("This is the translation voice test.");
                      }}
                    >
                      Test voice
                    </button>
                  </>
                )}

                <p className="status-text">
                  Use headphones, or share only the meeting tab. If the
                  translator hears its own voice, it will translate it again.
                </p>
              </div>
            )}
          </div>

          {recordingUrl && (
            <div className="recording-box">
              <p className="status-text">
                Recording saved: <strong>{recordingName}</strong>
              </p>
              <audio controls src={recordingUrl}></audio>
              <p>
                <a
                  className="download-link"
                  href={recordingUrl}
                  download={recordingName}
                >
                  Download recording
                </a>
              </p>
              <button
                className="start-button"
                onClick={translateRecording}
                disabled={loading || isLive}
              >
                {loading
                  ? "Translating..."
                  : "Translate this recording"}
              </button>
            </div>
          )}

          {audioFile && (
            <p className="status-text">
              Selected file: <strong>{audioFile.name}</strong>
            </p>
          )}

          <p className="status-text">
            Current status: <strong>{status}</strong>
            {loading && (
              <span>
                {" "}
                ({formatTime(elapsed)}
                {progress > 0 ? ` · ${progress}%` : ""})
              </span>
            )}
          </p>

          {error && (
            <p className="status-text">{error}</p>
          )}
        </section>

        <section className="subtitle-card card">
          <div className="subtitle-head">
            <span className={isLive ? "live-dot on" : "live-dot"}></span>
            <strong>{isLive ? "LIVE" : "READY"}</strong>
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
                <div className="subtitle-roman">
                  {latestEnglish.roman}
                </div>
              )}
            </div>
          )}
        </section>

        <section className="conversation card">
          <div className="section-heading">
            <div>
              <h3>Conversation</h3>
              <p className="muted">
                Each row shows what was said and its English meaning
              </p>
            </div>
            <span className="live-label">
              ● {rows.length} {rows.length === 1 ? "LINE" : "LINES"}
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
                <h4>The conversation will appear here</h4>
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
        </section>

        <section className="features">
          <div className="feature card">
            <span className="feature-icon">01</span>
            <h3>Local processing</h3>
            <p>Designed for approved internal infrastructure.</p>
          </div>

          <div className="feature card">
            <span className="feature-icon">02</span>
            <h3>Multilingual translation</h3>
            <p>Hindi, Bengali and Korean.</p>
          </div>

          <div className="feature card">
            <span className="feature-icon">03</span>
            <h3>Meeting transcript</h3>
            <p>Display recognised speech alongside English translation.</p>
          </div>
        </section>

        <footer>
          SAIL Meeting Translator · Development Prototype
        </footer>
      </main>
    </div>
  );
}

export default App;