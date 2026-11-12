import { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

// Dynamic API URL: handles localhost, GitHub Codespaces port subdomains, and Vite proxies
const getApiBase = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  const hostname = window.location.hostname || "127.0.0.1";
  const protocol = window.location.protocol || "http:";

  // Codespaces subdomain forwarding: e.g. *-5173.app.github.dev -> *-8000.app.github.dev
  if (hostname.includes(".app.github.dev") && hostname.includes("-5173.")) {
    return `${protocol}//${hostname.replace("-5173.", "-8000.")}`;
  }

  // Localhost development on Vite port 5173 -> point to FastAPI backend on 8000
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://127.0.0.1:8000";
  }

  // Fallback to origin
  return window.location.origin || `${protocol}//${hostname}:8000`;
};

const API = getApiBase();

const getWsBase = (apiBase) => {
  if (apiBase.startsWith("https://")) {
    return apiBase.replace(/^https:\/\//, "wss://");
  }
  if (apiBase.startsWith("http://")) {
    return apiBase.replace(/^http:\/\//, "ws://");
  }
  const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProto}//${window.location.host}`;
};

const WS_BASE = getWsBase(API);

// Target sample rate for Whisper
const TARGET_SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 5;

// Downsample float32 PCM buffer in browser
function downsampleAudio(buffer, inputSampleRate, targetSampleRate = TARGET_SAMPLE_RATE) {
  if (inputSampleRate === targetSampleRate) return buffer;
  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function App() {
  const [status, setStatus] = useState("Ready");
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [audioFile, setAudioFile] = useState(null);
  const [translation, setTranslation] = useState([]);
  const [originalTranscript, setOriginalTranscript] = useState([]);
  const [detectedLanguage, setDetectedLanguage] = useState("");
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState("");

  const [audioSource, setAudioSource] = useState("screen"); // "screen" | "mic"
  const [autoTalkback, setAutoTalkback] = useState(false);
  const [recordEnabled, setRecordEnabled] = useState(false);
  const [recordingUrl, setRecordingUrl] = useState("");
  const [recordingName, setRecordingName] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);

  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const contextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const gainRef = useRef(null);

  const audioBufferRef = useRef([]);
  const sampleCountRef = useRef(0);
  const stoppingRef = useRef(false);

  const recorderRef = useRef(null);
  const recordChunksRef = useRef([]);
  const recordEnabledRef = useRef(false);
  const recordingUrlRef = useRef("");
  const autoTalkbackRef = useRef(false);

  // Talkback playback queue
  const talkbackQueueRef = useRef([]);
  const isPlayingTalkbackRef = useRef(false);
  const processQueueRef = useRef(null);

  useEffect(() => {
    processQueueRef.current = async () => {
      if (isPlayingTalkbackRef.current || talkbackQueueRef.current.length === 0) {
        return;
      }

      isPlayingTalkbackRef.current = true;
      const nextText = talkbackQueueRef.current.shift();

      try {
        let audioBlob = null;
        try {
          const response = await fetch(`${API}/synthesize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: nextText })
          });
          if (response.ok) {
            audioBlob = await response.blob();
          }
        } catch (err) {
          console.warn("Backend TTS synthesis fetch failed:", err);
        }

        if (audioBlob) {
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          await new Promise(resolve => {
            audio.onended = () => {
              URL.revokeObjectURL(audioUrl);
              resolve();
            };
            audio.onerror = () => {
              URL.revokeObjectURL(audioUrl);
              resolve();
            };
            audio.play().catch(resolve);
          });
        } else if ("speechSynthesis" in window) {
          await new Promise(resolve => {
            const utterance = new SpeechSynthesisUtterance(nextText);
            utterance.lang = "en-US";
            utterance.rate = 1.0;
            utterance.onend = resolve;
            utterance.onerror = resolve;
            window.speechSynthesis.speak(utterance);
          });
        }
      } catch (err) {
        console.error("Talkback playback error:", err);
      } finally {
        isPlayingTalkbackRef.current = false;
        if (processQueueRef.current) {
          setTimeout(processQueueRef.current, 40);
        }
      }
    };
  }, []);

  const playTalkback = useCallback((text) => {
    if (!text || !text.trim()) return;
    talkbackQueueRef.current.push(text.trim());
    if (processQueueRef.current) {
      processQueueRef.current();
    }
  }, []);

  // Keep ref synchronized with state
  useEffect(() => {
    recordEnabledRef.current = recordEnabled;
  }, [recordEnabled]);

  useEffect(() => {
    autoTalkbackRef.current = autoTalkback;
  }, [autoTalkback]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    recorderRef.current = null;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch (err) {
        console.error("Recording stop error:", err);
      }
    }
  }, []);

  const cleanupAudio = useCallback(() => {
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
    setAudioLevel(0);
  }, [stopRecording]);

  const stopTranslation = useCallback(() => {
    stoppingRef.current = true;
    setIsStarting(false);

    // Empty speech talkback queue
    talkbackQueueRef.current = [];
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

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
  }, [cleanupAudio]);

  // Backend Health Ping
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
  }, [cleanupAudio]);

  const startRecording = useCallback((stream) => {
    try {
      if (recordingUrlRef.current) {
        URL.revokeObjectURL(recordingUrlRef.current);
        recordingUrlRef.current = "";
      }

      setRecordingUrl("");
      setRecordingName("");

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) return;

      const audioOnly = new MediaStream(audioTracks);
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
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
        const pad = num => String(num).padStart(2, "0");
        const name = `meeting-audio-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.webm`;

        setRecordingUrl(url);
        setRecordingName(name);
      };

      recorder.start(1000);
      recorderRef.current = recorder;
    } catch (err) {
      console.error("Recording error:", err);
      setError("Recording could not start: " + err.message);
    }
  }, []);

  // Quick loader for sample meeting audio
  const loadSampleAudio = async (filename) => {
    try {
      setError("");
      setStatus(`Loading sample ${filename}...`);
      const response = await fetch(`${API}/samples/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load ${filename}`);
      }
      const blob = await response.blob();
      const file = new File([blob], filename, { type: "audio/wav" });
      setAudioFile(file);
      setStatus(`Sample ${filename} loaded. Click "Translate File" to process.`);
    } catch (err) {
      setError(`Could not load sample: ${err.message}`);
      setStatus("Ready");
    }
  };

  const startFileTranslation = async () => {
    if (!audioFile) {
      setError("Please select an audio file first.");
      return;
    }

    setLoading(true);
    setError("");
    setTranslation([]);
    setOriginalTranscript([]);
    setDetectedLanguage("");
    setStatus("Translating audio file...");

    try {
      const formData = new FormData();
      formData.append("file", audioFile);

      const response = await fetch(`${API}/translate-audio`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Translation failed.");
      }

      setDetectedLanguage(data.language_name || data.language);
      setOriginalTranscript(data.original_transcript || []);
      setTranslation(data.translation || []);
      setStatus("Translation completed successfully");

      if (autoTalkback && data.translation && data.translation.length > 0) {
        const fullTranslation = data.translation.map(s => s.text).join(" ");
        playTalkback(fullTranslation);
      }
    } catch (err) {
      setError(err.message);
      setStatus("Translation failed");
    } finally {
      setLoading(false);
    }
  };

  const startLiveTranslation = async () => {
    if (isStarting || isLive || socketRef.current) return;

    setIsStarting(true);
    stoppingRef.current = false;
    setError("");
    setTranslation([]);
    setOriginalTranscript([]);
    setDetectedLanguage("");
    setStatus(`Requesting ${audioSource === "screen" ? "meeting screen/system audio" : "microphone"} permission...`);

    try {
      let stream;
      if (audioSource === "mic") {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true
          },
          video: false
        });
      } else {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true,
          systemAudio: "include",
          selfBrowserSurface: "exclude"
        });
      }

      if (stoppingRef.current) {
        stream.getTracks().forEach(track => track.stop());
        setIsStarting(false);
        return;
      }

      streamRef.current = stream;

      if (stream.getAudioTracks().length === 0) {
        throw new Error(
          audioSource === "screen"
            ? "No system audio track captured. Please choose Entire Screen and tick 'Share system audio'."
            : "No microphone audio track found."
        );
      }

      // Attach ended listeners to clean up automatically if user stops screen sharing or disconnects mic
      stream.getTracks().forEach(track => {
        track.addEventListener(
          "ended",
          () => {
            if (socketRef.current === socket) {
              stopTranslation();
            }
          },
          { once: true }
        );
      });

      if (recordEnabledRef.current) {
        startRecording(stream);
      }

      const socket = new WebSocket(`${WS_BASE}/ws/browser-live?sample_rate=16000`);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onmessage = event => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "status") {
            setStatus(data.message);
          } else if (data.type === "error") {
            setError(data.message);
          } else if (data.type === "original_transcript") {
            setDetectedLanguage(data.language_name || data.language);
            setOriginalTranscript(prev => [
              ...prev,
              { start: data.start, end: data.end, text: data.text }
            ]);
          } else if (data.type === "translation") {
            if (data.source_language_name) {
              setDetectedLanguage(data.source_language_name);
            }
            setTranslation(prev => [
              ...prev,
              { start: data.start, end: data.end, text: data.text }
            ]);

            if (autoTalkbackRef.current && data.text) {
              playTalkback(data.text);
            }
          }
        } catch (err) {
          console.error("WebSocket message parsing error:", err);
        }
      };

      socket.onerror = event => {
        console.error("WebSocket connection error:", event);
        setError("WebSocket connection failed. Ensure the SAIL backend is running.");
      };

      socket.onclose = event => {
        if (socketRef.current !== socket) return;
        socketRef.current = null;
        setIsStarting(false);
        cleanupAudio();
        setIsLive(false);

        if (!stoppingRef.current) {
          setStatus("Connection closed");
          if (event.code !== 1000) {
            setError(`Audio connection lost (code: ${event.code}).`);
          }
        }
      };

      socket.onopen = async () => {
        try {
          if (stoppingRef.current) {
            socket.close(1000, "Stopped");
            return;
          }

          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          const context = new AudioCtx();
          contextRef.current = context;
          await context.resume();

          const nativeRate = context.sampleRate;
          const sourceNode = context.createMediaStreamSource(stream);
          sourceRef.current = sourceNode;

          const processor = context.createScriptProcessor(4096, 1, 1);
          processorRef.current = processor;

          const gain = context.createGain();
          gain.gain.value = 0;
          gainRef.current = gain;

          audioBufferRef.current = [];
          sampleCountRef.current = 0;

          processor.onaudioprocess = event => {
            if (socket.readyState !== WebSocket.OPEN || stoppingRef.current) return;

            const inputSamples = event.inputBuffer.getChannelData(0);

            // Compute audio meter level
            let sumSq = 0;
            for (let i = 0; i < inputSamples.length; i++) {
              sumSq += inputSamples[i] * inputSamples[i];
            }
            const rms = Math.sqrt(sumSq / inputSamples.length);
            setAudioLevel(Math.min(100, Math.round(rms * 400)));

            // Resample to 16000 Hz if needed
            const samples16k = downsampleAudio(inputSamples, nativeRate, TARGET_SAMPLE_RATE);
            const copy = new Float32Array(samples16k);

            audioBufferRef.current.push(copy);
            sampleCountRef.current += copy.length;

            const targetSamples = TARGET_SAMPLE_RATE * CHUNK_SECONDS;
            if (sampleCountRef.current >= targetSamples) {
              const combined = new Float32Array(sampleCountRef.current);
              let offset = 0;
              for (const part of audioBufferRef.current) {
                combined.set(part, offset);
                offset += part.length;
              }

              if (socket.readyState === WebSocket.OPEN) {
                socket.send(combined.buffer);
              }

              audioBufferRef.current = [];
              sampleCountRef.current = 0;
            }
          };

          sourceNode.connect(processor);
          processor.connect(gain);
          gain.connect(context.destination);

          setIsStarting(false);
          setIsLive(true);
          setStatus(`Listening to ${audioSource === "screen" ? "system audio" : "microphone"}...`);
        } catch (err) {
          console.error("Audio pipeline initialization error:", err);
          setError("Audio pipeline error: " + err.message);
          setIsStarting(false);
          cleanupAudio();
          socket.close(1011, "Audio setup failed");
          setIsLive(false);
        }
      };

    } catch (err) {
      console.error("Live translation error:", err);
      cleanupAudio();
      setIsStarting(false);
      setError(err.message);
      setStatus("Capture initialization failed");
      setIsLive(false);
    }
  };

  const exportTranscript = (format = "txt") => {
    if (translation.length === 0 && originalTranscript.length === 0) {
      alert("No transcript data available to export.");
      return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    let content = "";
    let mimeType = "text/plain";
    let filename = `SAIL-Meeting-Transcript-${timestamp}.${format}`;

    if (format === "json") {
      mimeType = "application/json";
      content = JSON.stringify(
        {
          timestamp: new Date().toISOString(),
          detected_language: detectedLanguage,
          original_transcript: originalTranscript,
          english_translation: translation
        },
        null,
        2
      );
    } else {
      content = `SAIL MEETING TRANSLATOR - TRANSCRIPT\n`;
      content += `Date/Time: ${new Date().toLocaleString()}\n`;
      content += `Detected Language: ${detectedLanguage || "Unknown"}\n`;
      content += `-------------------------------------------------------\n\n`;

      const maxLen = Math.max(translation.length, originalTranscript.length);
      for (let i = 0; i < maxLen; i++) {
        const orig = originalTranscript[i];
        const trans = translation[i];
        if (orig) {
          content += `[${orig.start}s - ${orig.end}s] (${detectedLanguage || "Original"}): ${orig.text}\n`;
        }
        if (trans) {
          content += `[${trans.start}s - ${trans.end}s] (English Translation): ${trans.text}\n`;
        }
        content += `\n`;
      }
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearTranscript = () => {
    talkbackQueueRef.current = [];
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setTranslation([]);
    setOriginalTranscript([]);
    setDetectedLanguage("");
    setStatus("Ready");
  };

  // Filtered segments
  const filteredOriginal = originalTranscript.filter(seg =>
    searchQuery ? seg.text.toLowerCase().includes(searchQuery.toLowerCase()) : true
  );

  const filteredTranslation = translation.filter(seg =>
    searchQuery ? seg.text.toLowerCase().includes(searchQuery.toLowerCase()) : true
  );

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-logo">S</div>
          <div>
            <h2>SAIL Meeting Translator</h2>
            <p>Internal Multilingual Meeting Assistance System</p>
          </div>
        </div>

        <div className="connection-info">
          <div className="connection">
            <span
              className="status-dot"
              style={{
                background:
                  backendStatus === "Connected"
                    ? "#087650"
                    : backendStatus === "Checking..."
                    ? "#e5a52d"
                    : "#d9383a"
              }}
            ></span>
            Backend: <strong>{backendStatus}</strong>
          </div>
          {isLive && (
            <div className="audio-meter-container" title="Audio Activity">
              <span className="meter-label">Audio:</span>
              <div className="meter-bar">
                <div
                  className="meter-fill"
                  style={{ width: `${audioLevel}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="main">
        <section className="welcome">
          <span className="eyebrow">STEEL AUTHORITY OF INDIA LIMITED</span>
          <h1>Multilingual Meeting Translator</h1>
          <p>
            Real-time speech recognition, live translation into English, and neural talkback
            synthesizer for internal committee and delegate meetings.
          </p>
        </section>

        <section className="controls card">
          <h3>Meeting Controls</h3>
          <p className="muted">
            Translate spoken Hindi, Bengali, Korean, and other supported languages into English with automated speaker language detection.
          </p>

          <div className="control-row">
            <div className="source-toggle">
              <label htmlFor="audio-source">Audio Input:</label>
              <select
                id="audio-source"
                value={audioSource}
                onChange={e => setAudioSource(e.target.value)}
                disabled={isLive || isStarting}
              >
                <option value="screen">System Audio (Screen / Video Call)</option>
                <option value="mic">Microphone (Headset / Room)</option>
              </select>
            </div>

            <input
              type="file"
              accept="audio/*,.wav,.mp3,.webm,.ogg,.m4a,.flac"
              onChange={event => {
                setAudioFile(event.target.files[0] || null);
                setError("");
              }}
              disabled={isLive || isStarting}
            />

            <button
              className="start-button"
              onClick={startFileTranslation}
              disabled={loading || isLive || isStarting || !audioFile}
            >
              {loading ? "Translating..." : "Translate File"}
            </button>

            <button
              className="start-button"
              onClick={startLiveTranslation}
              disabled={isLive || loading || isStarting}
            >
              {isStarting
                ? "Connecting..."
                : isLive
                ? "Listening..."
                : "Start Live Translation"}
            </button>

            <button
              className="stop-button"
              onClick={stopTranslation}
              disabled={!isLive && !isStarting}
            >
              Stop
            </button>
          </div>

          {/* Quick sample audio testing buttons */}
          <div className="sample-quickload-row">
            <span className="quickload-label">Quick Test Samples:</span>
            <button
              className="sample-btn"
              type="button"
              disabled={isLive || isStarting || loading}
              onClick={() => loadSampleAudio("sample_hindi.wav")}
            >
              🇮🇳 Load Hindi Meeting Sample
            </button>
            <button
              className="sample-btn"
              type="button"
              disabled={isLive || isStarting || loading}
              onClick={() => loadSampleAudio("sample_english.wav")}
            >
              🇬🇧 Load English Meeting Sample
            </button>
          </div>

          <div className="options-row">
            <label className="checkbox-option">
              <input
                type="checkbox"
                checked={autoTalkback}
                onChange={e => setAutoTalkback(e.target.checked)}
              />
              🔊 <strong>Auto-Talkback</strong> (Read English translations aloud sequentially)
            </label>

            <label className="checkbox-option">
              <input
                type="checkbox"
                checked={recordEnabled}
                disabled={isLive || isStarting}
                onChange={e => setRecordEnabled(e.target.checked)}
              />
              ⏺ Also record meeting audio (saved locally)
            </label>
          </div>

          {recordingUrl && (
            <div className="recording-box">
              <p className="status-text">
                Meeting recording saved: <strong>{recordingName}</strong>
              </p>
              <audio controls src={recordingUrl}></audio>
              <p>
                <a
                  className="download-link"
                  href={recordingUrl}
                  download={recordingName}
                >
                  ⬇ Download Recording File (.webm)
                </a>
              </p>
            </div>
          )}

          {audioFile && (
            <p className="status-text">
              Selected file: <strong>{audioFile.name}</strong> ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}

          <div className="status-banner">
            <span>Status: <strong>{status}</strong></span>
            {detectedLanguage && (
              <span className="lang-badge">
                Language: <strong>{detectedLanguage}</strong>
              </span>
            )}
          </div>

          {error && <div className="error-alert">⚠ {error}</div>}
        </section>

        {/* Toolbar: Search, Clear, Export */}
        <div className="transcript-toolbar">
          <div className="search-box">
            <input
              type="text"
              placeholder="Search transcript text..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                className="clear-search-btn"
                onClick={() => setSearchQuery("")}
              >
                ✕
              </button>
            )}
          </div>

          <div className="toolbar-actions">
            <button
              className="action-btn"
              onClick={() => exportTranscript("txt")}
              disabled={translation.length === 0 && originalTranscript.length === 0}
            >
              ⬇ Export TXT
            </button>
            <button
              className="action-btn"
              onClick={() => exportTranscript("json")}
              disabled={translation.length === 0 && originalTranscript.length === 0}
            >
              ⬇ Export JSON
            </button>
            <button
              className="action-btn danger"
              onClick={clearTranscript}
              disabled={translation.length === 0 && originalTranscript.length === 0}
            >
              🗑 Clear
            </button>
          </div>
        </div>

        <div className="transcript-grid">
          {/* Original Transcript Column */}
          <section className="transcript card">
            <div className="section-heading">
              <div>
                <h3>Original Transcript</h3>
                <p className="muted">
                  Language: <strong>{detectedLanguage || "Awaiting speech..."}</strong>
                </p>
              </div>
              <span className="live-label">● ORIGINAL</span>
            </div>

            <div className="transcript-area">
              {filteredOriginal.length > 0 ? (
                <div className="segment-list">
                  {filteredOriginal.map((segment, index) => (
                    <div key={index} className="segment-item">
                      <span className="timestamp">
                        [{segment.start}s - {segment.end}s]
                      </span>
                      <p className="segment-text">{segment.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">Aa</div>
                  <h4>Original speech will appear here</h4>
                  <p>Speak in Hindi, Bengali, Korean, or any supported language.</p>
                </div>
              )}
            </div>
          </section>

          {/* English Translation Column */}
          <section className="transcript card">
            <div className="section-heading">
              <div>
                <h3>English Translation</h3>
                <p className="muted">Real-time English translation with Talkback</p>
              </div>
              <span className="live-label live-english">● ENGLISH</span>
            </div>

            <div className="transcript-area">
              {filteredTranslation.length > 0 ? (
                <div className="segment-list">
                  {filteredTranslation.map((segment, index) => (
                    <div key={index} className="segment-item translation-item">
                      <div className="segment-header">
                        <span className="timestamp">
                          [{segment.start}s - {segment.end}s]
                        </span>
                        <button
                          className="listen-btn"
                          title="Listen to translation"
                          onClick={() => playTalkback(segment.text)}
                        >
                          🔊 Listen
                        </button>
                      </div>
                      <p className="segment-text">{segment.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">🌐</div>
                  <h4>English translation will appear here</h4>
                  <p>Translations are rendered segment-by-segment as audio arrives.</p>
                </div>
              )}
            </div>
          </section>
        </div>

        <section className="features">
          <div className="feature card">
            <span className="feature-icon">01</span>
            <h3>Local & Secure Processing</h3>
            <p>Designed for SAIL on-premise infrastructure. No external audio leakage.</p>
          </div>

          <div className="feature card">
            <span className="feature-icon">02</span>
            <h3>Multilingual Whisper Engine</h3>
            <p>Accurate transcription and translation across Hindi, Bengali, Korean, and 20+ languages.</p>
          </div>

          <div className="feature card">
            <span className="feature-icon">03</span>
            <h3>Neural Talkback Synthesizer</h3>
            <p>Natural English voice synthesis powered by Piper TTS with non-overlapping sequential playback.</p>
          </div>
        </section>

        <footer>
          SAIL Meeting Translator · Enterprise Internal Release v1.0.0
        </footer>
      </main>
    </div>
  );
}

export default App;
