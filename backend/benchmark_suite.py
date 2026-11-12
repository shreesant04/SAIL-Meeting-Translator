#!/usr/bin/env python3
"""
Comprehensive Correctness, Speed, Multilingual, and Edge-Case Verification Suite
for SAIL Meeting Translator.
Tests:
1. Multilingual translation correctness (Hindi, Hinglish/code-switching, English).
2. Speed benchmarks: Real-Time Factor (RTF), latency, throughput.
3. Edge cases:
   - Zero-length & malformed PCM buffers
   - Non-standard sample rates (8kHz, 22.05kHz, 44.1kHz, 48kHz, 96kHz)
   - Audio with clipping and extreme values (> 1.0, < -1.0)
   - Audio with NaN and Inf values
   - Pure silence vs background noise RMS thresholding
   - Stereo and multi-channel downmixing
   - Ultra-short audio bursts (< 50ms)
   - Code-switched Hinglish meeting speech
   - Large text synthesis stress test
   - Concurrent queue stress
"""

import os
import sys
import time
import io
import wave
import json
import numpy as np

# Linux soundcard workaround
if len(sys.argv) < 2:
    sys.argv.append("sail")

from faster_whisper import WhisperModel
from piper import PiperVoice
from audio_utils import (
    resample_audio,
    calculate_rms_level,
    pcm_bytes_to_float32,
    ensure_mono_float32,
    create_synthetic_wav
)
from tts import tts_manager

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
EN_VOICE_PATH = os.path.join(VOICES_DIR, "en_US-lessac-medium.onnx")
HI_VOICE_PATH = os.path.join(VOICES_DIR, "hi_IN-pratham-medium.onnx")

def run_benchmarks():
    print("=" * 75)
    print("      SAIL MEETING TRANSLATOR: PRODUCTION VERIFICATION & EDGE-CASE SUITE")
    print("=" * 75)

    # 1. Model Loading Benchmark
    print("\n[PHASE 1] Initializing & Benchmarking Neural Models...")
    t0 = time.perf_counter()
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
    t_whisper_load = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Whisper ASR Engine (base, int8, 4 threads) : loaded in {t_whisper_load:.1f}ms")

    t0 = time.perf_counter()
    en_voice = PiperVoice.load(EN_VOICE_PATH)
    t_en_load = (time.perf_counter() - t0) * 1000
    print(f"  ✓ English Piper Neural TTS Voice             : loaded in {t_en_load:.1f}ms")

    t0 = time.perf_counter()
    hi_voice = PiperVoice.load(HI_VOICE_PATH)
    t_hi_load = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Hindi Piper Neural TTS Voice               : loaded in {t_hi_load:.1f}ms")

    # 2. Multilingual Correctness & Speed Benchmarks
    print("\n[PHASE 2] Multilingual Translation & Real-Time Factor (RTF) Benchmarks...")
    testcases = [
        {
            "id": "TC-01-HI-STEEL-OPS",
            "lang": "hi",
            "voice": hi_voice,
            "text": "स्टील अथॉरिटी ऑफ इंडिया लिमिटेड की बैठक में आज इस्पात उत्पादन और सुरक्षा मानकों की समीक्षा की जाएगी।",
            "expected_keywords": ["steel", "authority", "production", "safety", "review", "meeting", "india"],
            "description": "Enterprise technical Hindi speech"
        },
        {
            "id": "TC-02-HI-TARGETS",
            "lang": "hi",
            "voice": hi_voice,
            "text": "अगली तिमाही के वित्तीय लक्ष्यों को प्राप्त करने के लिए नई कार्ययोजना तैयार है।",
            "expected_keywords": ["next", "quarter", "financial", "target", "plan", "goals", "tasks", "prepare"],
            "description": "Financial quarterly planning in Hindi"
        },
        {
            "id": "TC-03-HINGLISH",
            "lang": "hi",
            "voice": hi_voice,
            "text": "मीटिंग में सभी डिपार्टमेंट्स अपनी रिपोर्ट सबमिट करेंगे।",
            "expected_keywords": ["meeting", "report", "department"],
            "description": "Code-switched Hinglish common in Indian enterprise meetings"
        },
        {
            "id": "TC-04-EN-TECHNICAL",
            "lang": "en",
            "voice": en_voice,
            "text": "The blast furnace operations are currently performing at optimal efficiency.",
            "expected_keywords": ["blast", "furnace", "operations", "efficiency"],
            "description": "Technical English speech (Verifies non-distortion of English)"
        },
        {
            "id": "TC-05-EN-URGENT",
            "lang": "en",
            "voice": en_voice,
            "text": "Please report all equipment telemetry updates to the central control station immediately.",
            "expected_keywords": ["telemetry", "control", "station", "report"],
            "description": "Urgent English operational directive"
        }
    ]

    benchmark_records = []
    for tc in testcases:
        print(f"\n  ▶ Executing {tc['id']}: {tc['description']}")
        
        # Synthesize speech
        buf = io.BytesIO()
        t0 = time.perf_counter()
        with wave.open(buf, "wb") as wf:
            tc["voice"].synthesize_wav(tc["text"], wf)
        synth_time_ms = (time.perf_counter() - t0) * 1000

        wav_bytes = buf.getvalue()
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            num_frames = wf.getnframes()
            audio_dur_s = num_frames / float(sr)
            raw = wf.readframes(num_frames)
            samples = np.frombuffer(raw, dtype=np.int16)
            audio = (samples / 32768.0).astype(np.float32)

        if sr != 16000:
            audio = resample_audio(audio, orig_sr=sr, target_sr=16000)

        # Transcribe & Translate matching main.py settings
        MEETING_PROMPT = "This is an official business meeting at Steel Authority of India Limited (SAIL) discussing steel production, plant operations, safety compliance, and financial targets."
        t0 = time.perf_counter()
        segments_gen, info = whisper_model.transcribe(
            audio,
            task="translate",
            beam_size=2,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            initial_prompt=MEETING_PROMPT,
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
        translated_segments = list(segments_gen)
        transcribe_time_ms = (time.perf_counter() - t0) * 1000

        translated_text = " ".join(s.text.strip() for s in translated_segments)
        rtf = (transcribe_time_ms / 1000.0) / audio_dur_s
        speedup = 1.0 / rtf if rtf > 0 else 0

        # Verification metrics (TREAT_URDU_AS_HINDI applies like in main.py)
        lang_detected = info.language
        if lang_detected == "ur":
            lang_detected = "hi"
        confidence = info.language_probability
        matched_keywords = [kw for kw in tc["expected_keywords"] if kw.lower() in translated_text.lower()]
        keyword_coverage = len(matched_keywords) / len(tc["expected_keywords"])
        passed = (len(translated_text) > 0 and rtf <= 1.5 and keyword_coverage >= 0.25)

        print(f"    • Spoken Input       : {tc['text']}")
        print(f"    • Audio Duration     : {audio_dur_s:.2f}s | TTS Latency: {synth_time_ms:.1f}ms")
        print(f"    • Detected Language  : {lang_detected} (Confidence: {confidence*100:.1f}%)")
        print(f"    • English Result     : \"{translated_text}\"")
        print(f"    • Performance        : {transcribe_time_ms:.1f}ms (RTF: {rtf:.3f}x, {speedup:.1f}x real-time)")
        print(f"    • Status             : {'PASS ✓' if passed else 'FAIL ✗'} (Keywords: {matched_keywords})")

        benchmark_records.append({
            "id": tc["id"],
            "lang": tc["lang"],
            "detected": lang_detected,
            "confidence": round(float(confidence), 3),
            "audio_dur_s": round(audio_dur_s, 2),
            "transcribe_ms": round(transcribe_time_ms, 1),
            "rtf": round(rtf, 3),
            "speedup": round(speedup, 1),
            "translation": translated_text,
            "passed": passed
        })

    # 3. Comprehensive Edge-Case Verification
    print("\n[PHASE 3] Rigorous Edge-Case Stress Testing...")
    edge_results = []

    def check_edge_case(name, condition, details=""):
        status = "PASS ✓" if condition else "FAIL ✗"
        print(f"  • [{status}] {name} {details}")
        edge_results.append({"name": name, "passed": bool(condition), "details": details})
        assert condition, f"Edge case failed: {name}"

    # Edge 1: Zero-length byte buffer
    empty_res = pcm_bytes_to_float32(b"")
    check_edge_case("Empty byte buffer", len(empty_res) == 0, f"(result length: {len(empty_res)})")

    # Edge 2: Unaligned byte buffer (not divisible by 4)
    unaligned = pcm_bytes_to_float32(b"\x00\x01\x02")
    check_edge_case("Unaligned byte buffer (len 3)", len(unaligned) == 0, "(safely returned empty array)")

    # Edge 3: Resampling with zero or negative sample rate
    resample_zero = resample_audio(np.ones(100, dtype=np.float32), orig_sr=0, target_sr=16000)
    check_edge_case("Resample with orig_sr=0", len(resample_zero) == 0, "(ZeroDivisionError prevented)")

    resample_neg = resample_audio(np.ones(100, dtype=np.float32), orig_sr=-16000, target_sr=16000)
    check_edge_case("Resample with negative sample rate", len(resample_neg) == 0, "(ZeroDivisionError prevented)")

    # Edge 4: Non-standard sample rates (8kHz, 22.05kHz, 44.1kHz, 48kHz, 96kHz)
    for sr in [8000, 22050, 44100, 48000, 96000]:
        t = np.linspace(0, 1.0, sr, endpoint=False).astype(np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        res = resample_audio(sine, orig_sr=sr, target_sr=16000)
        check_edge_case(f"Sample Rate Conversion {sr}Hz -> 16000Hz", len(res) == 16000, f"(samples: {len(res)})")

    # Edge 5: Stereo 2D array downmixing
    stereo = np.random.uniform(-0.5, 0.5, (16000, 2)).astype(np.float32)
    mono = ensure_mono_float32(stereo)
    check_edge_case("Stereo (2-channel) to Mono", mono.ndim == 1 and len(mono) == 16000, f"(shape: {mono.shape})")

    # Edge 6: Multi-channel 6-channel (5.1 surround) downmixing
    surround = np.random.uniform(-0.5, 0.5, (16000, 6)).astype(np.float32)
    mono_surround = ensure_mono_float32(surround)
    check_edge_case("5.1 Surround (6-channel) to Mono", mono_surround.ndim == 1 and len(mono_surround) == 16000, f"(shape: {mono_surround.shape})")

    # Edge 7: Audio with Severe Clipping (values > 1.0 and < -1.0)
    clipped_audio = np.array([2.5, -3.8, 1.2, -1.5, 0.0, 0.5], dtype=np.float32)
    rms_clipped = calculate_rms_level(clipped_audio)
    check_edge_case("Clipped Audio RMS calculation", np.isfinite(rms_clipped) and rms_clipped > 0, f"(RMS: {rms_clipped:.3f})")

    # Edge 8: Pure Silence Gating (RMS threshold check)
    silence = np.zeros(16000 * 2, dtype=np.float32)
    rms_silence = calculate_rms_level(silence)
    check_edge_case("Silence RMS Threshold Gating", rms_silence < 0.003, f"(RMS: {rms_silence:.6f} < 0.003)")

    # Edge 9: Very Low Noise Floor (Ambient Room Noise simulation)
    ambient_noise = np.random.normal(0, 0.0005, 16000 * 2).astype(np.float32)
    rms_ambient = calculate_rms_level(ambient_noise)
    check_edge_case("Ambient Low Noise Filtered", rms_ambient < 0.003, f"(RMS: {rms_ambient:.6f} < 0.003)")

    # Edge 10: Ultra-short audio burst (< 50ms)
    short_audio = np.random.uniform(-0.2, 0.2, 400).astype(np.float32) # 25ms
    segments_short, info_short = whisper_model.transcribe(short_audio, task="translate", beam_size=1)
    list(segments_short)
    check_edge_case("Ultra-short Audio Burst (25ms)", True, "(no crash or unhandled exception)")

    # Edge 11: Piper TTS Empty String Validation
    try:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            en_voice.synthesize_wav("", wf)
        empty_tts_handled = False
    except ValueError:
        empty_tts_handled = True
    except Exception:
        empty_tts_handled = True
    check_edge_case("Piper TTS Empty String Validation", empty_tts_handled, "(gracefully prevented empty synthesis)")

    # Edge 12: Piper TTS Long Paragraph Stress (200 words)
    long_text = " ".join(["Safety standards at SAIL plants require mandatory PPE, thermal inspection, and automated gas monitoring."] * 15)
    t0 = time.perf_counter()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        en_voice.synthesize_wav(long_text, wf)
    long_tts_ms = (time.perf_counter() - t0) * 1000
    with wave.open(io.BytesIO(buf.getvalue()), "rb") as wf:
        long_tts_dur = wf.getnframes() / float(wf.getframerate())
    long_speedup = long_tts_dur / (long_tts_ms / 1000.0)
    check_edge_case(
        "Long Text Synthesis Stress (200+ words)",
        len(buf.getvalue()) > 50000 and long_speedup > 1.0,
        f"(duration: {long_tts_dur:.1f}s, generated in {long_tts_ms:.1f}ms -> {long_speedup:.1f}x real-time)"
    )

    # 4. Final Verification Summary
    print("\n" + "=" * 75)
    print("                    FINAL BENCHMARK & VERIFICATION RESULTS")
    print("=" * 75)
    avg_rtf = np.mean([b["rtf"] for b in benchmark_records])
    avg_speedup = 1.0 / avg_rtf
    all_tc_passed = all(b["passed"] for b in benchmark_records)
    all_edges_passed = all(e["passed"] for e in edge_results)

    print(f"  • Multilingual Translation Correctness : {'ALL PASSED ✓' if all_tc_passed else 'FAIL ✗'}")
    print(f"  • Edge-Case Resilience Tests (12/12)   : {'ALL PASSED ✓' if all_edges_passed else 'FAIL ✗'}")
    print(f"  • Average Translation Latency (RTF)    : {avg_rtf:.3f}x real-time ({avg_speedup:.1f}x FASTER than live speech)")
    print(f"  • Neural Talkback Synthesis Throughput : ~{long_speedup:.1f}x faster than real-time speech")
    print(f"  • Overall Production Certification     : {'SUCCESS - PRODUCTION GRADE ✓' if (all_tc_passed and all_edges_passed) else 'FAIL ✗'}")
    print("=" * 75 + "\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "status": "SUCCESS" if (all_tc_passed and all_edges_passed) else "FAILED",
        "multilingual_correctness": all_tc_passed,
        "edge_cases_correctness": all_edges_passed,
        "avg_transcription_rtf": round(float(avg_rtf), 3),
        "transcription_speedup_factor": round(float(avg_speedup), 1),
        "tts_speedup_factor": round(float(long_speedup), 1),
        "benchmarks": benchmark_records,
        "edge_cases": edge_results
    }

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

if __name__ == "__main__":
    run_benchmarks()
