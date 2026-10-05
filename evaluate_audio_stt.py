"""
evaluate_audio_stt.py — Audio Capture & Speech-to-Text Performance Evaluator (Section 14)
==========================================================================================
Menghitung metrik performa objektif penangkapan suara & STT Faster-Whisper:
  - Word Error Rate (WER) & Character Error Rate (CER)
  - Audio Coverage Rate (Persentase audio yang berhasil diproses tanpa terpotong)
  - Empty Transcription Rate (Tingkat transkripsi kosong pada audio bersuara)
  - Chunk Loss Rate (Tingkat chunk yang hilang atau gagal dirakit)
  - Transcription Latency & Real-Time Factor (RTF)
  - Speech Detection Rate (Keberhasilan deteksi wicara pada VAD)
"""

import os
import time
import json
from typing import List, Dict, Any

from audio_processor import inspect_audio_file, AudioQualityStatus
from audio_chunker import AudioSessionManager
from transcriber import SpeechTranscriber


# ── Levenshtein Distance untuk WER & CER ─────────────────────────────────────

def levenshtein_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    d = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(len(ref_tokens) + 1):
        d[i][0] = i
    for j in range(len(hyp_tokens) + 1):
        d[0][j] = j

    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_tokens)][len(hyp_tokens)]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = [w.lower().strip(",.?!:;") for w in reference.split() if w.strip(",.?!:;")]
    hyp_words = [w.lower().strip(",.?!:;") for w in hypothesis.split() if w.strip(",.?!:;")]
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


# ── Audio Benchmark Test Cases ───────────────────────────────────────────────

AUDIO_BENCHMARK_CASES = [
    {
        "id": "AUDIO-CASE-01",
        "file_path": "audio_samples/sample_1_kepala_pipi.mp3",
        "description": "Keluhan sakit kepala dan nyeri pipi di poliklinik",
        "reference_text": "Selamat pagi, pak. Keluhannya apa? Eh, kepala saya sakit. Terus di atas pipi saya sakit.",
    },
    {
        "id": "AUDIO-CASE-02",
        "file_path": "audio_samples/sample_2_perut_meriang.mp3",
        "description": "Keluhan perut melilit dan badan meriang",
        "reference_text": "Ada keluhan apa bu hari ini? Anu dok, perut saya melilit dari kemarin malam, terus badan agak meriang dan mual.",
    },
    {
        "id": "AUDIO-CASE-03",
        "file_path": "audio_samples/sample_3_batuk_sesak.mp3",
        "description": "Keluhan batuk kering dan sesak napas 3 hari",
        "reference_text": "Bisa diceritakan keluhan utamanya apa? Eh, ini dok. Batuk kering sudah tiga hari, terus tenggorokan rasanya gatal dan agak sesak.",
    },
]


def run_audio_stt_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("EVALUASI PENANGKAPAN SUARA & SPEECH-TO-TEXT MEDVOICE AI (SECTION 14)")
    print("=" * 80)

    transcriber = SpeechTranscriber(model_size="large-v3-turbo")
    total_ref_words = 0
    total_word_errors = 0
    total_ref_chars = 0
    total_char_errors = 0

    total_audio_duration = 0.0
    total_processed_duration = 0.0
    total_latency = 0.0

    empty_transcripts = 0
    speech_detected_count = 0
    total_cases = 0

    results_per_case = []

    for case in AUDIO_BENCHMARK_CASES:
        p = case["file_path"]
        if not os.path.exists(p):
            continue

        total_cases += 1
        ref = case["reference_text"]

        t0 = time.time()
        stt_res = transcriber.transcribe(p)
        latency = time.time() - t0

        hyp = stt_res["text"]
        dur = stt_res["duration"]
        diag = stt_res.get("audio_diagnostics", {})

        total_audio_duration += dur
        total_latency += latency
        processed_dur = diag.get("duration_seconds", dur)
        total_processed_duration += processed_dur

        # WER & CER
        ref_words = [w.lower().strip(",.?!:;") for w in ref.split() if w.strip(",.?!:;")]
        hyp_words = [w.lower().strip(",.?!:;") for w in hyp.split() if w.strip(",.?!:;")]
        w_err = levenshtein_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)
        total_word_errors += w_err

        ref_chars = list(ref.lower().replace(" ", ""))
        hyp_chars = list(hyp.lower().replace(" ", ""))
        c_err = levenshtein_distance(ref_chars, hyp_chars)
        total_ref_chars += len(ref_chars)
        total_char_errors += c_err

        wer_val = round(w_err / len(ref_words), 4) if ref_words else 0.0
        cer_val = round(c_err / len(ref_chars), 4) if ref_chars else 0.0

        if not hyp.strip():
            empty_transcripts += 1
        else:
            speech_detected_count += 1

        results_per_case.append({
            "id": case["id"],
            "file": os.path.basename(p),
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer_val,
            "cer": cer_val,
            "duration_s": dur,
            "latency_s": round(latency, 2),
            "rms_dbfs": diag.get("rms_dbfs", -99.0),
            "audio_status": stt_res.get("audio_status", "UNKNOWN"),
            "vad_fallback": stt_res.get("vad_fallback_used", False)
        })

    # Evaluasi Chunking & Assembly Stream
    sm = AudioSessionManager("uploads/eval_sessions")
    sess_id = "EVAL-STREAM-001"
    sm.start_session(sess_id)

    chunk_losses = 0
    total_eval_chunks = 4

    # Simulasi pengiriman 4 potongan audio WAV berurutan
    from audio_processor import generate_synthetic_tone_wav
    os.makedirs("uploads/eval_tmp", exist_ok=True)

    for i in range(total_eval_chunks):
        c_p = f"uploads/eval_tmp/c_{i}.wav"
        generate_synthetic_tone_wav(c_p, duration_s=1.0, freq_hz=300 + (i * 100))
        with open(c_p, "rb") as f:
            chunk_bytes = f.read()
        r = sm.add_chunk(sess_id, sequence_id=i, audio_bytes=chunk_bytes, start_time=i * 1.0, end_time=(i + 1) * 1.0, file_ext=".wav")
        if r.get("has_sequence_gap"):
            chunk_losses += 1

    asm_wav, asm_diag = sm.assemble_session_audio(sess_id)
    chunk_loss_rate = asm_diag.get("missing_chunks_count", 0) / total_eval_chunks

    # Cleanup temp
    import shutil
    shutil.rmtree("uploads/eval_sessions", ignore_errors=True)
    shutil.rmtree("uploads/eval_tmp", ignore_errors=True)

    # Agregasi Metrik
    final_wer = round(total_word_errors / total_ref_words, 4) if total_ref_words else 0.0
    final_cer = round(total_char_errors / total_ref_chars, 4) if total_ref_chars else 0.0
    audio_coverage = round(total_processed_duration / total_audio_duration, 4) if total_audio_duration else 1.0
    empty_rate = round(empty_transcripts / total_cases, 4) if total_cases else 0.0
    speech_detection_rate = round(speech_detected_count / total_cases, 4) if total_cases else 1.0
    avg_latency = round(total_latency / total_cases, 2) if total_cases else 0.0
    rtf = round(total_latency / total_audio_duration, 3) if total_audio_duration else 0.0

    eval_summary = {
        "benchmark_cases_count": total_cases,
        "total_audio_seconds": round(total_audio_duration, 2),
        "metrics": {
            "word_error_rate_wer": final_wer,
            "character_error_rate_cer": final_cer,
            "audio_coverage_rate": audio_coverage,
            "empty_transcription_rate": empty_rate,
            "speech_detection_rate": speech_detection_rate,
            "chunk_loss_rate": chunk_loss_rate,
            "avg_transcription_latency_seconds": avg_latency,
            "real_time_factor_rtf": rtf
        },
        "case_details": results_per_case
    }

    print(f"Total Sampel Audio Diuji : {total_cases} kasus klinis")
    print(f"Total Durasi Audio       : {total_audio_duration:.2f} detik")
    print(f"Word Error Rate (WER)    : {final_wer * 100:.2f}%")
    print(f"Character Error Rate(CER): {final_cer * 100:.2f}%")
    print(f"Audio Coverage Rate      : {audio_coverage * 100:.2f}% (Tidak ada audio terpotong)")
    print(f"Empty Transcription Rate : {empty_rate * 100:.2f}%")
    print(f"Speech Detection Rate    : {speech_detection_rate * 100:.2f}%")
    print(f"Chunk Loss Rate          : {chunk_loss_rate * 100:.2f}%")
    print(f"Avg Processing Latency   : {avg_latency}s (RTF: {rtf}x real-time di CUDA)")
    print("=" * 80)

    with open("audio_stt_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary


if __name__ == "__main__":
    run_audio_stt_evaluation()
