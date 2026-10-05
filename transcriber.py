"""
transcriber.py — Hospital-Grade Speech-to-Text Engine (Faster-Whisper on CUDA)
==============================================================================
Menangani pengenalan wicara medis Bahasa Indonesia dengan proteksi:
  - Inspeksi kualitas audio (RMS dBFS, hening, volume rendah, clipping)
  - Adaptive Software Gain Boost untuk pasien bersuara lemah / mic berjarak
  - VAD sensitif (threshold 0.30, silence 800ms, padding 400ms)
  - Dual-Pass VAD Fallback (jika VAD menyaring audio non-hening, fallback tanpa filter)
  - Penguncian language='id' dengan prompt klinis Indonesia
  - Diagnostik lengkap status audio vs status transkripsi
"""

import os
import sys
import math
import re
from typing import Dict, Any, List

# Tambahkan direktori DLL CUDA dari virtualenv
venv_dir = os.path.dirname(os.path.dirname(sys.executable))
for lib_name in ["cublas", "cudnn", "cuda_nvrtc"]:
    p_path = os.path.join(venv_dir, "Lib", "site-packages", "nvidia", lib_name, "bin")
    if os.path.exists(p_path):
        try:
            os.add_dll_directory(p_path)
            os.environ["PATH"] = p_path + ";" + os.environ["PATH"]
        except Exception:
            pass

from faster_whisper import WhisperModel
from audio_processor import (
    inspect_audio_file,
    normalize_and_prepare_audio,
    AudioQualityStatus
)

# Prompt Prior Kosakata Medis Indonesia & Istilah Klinis Standar Rumah Sakit Indonesia
INDONESIAN_CLINICAL_PROMPT = (
    "Transkripsi percakapan klinis dokter dan pasien di rumah sakit Indonesia. "
    "Keluhan utama, anamnesa, riwayat penyakit, gejala demam, pusing, batuk pilek, sesak napas, "
    "mual muntah, nyeri perut, lemas, pemeriksaan fisik, tensi darah, suhu tubuh, denyut nadi, "
    "dan riwayat alergi obat."
)

# Frasa halusinasi Whisper & gumaman noise yang disaring secara aman
HALLUCINATIONS_AND_NOISE = {
    "terima kasih.", "terima kasih", "terimakasih.", "terimakasih",
    "terima kasih banyak.", "terima kasih banyak",
    "terima kasih sudah menonton.", "terima kasih telah menonton.",
    "sampai jumpa.", "sampai jumpa",
    "hmm", "hmmm", "hmmmm", "hmm.", "hmmm.", "mmm", "mmmm",
    "uh", "um", "ah", "oh",
    "amin", "aamiin", "amin.", "aamiin.", "amen",
    "sigh", "[sigh]", "(sigh)", "breath", "[breath]", "(breath)",
    "hembusan napas", "hembusan nafas", "tarikan napas", "tarikan nafas",
    "suara napas", "suara nafas", "batuk", "[batuk]", "(batuk)",
}


class SpeechTranscriber:
    def __init__(self, model_size: str = "large-v3-turbo", device: str = None):
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()

        if device is None:
            self.device = "cuda" if cuda_count > 0 else "cpu"
        else:
            self.device = device

        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"[Faster-Whisper] Loading model '{model_size}' pada device '{self.device}' ({self.compute_type})...")

        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            print(f"[Faster-Whisper] Gagal memuat pada {self.device} ({e}), mencoba fallback...")
            try:
                self.model = WhisperModel("small", device=self.device, compute_type=self.compute_type)
            except Exception:
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel("small", device="cpu", compute_type="int8")

        print(f"[Faster-Whisper] Model SIAP! Berjalan di {self.device.upper()} dengan konfigurasi Clinical Indonesia.")

    def _execute_transcribe_pass(self, audio_data, use_vad: bool = True) -> tuple:
        """Eksekusi satu putaran transkripsi Whisper dengan konfigurasi klinis teroptimasi."""
        vad_params = dict(
            threshold=0.30,              # Sensitif untuk ucapan pelan/pasien lemas
            min_speech_duration_ms=120,  # Pertahankan kata pendek bermakna (ya, dok, cek, sakit)
            max_speech_duration_s=float('inf'),
            min_silence_duration_ms=800, # Toleransi jeda berpikir 800ms tanpa memutus kalimat
            speech_pad_ms=400            # Padding 400ms agar konsonan awal/akhir tidak terpotong
        ) if use_vad else None

        segments, info = self.model.transcribe(
            audio_data,
            language="id",
            initial_prompt=INDONESIAN_CLINICAL_PROMPT,
            beam_size=5,
            temperature=0.0,
            repetition_penalty=1.08,
            no_repeat_ngram_size=0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.80,    # Rileks dari 0.60 agar ucapan pelan tidak diabaikan
            log_prob_threshold=-1.5,     # Rileks dari -1.0
            condition_on_previous_text=False,
            vad_filter=use_vad,
            vad_parameters=vad_params
        )
        return list(segments), info

    def transcribe(self, audio_path: str) -> dict:
        """
        Transkripsi audio lengkap dengan:
          1. Pre-flight quality inspection (RMS dBFS, hening, volume rendah)
          2. Adaptive gain boost untuk audio pelan
          3. Dual-pass VAD fallback untuk mencegah ucapan hilang
          4. Pelaporan diagnostik lengkap (Section 8 & 9)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")

        # ── 1. Inspeksi & Persiapan Audio ──
        audio_np, diag = normalize_and_prepare_audio(audio_path, auto_gain=True)

        if not diag["valid"] or audio_np is None:
            return {
                "text": "",
                "detected_language": "id",
                "language_probability": 0.0,
                "duration": diag.get("duration_seconds", 0.0),
                "segments": [],
                "audio_status": diag.get("status", AudioQualityStatus.CORRUPTED),
                "transcription_status": "corrupted",
                "audio_diagnostics": diag,
                "warning": diag.get("warning", "File audio tidak dapat diproses.")
            }

        # Jika audio terdeteksi hening / mic mute
        if diag["status"] == AudioQualityStatus.SILENT:
            return {
                "text": "",
                "detected_language": "id",
                "language_probability": 0.0,
                "duration": diag["duration_seconds"],
                "segments": [],
                "audio_status": AudioQualityStatus.SILENT,
                "transcription_status": "silent_audio",
                "audio_diagnostics": diag,
                "warning": "Audio hening atau mikrofon ter-mute. Tidak ada sinyal suara manusia yang terdeteksi."
            }

        # ── 2. Pass 1: Transkripsi dengan VAD Teroptimasi ──
        vad_fallback_used = False
        raw_segments, info = self._execute_transcribe_pass(audio_np, use_vad=True)

        # ── 3. Pass 2: Fallback Tanpa VAD Jika Pass 1 Kosong Padahal Audio Bersinyal ──
        # Ini mencegah Silero VAD secara keliru membuang suara pasien yang sangat lemah/pelan
        if not raw_segments and diag["rms_dbfs"] > -48.0 and diag["duration_seconds"] >= 0.5:
            raw_segments, info = self._execute_transcribe_pass(audio_np, use_vad=False)
            vad_fallback_used = True

        full_text = []
        segment_details = []

        for seg in raw_segments:
            raw_seg = seg.text.strip()
            if not raw_seg:
                continue

            # Bersihkan kurung deskripsi non-verbal Whisper seperti [sigh], (batuk)
            cleaned_seg = re.sub(r'\[.*?\]|\(.*?\)', '', raw_seg).strip()
            check_words = re.sub(r'[^\w\s]', '', cleaned_seg).lower().strip()
            if not check_words:
                continue

            # Filter segmen yang hanya berisi noise / filler murni
            if check_words in HALLUCINATIONS_AND_NOISE or re.match(r'^(?:h+m+|m{2,}|u+h+|u+m+|a+h+|amin|aamiin|amen)$', check_words):
                continue

            # Bersihkan inline filler tanpa merusak kata bermakna
            cleaned_seg = re.sub(r'(?<!\bpak\s)(?<!\bbapak\s)\b(?:amin|aamiin)\b', '', cleaned_seg, flags=re.I)
            cleaned_seg = re.sub(r'\b(?:h+m+|m{2,}|u+h+|u+m+)\b', '', cleaned_seg, flags=re.I)
            cleaned_seg = re.sub(r'\s+', ' ', cleaned_seg).strip()

            if cleaned_seg:
                full_text.append(cleaned_seg)
                avg_lp = getattr(seg, "avg_logprob", -0.2)
                try:
                    seg_conf = round(min(0.99, max(0.40, float(math.exp(avg_lp)))), 2)
                except Exception:
                    seg_conf = 0.85

                segment_details.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": cleaned_seg,
                    "confidence": seg_conf
                })

        combined_text = " ".join(full_text).strip()
        dur = round(info.duration, 2)

        # Filter sisa halusinasi pada audio sangat pendek
        if dur < 3.5 and combined_text.lower() in HALLUCINATIONS_AND_NOISE:
            combined_text = ""
            segment_details = []

        # Tentukan status transkripsi
        if combined_text:
            trans_status = "success"
        else:
            trans_status = "empty_speech"

        diag["vad_fallback_used"] = vad_fallback_used

        return {
            "text": combined_text,
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": dur,
            "segments": segment_details,
            "audio_status": diag["status"],
            "transcription_status": trans_status,
            "audio_diagnostics": diag,
            "vad_fallback_used": vad_fallback_used,
            "warning": diag.get("warning")
        }
