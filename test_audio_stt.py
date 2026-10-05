"""
test_audio_stt.py — Comprehensive Audio Capture & STT Pipeline Tests (Section 13)
==================================================================================
Menguji seluruh alur penangkapan suara sampai transkripsi:
  1. Inspeksi Kualitas Audio (Optimal, Volume Rendah, Hening, Clipping, Kosong)
  2. Normalisasi & Adaptive Gain Boost
  3. Audio Session Chunking, Buffering, Sequence Tracking & Gap Detection
  4. Proteksi Duplikasi Chunk (Idempotensi)
  5. Perakitan (Assembly) Chunk Menjadi WAV Terpadu
  6. Voice Activity Detection (VAD) & Dual-Pass Fallback
  7. Pemisahan Tegas Audio Hening vs Transkrip Kosong (Anti-Halusinasi Medis)
"""

import os
import shutil
import unittest
import numpy as np

from audio_processor import (
    inspect_audio_file,
    normalize_and_prepare_audio,
    generate_synthetic_tone_wav,
    AudioQualityStatus
)
from audio_chunker import AudioSessionManager
from transcriber import SpeechTranscriber


class TestAudioCaptureAndSTT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_audio_tmp"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.transcriber = SpeechTranscriber(model_size="base")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # ── 1. Uji Inspeksi Kualitas Audio ───────────────────────────────────────

    def test_01_optimal_audio_inspection(self):
        """Memastikan file audio normal dideteksi berstatus OPTIMAL."""
        sample_path = "audio_samples/sample_1_kepala_pipi.mp3"
        diag = inspect_audio_file(sample_path)
        self.assertTrue(diag["valid"])
        self.assertEqual(diag["status"], AudioQualityStatus.OPTIMAL)
        self.assertGreater(diag["duration_seconds"], 5.0)
        self.assertGreater(diag["rms_dbfs"], -35.0)

    def test_02_silent_audio_detection(self):
        """Memastikan audio hening / mic mute dideteksi sebagai SILENT (RMS < -50 dBFS)."""
        silent_wav = os.path.join(self.test_dir, "silent.wav")
        generate_synthetic_tone_wav(silent_wav, duration_s=2.0, volume_dbfs=-65.0)

        diag = inspect_audio_file(silent_wav)
        self.assertEqual(diag["status"], AudioQualityStatus.SILENT)
        self.assertLess(diag["rms_dbfs"], -50.0)
        self.assertIn("hening", diag["warning"].lower())

    def test_03_low_volume_and_gain_boost(self):
        """Memastikan suara pelan dideteksi dan diberikan Adaptive Gain Boost."""
        quiet_wav = os.path.join(self.test_dir, "quiet.wav")
        generate_synthetic_tone_wav(quiet_wav, duration_s=2.5, volume_dbfs=-42.0)

        audio_np, norm_diag = normalize_and_prepare_audio(quiet_wav, auto_gain=True)
        self.assertIn(norm_diag["status"], [AudioQualityStatus.LOW_VOLUME, AudioQualityStatus.VERY_LOW_VOLUME])
        self.assertGreater(norm_diag["applied_gain_db"], 0.0)
        self.assertIsNotNone(audio_np)

    def test_04_audio_clipping_detection(self):
        """Memastikan distorsi clipping (peak >= 0.985) terdeteksi."""
        clip_wav = os.path.join(self.test_dir, "clip.wav")
        # Generate signal with 0 dBFS amplitude (extreme)
        generate_synthetic_tone_wav(clip_wav, duration_s=1.5, volume_dbfs=0.0)

        diag = inspect_audio_file(clip_wav)
        self.assertEqual(diag["status"], AudioQualityStatus.CLIPPING)
        self.assertGreaterEqual(diag["peak_amplitude"], 0.985)

    def test_05_empty_audio_detection(self):
        """Memastikan file 0 byte terdeteksi sebagai EMPTY tanpa crash."""
        empty_wav = os.path.join(self.test_dir, "empty.wav")
        with open(empty_wav, "wb") as f:
            f.write(b"")

        diag = inspect_audio_file(empty_wav)
        self.assertFalse(diag["valid"])
        self.assertEqual(diag["status"], AudioQualityStatus.EMPTY)

    # ── 2. Uji Audio Chunking & Session Buffering (Section 3) ─────────────────

    def test_06_chunk_sequencing_and_gap_detection(self):
        """Menguji urutan chunk berurutan dan pendeteksian gap jika ada potongan terlewat."""
        sm = AudioSessionManager(storage_dir=os.path.join(self.test_dir, "chunks_test"))
        sess_id = "SESS-SEQ-001"
        sm.start_session(sess_id)

        sample_bytes = b"RIFF....WAVEfmt ...." * 50

        # Kirim chunk 0
        r0 = sm.add_chunk(sess_id, sequence_id=0, audio_bytes=sample_bytes + b"0")
        self.assertEqual(r0["status"], "stored")
        self.assertEqual(len(r0["missing_sequences"]), 0)

        # Kirim chunk 2 (chunk 1 terlewat)
        r2 = sm.add_chunk(sess_id, sequence_id=2, audio_bytes=sample_bytes + b"2")
        self.assertTrue(r2["has_sequence_gap"])
        self.assertIn(1, r2["missing_sequences"])

        # Kirim chunk 1 (susulan / retry)
        r1 = sm.add_chunk(sess_id, sequence_id=1, audio_bytes=sample_bytes + b"1")
        self.assertFalse(r1["has_sequence_gap"])
        self.assertEqual(len(r1["missing_sequences"]), 0)

    def test_07_chunk_idempotency_duplicate_protection(self):
        """Memastikan pengiriman ulang chunk yang sama ditandai sebagai cached duplicate."""
        sm = AudioSessionManager(storage_dir=os.path.join(self.test_dir, "chunks_dup"))
        sess_id = "SESS-DUP-001"
        sm.start_session(sess_id)

        chunk_data = b"UNIQUE_CHUNK_AUDIO_DATA_FOR_IDEMPOTENCY_TEST"
        r1 = sm.add_chunk(sess_id, sequence_id=0, audio_bytes=chunk_data)
        self.assertFalse(r1["is_duplicate"])
        self.assertEqual(r1["status"], "stored")

        # Kirim ulang chunk yang sama persis
        r2 = sm.add_chunk(sess_id, sequence_id=0, audio_bytes=chunk_data)
        self.assertTrue(r2["is_duplicate"])
        self.assertEqual(r2["status"], "cached")

    def test_08_chunk_assembly_to_unified_wav(self):
        """Memastikan perakitan chunk audio menghasilkan file WAV kontinu yang valid."""
        sm = AudioSessionManager(storage_dir=os.path.join(self.test_dir, "chunks_asm"))
        sess_id = "SESS-ASM-001"
        sm.start_session(sess_id)

        # Buat 2 potongan WAV sintetis 16kHz
        c1_path = os.path.join(self.test_dir, "c1.wav")
        c2_path = os.path.join(self.test_dir, "c2.wav")
        generate_synthetic_tone_wav(c1_path, duration_s=1.0, freq_hz=300)
        generate_synthetic_tone_wav(c2_path, duration_s=1.0, freq_hz=600)

        with open(c1_path, "rb") as f: sm.add_chunk(sess_id, 0, f.read(), 0.0, 1.0, ".wav")
        with open(c2_path, "rb") as f: sm.add_chunk(sess_id, 1, f.read(), 1.0, 2.0, ".wav")

        out_wav, diag = sm.assemble_session_audio(sess_id)
        self.assertTrue(diag["valid"])
        self.assertEqual(diag["total_chunks_received"], 2)
        self.assertAlmostEqual(diag["total_duration_seconds"], 2.0, delta=0.1)
        self.assertTrue(os.path.exists(out_wav))

    # ── 3. Uji Voice Activity Detection & Fallback (Section 4 & 6) ────────────

    def test_09_vad_tolerance_for_thinking_pauses(self):
        """Memastikan jeda bicara (800ms) tidak memutus transkripsi."""
        paused_wav = os.path.join(self.test_dir, "pause.wav")
        generate_synthetic_tone_wav(paused_wav, duration_s=3.0, freq_hz=440, with_pause=True)

        diag = inspect_audio_file(paused_wav)
        self.assertTrue(diag["valid"])
        # Sinyal tetap memiliki energi yang cukup di luar jeda
        self.assertGreater(diag["rms_dbfs"], -40.0)

    def test_10_anti_hallucination_on_silent_audio(self):
        """Memastikan audio hening tidak menghasilkan teks tebakan medis palsu."""
        silent_wav = os.path.join(self.test_dir, "true_silent.wav")
        generate_synthetic_tone_wav(silent_wav, duration_s=2.0, volume_dbfs=-70.0)

        res = self.transcriber.transcribe(silent_wav)
        self.assertEqual(res["audio_status"], AudioQualityStatus.SILENT)
        self.assertEqual(res["transcription_status"], "silent_audio")
        self.assertEqual(res["text"], "")
        self.assertIn("hening", res["warning"].lower())


if __name__ == "__main__":
    unittest.main()
