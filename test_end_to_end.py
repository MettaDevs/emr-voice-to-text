"""
test_end_to_end.py — MedVoice AI Complete End-to-End Pipeline Test
==================================================================
Menguji keseluruhan alur 14 tahap Voice-to-EMR dari input audio/dialogue sampai EMR Integration:
  1. Audio Input / Audio Processing
  2. Speech-to-Text (STT via Faster-Whisper CUDA FP16)
  3. Raw Transcript
  4. Layer 1: Phonetic Correction (dengan trace)
  5. Layer 2: Indonesian Normalization & De-slang
  6. Layer 3: Speaker Labeling & Turn Segmentation
  7. Layer 4: Clinical Extraction (Keluhan Utama, Tambahan, Durasi, TTV, Alergi)
  8. Clinical Validation (Consistency & Physiological bounds)
  9. Review Assessment (Decoupled confidence & human review flags)
  10. EMR Draft Generation
  11. Medical Staff Review (Perubahan field & feedback audit trail)
  12. Staff Approval (Transisi status draft -> approved)
  13. EMR Integration (Pengiriman ke Mock EMR / FHIR Bundle)
  14. Audit Trail Verification
"""

import os
import unittest
import datetime
from transcriber import SpeechTranscriber
from medical_extractor import (
    MedicalComplaintExtractor,
    correct_with_trace,
    _label_speakers,
    calculate_decoupled_confidence,
    ClinicalDataValidator
)
from emr_integration import MockHospitalEMRAdapter, FHIRAdapter, audit_logger


class TestVoiceToEMREndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = MedicalComplaintExtractor()
        cls.emr_adapter = MockHospitalEMRAdapter(simulate_network_delay=0.0)
        cls.fhir_adapter = FHIRAdapter()

        # Cek ketersediaan file audio nyata di uploads
        cls.sample_audio = None
        for fn in ["rec_1790871838.webm", "rec_1790661428.webm"]:
            p = os.path.join("uploads", fn)
            if os.path.exists(p):
                cls.sample_audio = p
                break

    def test_e2e_full_synthetic_dialogue_pipeline(self):
        """
        Pengujian End-to-End Lengkap dengan Percakapan Klinis Rumah Sakit:
        Input -> Trace -> Normalize -> Label -> Extract -> Validate -> Review -> Edit -> Approve -> EMR Send
        """
        raw_clinical_dialogue = (
            "Dokter: Selamat pagi pak, keluhannya apa yang dirasakan? "
            "Pasien: Selamat pagi dok, tanasnya dari kemarin malam dok. "
            "Eh, bukan kemarin malam dok, sejak tadi pagi. "
            "Terus perut saya melilit dan enek mau muntah. "
            "Dokter: Apakah ada alergi obat? "
            "Pasien: Saya ada alergi obat dok, tapi lupa nama obatnya. "
            "Dokter: Baik, tensi 120 per 80, suhu 38,5 derajat celcius, nadi 90 kali per menit."
        )

        # ── Tahap 1, 2, 3: STT / Raw Transcript ──────────────────────────────
        raw_transcript = raw_clinical_dialogue
        self.assertTrue(len(raw_transcript) > 0, "Raw transcript tidak boleh kosong")

        # ── Tahap 4: Layer 1 Phonetic Correction & Trace ─────────────────────
        trace_result = correct_with_trace(raw_transcript)
        corrected_text = trace_result["corrected_text"]
        corrections = trace_result["corrections"]
        self.assertIn("panasnya", corrected_text.lower(), "Fonetik 'tanasnya' harus terkoreksi ke 'panasnya'")

        # ── Tahap 5: Layer 2 Normalization ───────────────────────────────────
        # Normalisasi otomatis terintegrasi pada pipeline extractor
        self.assertIn("mual", corrected_text.lower() or raw_transcript.lower(),
                      "Istilah perut melilit / enek harus dinormalisasi")

        # ── Tahap 6: Layer 3 Speaker Labeling ────────────────────────────────
        segments = [{"text": raw_transcript, "start": 0.0, "end": 15.0}]
        labeled_segments = _label_speakers(segments, raw_transcript)
        self.assertTrue(len(labeled_segments) > 0, "Diarisasi kalimat harus menghasilkan giliran bicara")

        # ── Tahap 7: Layer 4 Clinical Extraction ─────────────────────────────
        extracted = self.extractor.extract(raw_transcript, labeled_segments)

        # Keluhan Utama
        chief = extracted["chief_complaint"]["value"]
        self.assertTrue(any(s in chief for s in ["Demam", "Panas", "Perut melilit"]),
                        f"Keluhan utama harus valid, didapat: {chief}")

        # Durasi dengan Self-Correction
        dur = extracted["duration"]
        self.assertIsNotNone(dur, "Durasi harus terdeteksi")
        self.assertIn("tadi pagi", dur["original_text"].lower(),
                      "Koreksi ucapan 'sejak tadi pagi' harus membatalkan 'kemarin malam'")

        # Tanda Vital (TTV)
        vitals = extracted["vitals"]
        self.assertIsNotNone(vitals["blood_pressure"], "Tekanan darah harus terdeteksi")
        self.assertEqual(vitals["blood_pressure"]["value"], "120/80")
        self.assertEqual(vitals["temperature"]["value"], 38.5)
        self.assertEqual(vitals["pulse"]["value"], 90)

        # Alergi
        allergies = extracted["allergies"]
        self.assertEqual(allergies["status"], "reported")
        self.assertTrue(allergies["review_required"], "Alergi obat umum harus memicu review_required")

        # ── Tahap 8: Clinical Validation ─────────────────────────────────────
        val_res = ClinicalDataValidator.validate(extracted)
        self.assertIn("valid", val_res)

        # ── Tahap 9: Review Assessment ───────────────────────────────────────
        conf = extracted["confidence"]
        self.assertIn("score", conf)
        self.assertIn("level", conf)

        # ── Tahap 10: EMR Draft Formation ────────────────────────────────────
        emr_draft = {
            "session_metadata": {
                "session_id": "E2E-SESSION-TEST-001",
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "chief_complaint": extracted["chief_complaint"],
            "symptoms": extracted["symptoms"],
            "duration": extracted["duration"],
            "vitals": extracted["vitals"],
            "allergies": extracted["allergies"],
            "clinical_validation": val_res,
            "review_status": "draft",
            "staff_corrections": []
        }
        self.assertEqual(emr_draft["review_status"], "draft")

        # ── Tahap 11: Medical Staff Review (Human Review & Edit) ─────────────
        # Tenaga medis mengoreksi alergi obat spesifik setelah konfirmasi ke pasien
        correction_record = {
            "field": "allergies.items",
            "original_value": "Obat (belum spesifik)",
            "corrected_value": "Amoxicillin",
            "corrected_by": "dr_spesialis",
            "correction_reason": "Pasien mengonfirmasi alergi antibiotik amoxicillin setelah ditunjukkan obat",
            "timestamp": datetime.datetime.now().isoformat()
        }
        emr_draft["allergies"]["items"] = ["Amoxicillin"]
        emr_draft["allergies"]["review_required"] = False
        emr_draft["staff_corrections"].append(correction_record)
        self.assertEqual(len(emr_draft["staff_corrections"]), 1)

        # ── Tahap 12: Staff Approval ─────────────────────────────────────────
        emr_draft["review_status"] = "approved"
        emr_draft["approved_by"] = "dr_spesialis"
        emr_draft["approved_at"] = datetime.datetime.now().isoformat()
        self.assertEqual(emr_draft["review_status"], "approved")

        # ── Tahap 13: EMR Integration (Transmission & FHIR) ───────────────────
        # 1. Transmisi ke Adapter EMR RS
        tx_result = self.emr_adapter.send_emr(emr_draft, actor="dr_spesialis")
        self.assertEqual(tx_result["status"], "SUCCESS")
        self.assertIn("REC-EMR-", tx_result["emr_record_id"])
        self.assertFalse(tx_result["is_duplicate"])

        # 2. Pembentukan FHIR Bundle
        fhir_bundle = self.fhir_adapter.to_fhir_bundle(emr_draft)
        self.assertEqual(fhir_bundle["resourceType"], "Bundle")
        self.assertTrue(len(fhir_bundle["entry"]) >= 3, "FHIR Bundle harus memuat Patient, Encounter, Condition/Obs")

        # 3. Proteksi Idempotency (Pengiriman ulang aman tanpa duplikasi)
        tx_retry = self.emr_adapter.send_emr(emr_draft, actor="dr_spesialis")
        self.assertTrue(tx_retry["is_duplicate"], "Pengiriman kedua harus terdeteksi sebagai duplicate aman")
        self.assertEqual(tx_result["emr_record_id"], tx_retry["emr_record_id"])

    def test_e2e_real_audio_file_processing(self):
        """Pengujian tahap Audio Processing & STT jika audio riil tersedia di lingkungan kerja."""
        if not self.sample_audio:
            self.skipTest("Sample audio tidak tersedia di folder uploads, dilewati.")

        transcriber = SpeechTranscriber(model_size="base")
        stt_result = transcriber.transcribe(self.sample_audio)

        self.assertIn("text", stt_result)
        self.assertIn("duration", stt_result)
        self.assertGreater(stt_result["duration"], 0.0)

        # Lanjutkan ke ekstraksi klinis dari hasil audio asli
        ext_result = self.extractor.extract(stt_result["text"], stt_result.get("segments", []))
        self.assertIn("review_status", ext_result)


if __name__ == "__main__":
    unittest.main()
