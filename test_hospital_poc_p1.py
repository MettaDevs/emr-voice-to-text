"""
Test Suite P1: Kelengkapan Ekstraksi & Struktur Data EMR Konsisten (Hospital POC)
Memvalidasi:
1. Ekstraksi keluhan utama terstruktur (body part, onset, duration, severity, source).
2. Normalisasi durasi & waktu klinis (value, unit, is_approximate, event_time).
3. Ekstraksi tanda-tanda vital lengkap (8 parameter + physiological bounds check).
4. Decoupled multi-dimensional confidence scoring & review trigger.
5. Konsistensi struktur data EMR terpadu (16 bagian Section 10).
"""

import unittest
from medical_extractor import MedicalComplaintExtractor


class TestHospitalPOCP1(unittest.TestCase):

    def setUp(self):
        self.extractor = MedicalComplaintExtractor()

    # ── 1. Structured Chief Complaint ───────────────────────────────────────
    def test_structured_chief_complaint(self):
        text = "Pasien: Perut saya melilit sangat parah sejak kemarin malam dok."
        res = self.extractor.extract(text)
        cc = res["chief_complaint"]

        self.assertIsNotNone(cc["value"], "Keluhan utama harus terisi")
        self.assertIn("Perut", cc["value"])
        self.assertEqual(cc["body_part"], "Saluran Cerna / Abdomen")
        self.assertEqual(cc["severity"], "berat")
        self.assertEqual(cc["source"], "patient")
        self.assertIn("Kemarin", str(cc["onset"]))

    # ── 2. Structured Duration & Time ───────────────────────────────────────
    def test_structured_duration_various_formats(self):
        # Format A: "sudah tiga hari"
        res_a = self.extractor.extract("Pasien: Batuk kering sudah tiga hari dok.")
        dur_a = res_a["duration"]
        self.assertIsNotNone(dur_a)
        self.assertEqual(dur_a["value"], 3)
        self.assertEqual(dur_a["unit"], "day")

        # Format B: "sejak tadi pagi"
        res_b = self.extractor.extract("Pasien: Nyeri dada sejak tadi pagi dok.")
        dur_b = res_b["duration"]
        self.assertIsNotNone(dur_b)
        self.assertEqual(dur_b["event_time"], "Tadi pagi")

        # Format C: "sekitar seminggu"
        res_c = self.extractor.extract("Pasien: Sakit pinggang sekitar seminggu dok.")
        dur_c = res_c["duration"]
        self.assertIsNotNone(dur_c)
        self.assertTrue(dur_c["is_approximate"])

        # Format D: Tidak disebutkan durasi
        res_d = self.extractor.extract("Pasien: Saya merasa pusing dok.")
        self.assertIsNone(res_d["duration"], "Jika durasi tidak disebutkan, harus bernilai None")

    # ── 3. Extended Vitals & Physiological Validation ───────────────────────
    def test_extended_vitals_extraction(self):
        # Percakapan pemeriksaan TTV lengkap
        dialogue = (
            "Dokter: Saya cek tanda vitalnya ya bu. "
            "Tensinya 120/80 mmHg, suhunya 37.2 derajat celcius, "
            "saturasi oksigen 98 persen, nadi 82 x/menit, napas 18 kali per menit, "
            "berat badan 65 kg, tinggi badan 168 cm, skala nyeri 3."
        )
        res = self.extractor.extract(dialogue)
        vitals = res["vitals"]

        # 1. Tekanan Darah
        self.assertIsNotNone(vitals["blood_pressure"])
        self.assertEqual(vitals["blood_pressure"]["value"], "120/80")
        self.assertEqual(vitals["blood_pressure"]["unit"], "mmHg")
        self.assertFalse(vitals["blood_pressure"]["review_required"])

        # 2. Suhu
        self.assertIsNotNone(vitals["temperature"])
        self.assertEqual(vitals["temperature"]["value"], 37.2)
        self.assertEqual(vitals["temperature"]["unit"], "°C")

        # 3. SpO2
        self.assertIsNotNone(vitals["spo2"])
        self.assertEqual(vitals["spo2"]["value"], 98)
        self.assertEqual(vitals["spo2"]["unit"], "%")

        # 4. Nadi
        self.assertIsNotNone(vitals["pulse"])
        self.assertEqual(vitals["pulse"]["value"], 82)
        self.assertEqual(vitals["pulse"]["unit"], "bpm")

        # 5. Frekuensi Napas
        self.assertIsNotNone(vitals["respiratory_rate"])
        self.assertEqual(vitals["respiratory_rate"]["value"], 18)
        self.assertEqual(vitals["respiratory_rate"]["unit"], "x/min")

        # 6. Berat & Tinggi
        self.assertEqual(vitals["weight"]["value"], 65.0)
        self.assertEqual(vitals["height"]["value"], 168.0)

        # 7. Skala Nyeri
        self.assertEqual(vitals["pain_scale"]["value"], 3)

    def test_vital_abnormal_triggers_review(self):
        # Suhu 40.5°C harus memicu review_required
        res = self.extractor.extract("Dokter: Suhu tubuh bapak 40.5 derajat celcius, demam sangat tinggi.")
        vitals = res["vitals"]
        self.assertTrue(vitals["temperature"]["review_required"], "Suhu 40.5°C harus memicu review_required")

    # ── 4. Decoupled Multi-Dimensional Confidence ───────────────────────────
    def test_decoupled_confidence_structure(self):
        text = "Dokter: Ada keluhan apa? Pasien: Pusing sejak kemarin."
        res = self.extractor.extract(text)
        conf = res["confidence"]

        self.assertIn("transcription", conf)
        self.assertIn("speaker", conf)
        self.assertIn("clinical_extraction", conf)
        self.assertIn("normalization", conf)

        # Review assessment
        review = res["review"]
        self.assertIn("required", review)
        self.assertIn("reasons", review)
        self.assertIsInstance(review["reasons"], list)

    # ── 5. Unified EMR Output Schema (Section 10) ───────────────────────────
    def test_unified_emr_output_schema(self):
        dialogue = (
            "Dokter: Selamat pagi, keluhan apa bu? "
            "Pasien: Demam tinggi dari kemarin lusa dok, sama batuk. Tidak ada alergi obat. "
            "Dokter: Tensinya 110/70 mmHg ya bu."
        )
        res = self.extractor.extract(dialogue)

        # Verifikasi 16 komponen utama Section 10
        self.assertIn("session_metadata", res)
        self.assertIn("original_text", res)
        self.assertIn("corrected_text", res)
        self.assertIn("corrections", res)
        self.assertIn("segments", res)
        self.assertIn("chief_complaint", res)
        self.assertIn("symptoms", res)
        self.assertIn("duration", res)
        self.assertIn("vitals", res)
        self.assertIn("allergies", res)
        self.assertIn("medical_history", res)
        self.assertIn("medications_mentioned", res)
        self.assertIn("procedures_mentioned", res)
        self.assertIn("uncategorized", res)
        self.assertIn("confidence", res)
        self.assertIn("clinical_validation", res)
        self.assertIn("review_status", res)

        # Status awal draft / needs_review
        self.assertIn(res["review_status"], ["draft", "needs_review"])

        # Verifikasi backward compatibility keys
        self.assertIn("fields", res)
        self.assertIn("keluhan_utama", res)
        self.assertIn("keluhan_tambahan", res)
        self.assertIn("lama_sakit", res)
        self.assertIn("raw_transcript", res)
        self.assertIn("cleaned_transcript", res)


if __name__ == "__main__":
    unittest.main()
