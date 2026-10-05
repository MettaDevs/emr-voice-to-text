"""
test_clinical_fidelity.py — MedVoice AI Clinical Fidelity & Meaning Safety Test
==============================================================================
Memverifikasi 10 Aturan Mutlak Keselamatan Makna Klinis (Clinical Fidelity Rules):
  1. "Saya sesak" != "Saya tidak sesak"
  2. "Saya tidak sesak" != "Saya sesak"
  3. "Saya alergi obat" tidak boleh mengarang nama obat tertentu
  4. "Saya tidak tahu punya alergi atau tidak" (unknown) != "Saya tidak memiliki alergi" (denied)
  5. "Bukan sejak kemarin, tetapi sejak tadi pagi" menggunakan durasi koreksi terbaru
  6. Pertanyaan dokter tidak boleh dianggap sebagai pernyataan pasien
  7. Pernyataan pendamping harus tetap memiliki sumber 'companion'
  8. Angka dan dosis tidak boleh diubah tanpa penandaan
  9. Kata tidak jelas harus ditandai sebagai membutuhkan review / uncertain
  10. Informasi yang tidak disebutkan tidak boleh dibuat secara otomatis
"""

import unittest
from medical_extractor import (
    MedicalComplaintExtractor,
    correct_with_trace,
    _extract_standardized_allergies,
    _extract_structured_duration,
    _extract_structured_symptoms,
    _classify_sentence_speaker
)


class TestClinicalFidelity(unittest.TestCase):
    def setUp(self):
        self.extractor = MedicalComplaintExtractor()

    # ── Rule 1 & 2: Preservasi Polaritas Gejala Positif vs Negatif ───────────
    def test_rule_1_positive_symptom_not_negated(self):
        """Rule 1: 'Saya sesak' tidak boleh berubah menjadi 'Saya tidak sesak'."""
        text = "Dokter: Keluhan apa? Pasien: Saya sesak napas dok."
        res = self.extractor.extract(text)
        symptoms = res.get("symptoms", [])
        sesak_symptom = next((s for s in symptoms if "sesak" in s["name"].lower()), None)
        self.assertIsNotNone(sesak_symptom, "Sesak napas harus terdeteksi")
        self.assertEqual(sesak_symptom["status"], "present", "Sesak napas harus berstatus 'present'")

    def test_rule_2_negative_symptom_not_asserted(self):
        """Rule 2: 'Saya tidak sesak' tidak boleh berubah menjadi 'Saya sesak'."""
        text = "Dokter: Ada sesak napas? Pasien: Saya tidak merasa sesak, tetapi badan terasa lemas."
        res = self.extractor.extract(text)
        symptoms = res.get("symptoms", [])

        # Cari gejala sesak
        sesak_symptom = next((s for s in symptoms if "sesak" in s["name"].lower()), None)
        if sesak_symptom:
            self.assertEqual(sesak_symptom["status"], "absent",
                             "Gejala 'sesak' yang dinegasikan harus berstatus 'absent', bukan 'present'")

        # Keluhan utama dilarang mencatat sesak
        chief = res["chief_complaint"]["value"] or res["keluhan_utama"]
        self.assertNotIn("sesak", chief.lower(), "Keluhan utama tidak boleh diisi 'sesak' saat pasien menyangkal")

    # ── Rule 3: Tidak Mengarang Nama Obat pada Alergi Obat Umum ──────────────
    def test_rule_3_generic_allergy_does_not_hallucinate_drug_name(self):
        """Rule 3: 'Saya alergi obat' tidak boleh mengarang obat tertentu (mis. amoxicillin/paracetamol)."""
        text = "Pasien: Saya ada alergi obat dok, tapi lupa nama obatnya."
        res = _extract_standardized_allergies(text)
        self.assertEqual(res["status"], "reported")

        # Pastikan tidak ada halusinasi nama obat spesifik
        forbid_drugs = ["amoxicillin", "paracetamol", "ibuprofen", "cefixime", "penisilin"]
        for drug in forbid_drugs:
            for item in res["items"]:
                self.assertNotIn(drug, item.lower(), f"Dilarang mengarang nama obat '{drug}' saat tidak disebutkan")

        # Harus memicu review_required karena nama obat belum spesifik
        self.assertTrue(res["review_required"], "Alergi obat tanpa nama spesifik harus memicu review_required")

    # ── Rule 4: Pembedaan Tegas Antara Unknown dan Denied Alergi ─────────────
    def test_rule_4_allergy_unknown_vs_denied_distinction(self):
        """Rule 4: 'Saya tidak tahu punya alergi atau tidak' != 'Saya tidak memiliki alergi'."""
        # Kasus A: Unknown
        res_unknown = _extract_standardized_allergies("Saya tidak tahu punya alergi atau tidak dok.")
        self.assertEqual(res_unknown["status"], "unknown", "Harus berstatus 'unknown'")
        self.assertTrue(res_unknown["review_required"], "Status unknown harus memerlukan review dokter")

        # Kasus B: Denied
        res_denied = _extract_standardized_allergies("Saya tidak ada riwayat alergi dok.")
        self.assertEqual(res_denied["status"], "denied", "Harus berstatus 'denied'")
        self.assertFalse(res_denied["review_required"], "Status denied tidak memerlukan review dokter")

    # ── Rule 5: Koreksi Ucapan Durasi Terbaru ────────────────────────────────
    def test_rule_5_self_correction_of_duration(self):
        """Rule 5: 'Bukan sejak kemarin, tetapi sejak tadi pagi' harus menggunakan informasi koreksi terbaru."""
        text = "Pasien: Demamnya bukan sejak kemarin, tetapi sejak tadi pagi dok."
        dur_res = _extract_structured_duration(text)
        self.assertIsNotNone(dur_res, "Durasi harus terdeteksi")
        self.assertIn("tadi pagi", dur_res["original_text"].lower(),
                      "Durasi harus mengambil koreksi terbaru 'tadi pagi', bukan 'kemarin'")
        self.assertEqual(dur_res["event_time"], "Tadi pagi")

    # ── Rule 6: Pertanyaan Dokter Bukan Pernyataan Pasien ────────────────────
    def test_rule_6_doctor_question_not_patient_statement(self):
        """Rule 6: Pertanyaan dokter tidak boleh dianggap sebagai keluhan pasien."""
        text = (
            "Dokter: Apakah ada muntah darah atau batuk darah? "
            "Pasien: Tidak ada dok, saya hanya pusing."
        )
        res = self.extractor.extract(text)
        chief = res["chief_complaint"]["value"] or res["keluhan_utama"]
        self.assertIn("Pusing", chief)
        self.assertNotIn("darah", chief.lower(), "Pertanyaan dokter tentang muntah/batuk darah tidak boleh jadi keluhan utama")

    # ── Rule 7: Atribusi Sumber Pendamping (Companion) ───────────────────────
    def test_rule_7_companion_statement_retains_companion_source(self):
        """Rule 7: Pernyataan pendamping harus tetap memiliki sumber 'companion'."""
        text = (
            "Dokter: Selamat pagi, keluhannya apa? "
            "Pendamping: Anak saya badannya panas dok sejak tadi malam."
        )
        res = self.extractor.extract(text)
        # Sumber keluhan utama harus companion
        self.assertEqual(res["chief_complaint"]["source"], "companion",
                         "Sumber keluhan utama yang disampaikan pendamping harus 'companion'")
        # Gejala terstruktur harus memiliki sumber companion
        symptoms = res.get("symptoms", [])
        self.assertTrue(any(s["source"] == "companion" for s in symptoms),
                        "Gejala harus mencatat source 'companion'")

    # ── Rule 8: Proteksi Angka dan Dosis ────────────────────────────────────
    def test_rule_8_dosages_and_vital_numbers_protected(self):
        """Rule 8: Angka dan dosis tidak boleh diubah tanpa penandaan."""
        raw_text = "Pasien minum paracetamol 500 mg dua kali sehari dan tensi 120/80 mmHg."
        trace = correct_with_trace(raw_text)
        corrected = trace["corrected_text"]

        self.assertIn("500 mg", corrected, "Dosis '500 mg' harus dipertahankan utuh")
        self.assertIn("120/80", corrected, "Angka tensi '120/80' harus dipertahankan utuh")

    # ── Rule 9: Kata Tidak Jelas Ditandai Membutuhkan Review ─────────────────
    def test_rule_9_uncertain_speech_flagged_for_review(self):
        """Rule 9: Koreksi fonetik atau kata ambigu harus ditandai review_required=True."""
        trace = correct_with_trace("Pasien mengeluh tanasnya dari kemarin.")
        corr_items = trace.get("corrections", [])
        tanas_corr = next((c for c in corr_items if "tanas" in c["original"].lower()), None)
        self.assertIsNotNone(tanas_corr, "Koreksi 'tanas' harus tercatat")
        self.assertTrue(tanas_corr.get("review_required", False),
                        "Koreksi fonetik harus memicu review_required=True")

    # ── Rule 10: Dilarang Mengarang Informasi yang Tidak Disebutkan ──────────
    def test_rule_10_unmentioned_info_not_fabricated(self):
        """Rule 10: Informasi yang tidak disebutkan tidak boleh dibuat secara otomatis."""
        text = "Dokter: Keluhan apa? Pasien: Saya merasa lemas dan tidak enak badan, tapi lupa mulai kapan."
        res = self.extractor.extract(text)

        # Durasi tidak boleh dikarang
        self.assertEqual(res["lama_sakit"], "", "Durasi tidak boleh diisi jika pasien menyatakan lupa")
        self.assertIsNone(res["duration"], "Structured duration harus None")


if __name__ == "__main__":
    unittest.main()
