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

    # ── Rule 11: Turn Switch Splitting in Continuous Segment ────────────────
    def test_doctor_patient_turn_split_in_continuous_segment(self):
        """Turn switch: '...surat DC juga, ya bu, coba kita cek tensi dulu ya' harus memisahkan Dokter vs Pasien."""
        raw_segments = [
            {"start": 0.7, "end": 2.9, "text": "Malam dok, malam."},
            {"start": 2.9, "end": 7.9, "text": "Kalau boleh tahu keluhannya apa ya bu?"},
            {"start": 7.9, "end": 29.2, "text": "Ini dok, saya pusing dari kemarin, sama panas juga badannya, kalau panasnya dari semalam dok ya, sama mau cari surat DC juga, ya bu, coba kita cek tensi dulu ya."}
        ]
        res = self.extractor.extract(" ".join(s["text"] for s in raw_segments), segments=raw_segments)
        labeled = res["labeled_segments"]

        # Harus terbagi menjadi 4 segmen dialog
        self.assertEqual(len(labeled), 4)

        # Periksa segmen terakhir adalah ucapan dokter mengenai pemeriksaan tensi
        last_seg = labeled[-1]
        self.assertEqual(last_seg["speaker"], "Dokter")
        self.assertEqual(last_seg["speaker_role"], "doctor")
        self.assertIn("cek tensi", last_seg["text"].lower())

        # Periksa segmen sebelum terakhir adalah ucapan pasien
        penultimate_seg = labeled[-2]
        self.assertEqual(penultimate_seg["speaker"], "Pasien")
        self.assertEqual(penultimate_seg["speaker_role"], "patient")
        self.assertIn("pusing", penultimate_seg["text"].lower())

    # ── Rule 12: Ekstraksi Gejala dengan Kata Keterangan (Adverb) & Cross-Turn ───
    def test_throat_and_cough_symptom_with_adverbs(self):
        """Ucapan 'Ada sedikit batuk tapi tidak terlalu sering, tenggorokan juga agak sakit'
        harus mengekstrak batuk dan sakit tenggorokan sebagai present, bukan absent."""
        segments = [
            {"start": 0.6, "end": 5.4, "text": "Pagi dok, saya merasa panas dan badan saya agak lemas."},
            {"start": 5.4, "end": 8.4, "text": "Demamnya mulai terasa sejak kapan?"},
            {"start": 8.4, "end": 10.4, "text": "Sejak kemarin sore dok."},
            {"start": 10.4, "end": 14.5, "text": "Apakah ada batuk pilek atau sakit tenggorokan?"},
            {"start": 14.5, "end": 20.9, "text": "Ada sedikit batuk tapi tidak terlalu sering, tenggorokan juga agak sakit."},
            {"start": 20.9, "end": 23.4, "text": "Apakah ada mual atau muntah?"},
            {"start": 23.4, "end": 28.0, "text": "Tidak ada dok, tapi nafsu makan saya agak berkurang."}
        ]
        full_text = " ".join(s["text"] for s in segments)
        res = self.extractor.extract(full_text, segments=segments)
        symptoms = {s["name"].lower(): s["status"] for s in res.get("symptoms", [])}

        self.assertEqual(symptoms.get("sakit tenggorokan"), "present",
                         "Sakit tenggorokan harus diekstrak sebagai present")
        self.assertEqual(symptoms.get("batuk"), "present",
                         "Batuk harus diekstrak sebagai present")
        self.assertEqual(symptoms.get("mual"), "absent",
                         "Mual yang disangkal harus diekstrak sebagai absent")
        self.assertEqual(symptoms.get("muntah"), "absent",
                         "Muntah yang disangkal harus diekstrak sebagai absent")
        self.assertIn("sakit tenggorokan", res["keluhan_tambahan"].lower(),
                         "Sakit tenggorokan harus muncul di keluhan tambahan")

    # ── Rule 13: Parsing Numerik Durasi / Lama Sakit (Thn / Bln / Hari) ─────────
    def test_numeric_duration_parsing(self):
        """Memverifikasi parsing durasi klinis ke nilai angka Tahun, Bulan, Hari (CoreERP EMR)."""
        # Kasus 1: "Sejak kemarin sore" -> 1 hari
        res1 = self.extractor.extract("Dokter: Sakit sejak kapan? Pasien: Sejak kemarin sore dok.")
        dur1 = res1.get("duration")
        self.assertIsNotNone(dur1)
        self.assertEqual(dur1["days"], 1)
        self.assertEqual(dur1["thn"], "")
        self.assertEqual(dur1["bln"], "")
        self.assertEqual(dur1["hari"], "1")
        self.assertEqual(res1["fields"]["lama_sakit_hari"], "1")

        # Kasus 2: "Sudah 3 hari" -> 3 hari
        res2 = self.extractor.extract("Pasien: Saya pusing sudah 3 hari.")
        dur2 = res2.get("duration")
        self.assertEqual(dur2["days"], 3)
        self.assertEqual(dur2["hari"], "3")

        # Kasus 3: "2 minggu" -> 14 hari
        res3 = self.extractor.extract("Pasien: Batuk sudah 2 minggu dok.")
        dur3 = res3.get("duration")
        self.assertEqual(dur3["days"], 14)
        self.assertEqual(dur3["hari"], "14")

        # Kasus 4: "1 bulan" -> 1 bulan
        res4 = self.extractor.extract("Pasien: Sakit pinggang sudah 1 bulan.")
        dur4 = res4.get("duration")
        self.assertEqual(dur4["months"], 1)
        self.assertEqual(dur4["bln"], "1")

        # Kasus 5: "1 tahun 2 bulan" -> 1 tahun, 2 bulan
        res5 = self.extractor.extract("Pasien: Keluhan ini sudah 1 tahun 2 bulan dok.")
        dur5 = res5.get("duration")
        self.assertEqual(dur5["years"], 1)
        self.assertEqual(dur5["months"], 2)
        self.assertEqual(dur5["thn"], "1")
        self.assertEqual(dur5["bln"], "2")

    # ── Rule 14: Pemisahan Giliran Sapaan Awal & Durasi Relatif Tahun/Bulan Kemarin ──
    def test_greeting_turn_split_and_relative_duration(self):
        """'Hai malam dok, malam bu, keluhannya apa ya bu?' terpisah menjadi Pasien dan Dokter,
        dan 'saya pilek dari tahun kemarin' terekstrak durasi tahun kemarin [1] Thn."""
        segs = [
            {"start": 0.2, "end": 5.5, "text": "Hai malam dok, malam bu, keluhannya apa ya bu?"},
            {"start": 5.5, "end": 10.0, "text": "Ini dok, saya pilek dari tahun kemarin."},
            {"start": 10.0, "end": 14.8, "text": "Dan ini juga ada panas dari bulan kemarin."},
            {"start": 14.8, "end": 17.8, "text": "Kira-kira penyebabnya apa."},
            {"start": 17.8, "end": 18.6, "text": "Ya dok?"}
        ]
        full_text = " ".join(s["text"] for s in segs)
        res = self.extractor.extract(full_text, segments=segs)

        # 1. Verifikasi pemisahan giliran dialog segmen 1
        labeled = res["labeled_segments"]
        self.assertEqual(labeled[0]["speaker"], "Pasien")
        self.assertIn("hai malam dok", labeled[0]["text"].lower())

        self.assertEqual(labeled[1]["speaker"], "Dokter")
        self.assertIn("keluhannya apa", labeled[1]["text"].lower())

        # 2. Verifikasi keluhan dan durasi
        self.assertEqual(res["keluhan_utama"], "Pilek")
        self.assertIn("panas", res["keluhan_tambahan"].lower())
        self.assertEqual(res["lama_sakit"], "Dari Tahun Kemarin")
        self.assertEqual(res["fields"]["lama_sakit_thn"], "1")
        self.assertEqual(res["fields"]["lama_sakit_bln"], "")
        self.assertEqual(res["fields"]["lama_sakit_hari"], "")
        self.assertEqual(res["duration"]["years"], 1)


if __name__ == "__main__":
    unittest.main()



