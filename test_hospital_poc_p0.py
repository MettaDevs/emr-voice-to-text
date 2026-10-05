"""
Test Suite P0: Keamanan & Ketepatan Informasi Klinis (Hospital POC)
Memvalidasi:
1. Koreksi fonetik kontekstual dengan trace audit (tanpa merusak nama obat/dosis).
2. Isolasi speaker (Dokter, Pasien, Pendamping, Perawat, Unknown).
3. Preservasi negasi ("Saya panas" vs "Saya tidak panas").
4. Ekstraksi alergi terstandarisasi (reported, denied, unknown, not_asked, uncertain).
5. Clinical Data Validator (validasi pra-EMR).
"""

import unittest
from medical_extractor import (
    correct_with_trace,
    MedicalComplaintExtractor,
    ClinicalDataValidator,
    _classify_sentence_speaker
)


class TestHospitalPOCP0(unittest.TestCase):

    def setUp(self):
        self.extractor = MedicalComplaintExtractor()

    # ── 1. Traceable Phonetic Correction & Protected Entities ───────────────
    def test_phonetic_correction_with_trace(self):
        # "tanas" -> "panas", tetapi "amoxicillin 500 mg" tidak boleh terpotong atau rusak
        input_text = "Saya tanas sejak kemarin dan minum amoxicillin 500 mg."
        trace_res = correct_with_trace(input_text)
        corrected = trace_res["corrected_text"]
        traces = trace_res["corrections"]

        self.assertIn("panas", corrected.lower(), "Kata 'tanas' harus terkoreksi menjadi 'panas'")
        self.assertIn("amoxicillin 500 mg", corrected.lower(), "Nama obat dan dosis harus terlindungi dari mutasi")

        # Cek jejak audit koreksi
        tanas_trace = next((t for t in traces if t["original"].lower() == "tanas"), None)
        self.assertIsNotNone(tanas_trace, "Koreksi 'tanas' harus memiliki log jejak audit")
        self.assertEqual(tanas_trace["corrected"], "panas")
        self.assertIn("confidence", tanas_trace)
        self.assertIn("review_required", tanas_trace)

    def test_protected_drugs_and_dosages_unmutated(self):
        # Pastikan angka tanda vital & obat terlindungi
        prescription = "Berikan paracetamol 500 mg 3x1 sehari dan cek tensi 120/80 mmHg."
        trace_res = correct_with_trace(prescription)
        corrected = trace_res["corrected_text"]
        self.assertIn("paracetamol 500 mg", corrected.lower())
        self.assertIn("120/80 mmhg", corrected.lower())

    # ── 2. Speaker Role Isolation ───────────────────────────────────────────
    def test_multi_role_speaker_detection(self):
        # Dokter
        d_res = _classify_sentence_speaker("Apa keluhan yang dirasakan hari ini?")
        self.assertEqual(d_res[2], "doctor", "Pertanyaan dokter harus teridentifikasi sebagai 'doctor'")

        # Pasien
        p_res = _classify_sentence_speaker("Saya merasa lemas dan pusing dok.")
        self.assertEqual(p_res[2], "patient", "Keluhan pasien harus teridentifikasi sebagai 'patient'")

        # Pendamping
        c_res = _classify_sentence_speaker("Ibu saya mengeluh sesak napas dok.")
        self.assertEqual(c_res[2], "companion", "Tuturan pendamping harus teridentifikasi sebagai 'companion'")

        # Perawat
        n_res = _classify_sentence_speaker("Suster sudah ukur tensi pasien 120/80 ya dok.")
        self.assertEqual(n_res[2], "nurse", "Tuturan perawat harus teridentifikasi sebagai 'nurse'")

    def test_doctor_question_not_extracted_as_patient_symptom(self):
        # Dokter bertanya apakah ada muntah atau sesak, pasien menjawab hanya demam
        dialogue = (
            "Dokter: Apakah bapak ada muntah atau sesak napas? "
            "Pasien: Tidak ada dok, saya cuma demam sejak kemarin."
        )
        res = self.extractor.extract(dialogue)

        # Keluhan utama harus demam
        chief_val = res["chief_complaint"]["value"] or res["keluhan_utama"]
        self.assertIn("Demam", chief_val)

        # Gejala muntah dan sesak napas dari pertanyaan dokter TIDAK boleh masuk sebagai keluhan utama
        self.assertNotIn("muntah", chief_val.lower())
        self.assertNotIn("sesak", chief_val.lower())

    # ── 3. Negation Preservation ────────────────────────────────────────────
    def test_symptom_negation_accuracy(self):
        # Kasus A: Pasien menyatakan panas
        res_present = self.extractor.extract("Pasien: Badan saya terasa panas sejak kemarin.")
        symptoms_present = [s["name"].lower() for s in res_present["symptoms"] if s["status"] == "present"]
        self.assertTrue(any("panas" in s or "demam" in s for s in symptoms_present))

        # Kasus B: Pasien menyangkal panas
        res_negated = self.extractor.extract("Pasien: Saya tidak panas sama sekali dok, cuma batuk.")
        symptoms_present_neg = [s["name"].lower() for s in res_negated["symptoms"] if s["status"] == "present"]
        symptoms_absent_neg = [s["name"].lower() for s in res_negated["symptoms"] if s["status"] == "absent"]

        self.assertFalse(any("panas" in s for s in symptoms_present_neg), "Panas yang dinegasikan tidak boleh berstatus 'present'")
        self.assertTrue(any("panas" in s for s in symptoms_absent_neg), "Panas yang dinegasikan harus berstatus 'absent'")

    # ── 4. Standardized Allergy Status ──────────────────────────────────────
    def test_allergy_denied(self):
        res = self.extractor.extract("Pasien: Saya tidak punya riwayat alergi obat maupun makanan dok.")
        allergy = res["allergies"]
        self.assertEqual(allergy["status"], "denied", "Status alergi harus 'denied'")
        self.assertEqual(len(allergy["items"]), 0)

    def test_allergy_reported_with_item(self):
        res = self.extractor.extract("Pasien: Saya alergi amoxicillin dan penisilin dok.")
        allergy = res["allergies"]
        self.assertEqual(allergy["status"], "reported", "Status alergi harus 'reported'")
        self.assertTrue(any("amoxicillin" in i.lower() for i in allergy["items"]))

    def test_allergy_unknown(self):
        res = self.extractor.extract("Pasien: Saya tidak tahu apakah punya alergi obat atau tidak.")
        allergy = res["allergies"]
        self.assertEqual(allergy["status"], "unknown", "Status alergi harus 'unknown'")

    def test_allergy_not_asked(self):
        res = self.extractor.extract("Pasien: Saya pusing dan batuk sejak kemarin malam.")
        allergy = res["allergies"]
        self.assertEqual(allergy["status"], "not_asked", "Jika tidak dibahas, status alergi harus 'not_asked'")

    # ── 5. Clinical Data Validator ──────────────────────────────────────────
    def test_clinical_data_validator_rules(self):
        # Payload valid
        valid_payload = {
            "chief_complaint": {"value": "Demam", "source": "patient"},
            "vitals": {
                "blood_pressure": {"value": "120/80", "unit": "mmHg", "review_required": False}
            },
            "allergies": {"status": "denied"},
            "duration": {"original_text": "3 hari", "value": 3, "unit": "day"}
        }
        val_res = ClinicalDataValidator.validate(valid_payload)
        self.assertTrue(val_res["valid"], "Payload valid harus lolos validasi klinis")

        # Payload invalid (Tensi di luar batas fisiologis wajar: 320/210)
        invalid_payload = {
            "chief_complaint": {"value": "Pusing", "source": "patient"},
            "vitals": {
                "blood_pressure": {"value": "320/210", "unit": "mmHg", "review_required": True}
            },
            "allergies": {"status": "not_asked"},
            "duration": None
        }
        val_res_inv = ClinicalDataValidator.validate(invalid_payload)
        self.assertFalse(val_res_inv["valid"], "Tensi 320/210 harus memicu error validasi klinis")
        self.assertTrue(any("blood_pressure" in i["field"] for i in val_res_inv["issues"]))


if __name__ == "__main__":
    unittest.main()
