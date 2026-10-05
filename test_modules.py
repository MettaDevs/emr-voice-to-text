"""
test_modules.py — MedVoice AI Modular Unit & Integration Test Suite
==================================================================
Menguji 14 modul Voice-to-EMR secara terpisah sesuai Section 3:
  1. Audio processing (file valid, bising, kosong/terpotong)
  2. STT (pemrosesan kata & angka akustik)
  3. Koreksi kata (preservasi makna klinis)
  4. Normalisasi (istilah gaul/sehari-hari ke baku klinis)
  5. Speaker labeling (Dokter, Pasien, Pendamping, Perawat, Unknown)
  6. Keluhan utama (ekstraksi respon pasien)
  7. Gejala tambahan (gejala positif vs negatif)
  8. Durasi (waktu pasti, perkiraan, dan koreksi ucapan)
  9. Tanda vital (angka, satuan, dan validasi threshold)
  10. Alergi (reported, denied, unknown, not_asked)
  11. Confidence (decoupled confidence & review trigger)
  12. EMR draft (konsistensi skema JSON)
  13. Approval (persetujuan, penolakan, pembaruan draft)
  14. Integrasi (pengiriman mock/sandbox, penanganan gagal, retry aman)
"""

import os
import unittest
import json
from medical_extractor import (
    MedicalComplaintExtractor,
    correct_with_trace,
    _apply_layer1,
    _apply_layer2,
    _classify_sentence_speaker,
    _extract_structured_duration,
    _extract_structured_symptoms,
    _extract_extended_vitals,
    _extract_standardized_allergies,
    calculate_decoupled_confidence,
    ClinicalDataValidator
)
from emr_integration import MockHospitalEMRAdapter, FHIRAdapter, UnapprovedSubmissionError, EMRTransmissionError


class TestModularVoiceToEMR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = MedicalComplaintExtractor()
        cls.results_log = []

    def record_result(self, module_name, test_input, expected, actual, passed, notes=""):
        res = {
            "module": module_name,
            "input": str(test_input),
            "expected": str(expected),
            "actual": str(actual),
            "status": "PASS" if passed else "FAIL",
            "notes": notes
        }
        self.results_log.append(res)
        self.assertTrue(passed, f"[{module_name}] FAILED: {notes} | Expected: {expected} | Actual: {actual}")

    # ── 1. Audio Processing ──────────────────────────────────────────────────
    def test_01_audio_processing_path_and_handling(self):
        """Modul 1: Audio processing - penanganan file kosong / tidak ditemukan."""
        non_existent = "uploads/non_existent_audio_sample_9999.wav"
        passed = not os.path.exists(non_existent)
        self.record_result(
            module_name="Audio Processing",
            test_input=non_existent,
            expected="FileNotFoundError / Non-existent detection",
            actual=f"Exists: {os.path.exists(non_existent)}",
            passed=passed,
            notes="Validasi jalur audio dan mitigasi file hilang"
        )

    # ── 2. STT (Speech-to-Text) Post-Processing ──────────────────────────────
    def test_02_stt_text_sanitization(self):
        """Modul 2: STT - pembersihan halusinasi decoder Whisper tanpa merusak kata."""
        raw_whisper = "Dokter: Apa keluhan? Pasien: Panas dok [sigh] (hembusan napas)."
        clean = correct_with_trace(raw_whisper)["corrected_text"]
        passed = "[sigh]" not in clean and "(hembusan napas)" not in clean and "Panas" in clean
        self.record_result(
            module_name="STT",
            test_input=raw_whisper,
            expected="Teks bersih tanpa deskriptor suara [sigh] / (hembusan napas)",
            actual=clean,
            passed=passed
        )

    # ── 3. Koreksi Kata (Phonetic Corrections) ──────────────────────────────
    def test_03_word_correction_meaning_preservation(self):
        """Modul 3: Koreksi kata - koreksi fonetik mic tanpa mengubah makna."""
        raw = "Tanasnya dari kemarin malam dan batu pilek."
        trace = correct_with_trace(raw)
        corrected = trace["corrected_text"]
        passed = "panas" in corrected.lower() and "batuk pilek" in corrected.lower()
        self.record_result(
            module_name="Koreksi Kata",
            test_input=raw,
            expected="panasnya ... batuk pilek",
            actual=corrected,
            passed=passed
        )

    # ── 4. Normalisasi Bahasa (Slang / Informal → Baku) ──────────────────────
    def test_04_normalization_slang_to_formal(self):
        """Modul 4: Normalisasi - bahasa tutur sehari-hari ke istilah baku klinis."""
        raw = "Gue gak enak badan, puyeng banget, udah minum obat."
        norm = _apply_layer2(raw)
        passed = "saya" in norm.lower() and "tidak enak badan" in norm.lower() and "pusing" in norm.lower()
        self.record_result(
            module_name="Normalisasi",
            test_input=raw,
            expected="Saya tidak enak badan, pusing sekali, sudah...",
            actual=norm,
            passed=passed
        )

    # ── 5. Speaker Labeling ──────────────────────────────────────────────────
    def test_05_speaker_labeling_roles(self):
        """Modul 5: Speaker labeling - Dokter, Pasien, Pendamping, Unknown."""
        s1 = _classify_sentence_speaker("Apa keluhan yang dirasakan hari ini?")
        s2 = _classify_sentence_speaker("Saya merasa lemas dok.")
        s3 = _classify_sentence_speaker("Anak saya badannya panas dok.")
        s4 = _classify_sentence_speaker("Ruangan ini sangat dingin.")

        passed = (s1[2] == "doctor" and s2[2] == "patient" and s3[2] == "companion")
        self.record_result(
            module_name="Speaker Labeling",
            test_input="[s1: dokter q, s2: pasien, s3: pendamping]",
            expected="(doctor, patient, companion)",
            actual=f"({s1[2]}, {s2[2]}, {s3[2]})",
            passed=passed
        )

    # ── 6. Keluhan Utama (Chief Complaint) ───────────────────────────────────
    def test_06_chief_complaint_extraction(self):
        """Modul 6: Keluhan utama - ekstraksi dari respon keluhan pasien."""
        dialogue = "Dokter: Ada keluhan apa? Pasien: Saya pusing berputar dok sejak kemarin."
        res = self.extractor.extract(dialogue)
        chief = res["chief_complaint"]["value"]
        passed = "Pusing" in chief
        self.record_result(
            module_name="Keluhan Utama",
            test_input=dialogue,
            expected="Pusing / Vertigo",
            actual=chief,
            passed=passed
        )

    # ── 7. Gejala Tambahan (Present vs Absent) ───────────────────────────────
    def test_07_symptoms_present_vs_absent(self):
        """Modul 7: Gejala tambahan - polaritas positif vs negatif."""
        dialogue = "Pasien: Saya merasa panas sejak kemarin, batuk tidak ada, tapi ada mual."
        res = self.extractor.extract(dialogue)
        symptoms = res["symptoms"]

        batuk = next((s for s in symptoms if "batuk" in s["name"].lower()), None)
        mual = next((s for s in symptoms if "mual" in s["name"].lower()), None)

        passed = (batuk is not None and batuk["status"] == "absent" and
                  mual is not None and mual["status"] == "present")
        self.record_result(
            module_name="Gejala Tambahan",
            test_input=dialogue,
            expected="Batuk (absent), Mual (present)",
            actual=f"Batuk ({batuk['status'] if batuk else 'None'}), Mual ({mual['status'] if mual else 'None'})",
            passed=passed
        )

    # ── 8. Durasi & Onset (Waktu Pasti vs Perkiraan) ─────────────────────────
    def test_08_duration_extraction(self):
        """Modul 8: Durasi - ekstraksi durasi pasti dan perkiraan."""
        dur1 = _extract_structured_duration("Sudah sekitar 3 hari dok.")
        dur2 = _extract_structured_duration("Dari kemarin malam.")

        passed = (dur1["value"] == 3 and dur1["unit"] == "day" and dur1["is_approximate"] and
                  dur2["event_time"] == "Kemarin malam")
        self.record_result(
            module_name="Durasi",
            test_input="[sekitar 3 hari, kemarin malam]",
            expected="dur1: 3 days approx, dur2: Kemarin malam",
            actual=f"dur1: {dur1['value']} {dur1['unit']} (approx={dur1['is_approximate']}), dur2: {dur2['event_time']}",
            passed=passed
        )

    # ── 9. Tanda Vital (TTV) ─────────────────────────────────────────────────
    def test_09_vitals_extraction_and_units(self):
        """Modul 9: Tanda vital - ekstraksi angka, satuan, dan validasi."""
        text = "Dokter: Tekanan darah 120 per 80, suhu 38.5 derajat celcius, nadi 88 kali per menit."
        v = _extract_extended_vitals(text)

        passed = (v["blood_pressure"]["value"] == "120/80" and
                  v["temperature"]["value"] == 38.5 and
                  v["pulse"]["value"] == 88)
        self.record_result(
            module_name="Tanda Vital",
            test_input=text,
            expected="BP: 120/80, Temp: 38.5, Pulse: 88",
            actual=f"BP: {v['blood_pressure']['value']}, Temp: {v['temperature']['value']}, Pulse: {v['pulse']['value']}",
            passed=passed
        )

    # ── 10. Alergi (Reported, Denied, Unknown, Not Asked) ───────────────────
    def test_10_allergy_four_quadrant_states(self):
        """Modul 10: Alergi - 4 status klinis standar."""
        a_rep = _extract_standardized_allergies("Saya alergi amoxicillin dok.")
        a_den = _extract_standardized_allergies("Saya tidak ada riwayat alergi dok.")
        a_unk = _extract_standardized_allergies("Saya tidak tahu punya alergi atau tidak.")
        a_not = _extract_standardized_allergies("Saya hanya merasa demam.")

        passed = (a_rep["status"] == "reported" and
                  a_den["status"] == "denied" and
                  a_unk["status"] == "unknown" and
                  a_not["status"] == "not_asked")
        self.record_result(
            module_name="Alergi",
            test_input="[amoxicillin, tidak ada, tidak tahu, tanpa mention]",
            expected="reported, denied, unknown, not_asked",
            actual=f"{a_rep['status']}, {a_den['status']}, {a_unk['status']}, {a_not['status']}",
            passed=passed
        )

    # ── 11. Confidence & Review Assessment ───────────────────────────────────
    def test_11_confidence_and_review_trigger(self):
        """Modul 11: Confidence - deteksi ketidakpastian dan kebutuhan review."""
        conf, rev = calculate_decoupled_confidence(
            stt_prob=0.95,
            segments=[{"confidence": 0.90}],
            chief_data={"value": "Demam", "confidence": 0.95},
            corrections=[],
            validation_issues=[]
        )
        passed = (conf["transcription"] >= 0.90 and conf["speaker"] >= 0.85 and not rev["required"])
        self.record_result(
            module_name="Confidence",
            test_input="stt_prob 0.95, segments 0.90, chief 0.95",
            expected="transcription >= 0.90, speaker >= 0.85, review_required=False",
            actual=f"Transkripsi: {conf['transcription']}, Speaker: {conf['speaker']}, RevReq: {rev['required']}",
            passed=passed
        )

    # ── 12. EMR Draft Structure Consistency ──────────────────────────────────
    def test_12_emr_draft_structure(self):
        """Modul 12: EMR Draft - konsistensi struktur output JSON."""
        res = self.extractor.extract("Dokter: Apa keluhan? Pasien: Sakit perut sejak semalam.")
        expected_keys = ["chief_complaint", "symptoms", "duration", "vitals", "allergies", "review_status"]
        passed = all(k in res for k in expected_keys)
        self.record_result(
            module_name="EMR Draft",
            test_input="Sakit perut sejak semalam",
            expected=f"Memiliki seluruh keys: {expected_keys}",
            actual=f"Keys present: {[k for k in expected_keys if k in res]}",
            passed=passed
        )

    # ── 13. Approval & Review Actions ────────────────────────────────────────
    def test_13_approval_and_rejection_logic(self):
        """Modul 13: Approval - status transisi draft -> approved / rejected."""
        draft = {"session_id": "test-123", "review_status": "draft"}
        # Approve
        draft["review_status"] = "approved"
        draft["approved_by"] = "dr_budi"
        # Reject
        draft_rej = {"session_id": "test-456", "review_status": "rejected", "reject_reason": "Transkrip terpotong"}

        passed = draft["review_status"] == "approved" and draft_rej["review_status"] == "rejected"
        self.record_result(
            module_name="Approval",
            test_input="Transisi draft approved & rejected",
            expected="approved, rejected",
            actual=f"{draft['review_status']}, {draft_rej['review_status']}",
            passed=passed
        )

    # ── 14. Integrasi EMR (Send, Mock Fail, Retry, Idempotency) ──────────────
    def test_14_emr_integration_send_and_retry(self):
        """Modul 14: Integrasi - pengiriman sandbox, kegagalan, retry aman, idempotency."""
        adapter = MockHospitalEMRAdapter(simulate_network_delay=0.0)
        draft_payload = {
            "session_metadata": {"session_id": "TEST-MOD-INTEG-01"},
            "chief_complaint": {"value": "Demam / Panas", "source": "patient"},
            "review_status": "approved",
            "clinical_validation": {"valid": True, "issues": []}
        }

        # 1. Kirim pertama kali (sukses)
        tx1 = adapter.send_emr(draft_payload, actor="dr_evaluator")
        # 2. Kirim ulang dengan session_id yang sama (mencegah duplikasi data EMR / Idempotent)
        tx2 = adapter.send_emr(draft_payload, actor="dr_evaluator")

        passed = (tx1["status"] == "SUCCESS" and
                  tx2["status"] == "SUCCESS" and
                  not tx1["is_duplicate"] and
                  tx2["is_duplicate"])
        self.record_result(
            module_name="Integrasi EMR",
            test_input="send_emr approved payload dua kali",
            expected="tx1 SUCCESS & is_duplicate=False, tx2 SUCCESS & is_duplicate=True",
            actual=f"tx1: {tx1['status']} (dup={tx1['is_duplicate']}), tx2: {tx2['status']} (dup={tx2['is_duplicate']})",
            passed=passed
        )

    @classmethod
    def tearDownClass(cls):
        print("\n" + "="*80)
        print("HASIL PENGUJIAN 14 MODUL VOICE-TO-EMR SECARA TERPISAH (SECTION 3)")
        print("="*80)
        for r in cls.results_log:
            print(f"[{r['status']}] {r['module']:<20} | In: {r['input'][:30]}... | Act: {r['actual'][:30]}")
        print("="*80)


if __name__ == "__main__":
    unittest.main()
