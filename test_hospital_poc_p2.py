"""
Test Suite P2: Kesiapan Operasional, Integrasi EMR, Keamanan & Evaluasi (Hospital POC)
Memvalidasi:
1. EMR Integration Gatekeeping (Pencegahan pengiriman draft tanpa persetujuan dokter).
2. Lifecycle Review Tenaga Medis (Draft -> Needs Review -> Approved -> EMR Export).
3. Proteksi Idempotency (Mencegah duplikasi data saat pengiriman ulang).
4. Konversi standar HL7 FHIR Bundle (Encounter, Condition, Observation, AllergyIntolerance).
5. Audit Trail & PII Masking (Audit logging tanpa kebocoran PHI).
6. Evaluasi Metrik Kualitas Sistem (WER, CER, F1, Negation Accuracy).
"""

import unittest
from medical_extractor import MedicalComplaintExtractor
from security_audit import audit_logger, AuditAction, mask_pii
from emr_integration import (
    MockHospitalEMRAdapter,
    FHIRAdapter,
    UnapprovedSubmissionError,
    EMRTransmissionError
)
from evaluation import (
    calculate_wer,
    calculate_cer,
    calculate_clinical_extraction_metrics,
    calculate_negation_accuracy,
    run_benchmark_suite
)


class TestHospitalPOCP2(unittest.TestCase):

    def setUp(self):
        self.extractor = MedicalComplaintExtractor()
        self.emr_adapter = MockHospitalEMRAdapter(simulate_network_delay=0.0)
        self.fhir_adapter = FHIRAdapter(self.emr_adapter)

    # ── 1. EMR Gatekeeping & Human Approval Check ────────────────────────────
    def test_unapproved_submission_blocked(self):
        # Data yang masih berstatus 'draft' dilarang dikirim ke EMR
        draft_payload = {
            "session_metadata": {"session_id": "TEST-UNAPPROVED-01"},
            "chief_complaint": {"value": "Demam", "source": "patient"},
            "review_status": "draft"
        }

        with self.assertRaises(UnapprovedSubmissionError):
            self.emr_adapter.send_emr(draft_payload, actor="system")

    def test_approved_submission_succeeds(self):
        # Setelah disetujui (approved) oleh dokter, transmisi berhasil
        approved_payload = {
            "session_metadata": {"session_id": "TEST-APPROVED-01"},
            "chief_complaint": {"value": "Demam / Panas", "source": "patient"},
            "review_status": "approved",
            "clinical_validation": {"valid": True, "issues": []}
        }

        result = self.emr_adapter.send_emr(approved_payload, actor="dr_spesialis")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("REC-EMR-", result["emr_record_id"])
        self.assertFalse(result["is_duplicate"])

    # ── 2. Idempotency Protection ───────────────────────────────────────────
    def test_idempotency_prevents_duplicates(self):
        session_id = "TEST-IDEMPOTENT-01"
        payload = {
            "session_metadata": {"session_id": session_id},
            "chief_complaint": {"value": "Batuk kering", "source": "patient"},
            "review_status": "approved",
            "clinical_validation": {"valid": True, "issues": []}
        }

        # Transmisi pertama
        res1 = self.emr_adapter.send_emr(payload, actor="dr_andi")
        self.assertFalse(res1["is_duplicate"])

        # Transmisi ulang dengan ID sesi yang sama (harus idempotent)
        res2 = self.emr_adapter.send_emr(payload, actor="dr_andi")
        self.assertTrue(res2["is_duplicate"], "Pengiriman ulang harus terdeteksi sebagai duplicate/idempotent hit")
        self.assertEqual(res1["emr_record_id"], res2["emr_record_id"], "ID record EMR harus tetap sama")

    # ── 3. HL7 FHIR Bundle Conversion ───────────────────────────────────────
    def test_fhir_bundle_conversion(self):
        payload = {
            "session_metadata": {"session_id": "SESS-FHIR-01"},
            "chief_complaint": {"value": "Nyeri dada", "duration": "Sejak tadi pagi"},
            "vitals": {
                "blood_pressure": {"value": "130/85", "unit": "mmHg", "review_required": False},
                "temperature": {"value": 37.0, "unit": "°C", "review_required": False}
            },
            "allergies": {
                "status": "reported",
                "items": ["Amoxicillin"]
            },
            "review_status": "approved",
            "clinical_validation": {"valid": True, "issues": []}
        }

        bundle = self.fhir_adapter.to_fhir_bundle(payload)
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "transaction")

        resource_types = [entry["resource"]["resourceType"] for entry in bundle["entry"]]
        self.assertIn("Encounter", resource_types)
        self.assertIn("Condition", resource_types)
        self.assertIn("Observation", resource_types)
        self.assertIn("AllergyIntolerance", resource_types)

    # ── 4. Security Audit Trail & PII Masking ───────────────────────────────
    def test_pii_masking(self):
        text_with_pii = "Pasien bernama Budi NIK 3201012345678001 nomor telpon 081234567890 lahir 12-05-1990."
        masked = mask_pii(text_with_pii)

        self.assertNotIn("3201012345678001", masked)
        self.assertIn("[NIK-REDACTED]", masked)
        self.assertNotIn("081234567890", masked)
        self.assertIn("[PHONE-REDACTED]", masked)

    def test_audit_logger_records_and_retrieves(self):
        test_session = "AUDIT-TEST-SESSION-99"
        rec = audit_logger.log_event(
            session_id=test_session,
            action=AuditAction.STAFF_APPROVED,
            actor="dr_santoso",
            role="doctor",
            status="SUCCESS",
            details={"notes": "Verifikasi klinis selesai"}
        )

        self.assertEqual(rec["action"], AuditAction.STAFF_APPROVED)
        self.assertIn("checksum", rec)

        history = audit_logger.get_session_history(test_session)
        self.assertTrue(len(history) >= 1)
        self.assertEqual(history[-1]["session_id"], test_session)

    # ── 5. System Evaluation Metrics ────────────────────────────────────────
    def test_evaluation_metrics_calculations(self):
        # WER & CER
        wer = calculate_wer("saya panas sejak kemarin", "saya tanas sejak kemarin")
        self.assertEqual(wer, 0.25)  # 1 dari 4 kata salah

        cer = calculate_cer("panas", "tanas")
        self.assertEqual(cer, 0.2)   # 1 dari 5 karakter salah

        # Extraction F1
        metrics = calculate_clinical_extraction_metrics(["demam", "batuk"], ["demam", "batuk"])
        self.assertEqual(metrics["f1"], 1.0)

        # Negation accuracy
        neg_acc = calculate_negation_accuracy(
            [("batuk", True), ("pilek", False)],
            [{"name": "Batuk", "status": "absent"}, {"name": "Pilek", "status": "present"}]
        )
        self.assertEqual(neg_acc, 1.0)

    def test_benchmark_suite_execution(self):
        report = run_benchmark_suite(self.extractor)
        self.assertEqual(report["cases_evaluated"], 3)
        self.assertGreaterEqual(report["mean_symptom_f1"], 0.90)
        self.assertGreaterEqual(report["mean_negation_accuracy"], 0.90)
        self.assertGreaterEqual(report["mean_vitals_accuracy"], 0.90)


if __name__ == "__main__":
    unittest.main()
