"""
EMR Integration Abstraction & Adapters (Hospital POC)
Memisahkan pipeline ekstraksi dari pengiriman EMR, memvalidasi schema sebelum transmisi,
mencegah pengiriman otomatis tanpa persetujuan dokter (Human-in-the-Loop),
mencegah duplikasi data (idempotency), dan mendukung format standar HL7 FHIR.
"""

import time
import uuid
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from security_audit import audit_logger, AuditAction, mask_pii


class EMRTransmissionError(Exception):
    """Exception khusus untuk kegagalan transmisi ke EMR eksternal."""
    pass


class UnapprovedSubmissionError(Exception):
    """Exception jika pengiriman EMR dicoba sebelum ada persetujuan dokter."""
    pass


def validate_emr_payload_schema(payload: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validasi schema payload EMR sebelum dikirim ke sistem rumah sakit.
    Memastikan field penting terisi dan tipe data valid.
    """
    errors = []
    if not isinstance(payload, dict):
        return False, ["Payload must be a dictionary"]

    # 1. Cek review_status
    status = payload.get("review_status")
    if not status:
        errors.append("Field 'review_status' wajib ada.")
    elif status != "approved":
        errors.append(f"Data tidak dapat dikirim ke EMR karena status adalah '{status}'. Hanya status 'approved' yang diizinkan.")

    # 2. Cek metadata sesi
    meta = payload.get("session_metadata", {})
    if not meta.get("session_id"):
        errors.append("Field 'session_metadata.session_id' tidak boleh kosong.")

    # 3. Cek keluhan utama
    chief = payload.get("chief_complaint")
    if not chief or not isinstance(chief, dict):
        errors.append("Field 'chief_complaint' harus berupa dictionary.")
    elif not chief.get("value") and not payload.get("keluhan_utama"):
        errors.append("Keluhan utama tidak boleh kosong saat pengiriman ke EMR.")

    # 4. Validasi klinis internal
    val_res = payload.get("clinical_validation", {})
    if val_res and not val_res.get("valid", True):
        # Cek apakah ada error kritis yang belum diselesaikan
        criticals = [i for i in val_res.get("issues", []) if i.get("severity") == "error"]
        if criticals:
            errors.append(f"Terdapat {len(criticals)} issue validasi klinis berkategori 'error' yang belum diselesaikan.")

    return len(errors) == 0, errors


class BaseEMRAdapter(ABC):
    """Interface / Abstraction untuk seluruh adapter EMR rumah sakit."""

    @abstractmethod
    def send_emr(self, payload: Dict[str, Any], actor: str = "doctor") -> Dict[str, Any]:
        """Mengirim data draft yang telah disetujui ke EMR."""
        pass

    @abstractmethod
    def get_sync_status(self, session_id: str) -> Dict[str, Any]:
        """Memeriksa status sinkronisasi suatu sesi EMR."""
        pass


class MockHospitalEMRAdapter(BaseEMRAdapter):
    """
    Adapter EMR Simulasi Rumah Sakit dengan:
    - Enforced Doctor Approval Check
    - Pre-submission schema validation
    - Idempotency key protection
    - Safe retry mechanism
    - Audit logging
    """

    def __init__(self, simulate_network_delay: float = 0.05, max_retries: int = 3):
        self.simulate_network_delay = simulate_network_delay
        self.max_retries = max_retries
        # In-memory storage simulasi EMR database (keyed by session_id)
        self._emr_database: Dict[str, Dict[str, Any]] = {}
        # In-memory idempotency register
        self._transmission_log: Dict[str, Dict[str, Any]] = {}

    def send_emr(self, payload: Dict[str, Any], actor: str = "doctor") -> Dict[str, Any]:
        session_id = payload.get("session_metadata", {}).get("session_id") or payload.get("session_id", "UNKNOWN")

        # 1. Audit awal upaya pengiriman
        audit_logger.log_event(
            session_id=session_id,
            action=AuditAction.EMR_EXPORT_ATTEMPTED,
            actor=actor,
            role="medical_staff",
            status="PENDING",
            details={"review_status": payload.get("review_status")}
        )

        # 2. Cek Human-in-the-Loop Approval
        if payload.get("review_status") != "approved":
            err_msg = f"Submission ditolak: status EMR saat ini '{payload.get('review_status')}'. Wajib diverifikasi dan disetujui (approved) oleh dokter terlebih dahulu."
            audit_logger.log_event(
                session_id=session_id,
                action=AuditAction.EMR_EXPORT_FAILED,
                actor=actor,
                role="medical_staff",
                status="REJECTED_UNAPPROVED",
                details={"reason": err_msg}
            )
            raise UnapprovedSubmissionError(err_msg)

        # 3. Validasi Schema Sebelum Pengiriman
        is_valid, validation_errors = validate_emr_payload_schema(payload)
        if not is_valid:
            err_msg = f"Schema validation error: {'; '.join(validation_errors)}"
            audit_logger.log_event(
                session_id=session_id,
                action=AuditAction.EMR_EXPORT_FAILED,
                actor=actor,
                role="medical_staff",
                status="SCHEMA_INVALID",
                details={"errors": validation_errors}
            )
            raise EMRTransmissionError(err_msg)

        # 4. Idempotency Check (Mencegah submit ganda dari pengiriman ulang)
        if session_id in self._transmission_log:
            existing = self._transmission_log[session_id]
            audit_logger.log_event(
                session_id=session_id,
                action=AuditAction.EMR_EXPORT_SUCCESS,
                actor=actor,
                role="medical_staff",
                status="IDEMPOTENT_HIT",
                details={"info": "Data sudah pernah terkirim sebelumnya, mengembalikan record yang ada."}
            )
            return {
                "status": "SUCCESS",
                "message": "Data sudah terdaftar sebelumnya (idempotent submission).",
                "emr_record_id": existing["emr_record_id"],
                "sync_timestamp": existing["sync_timestamp"],
                "is_duplicate": True
            }

        # 5. Simulasi Transmisi dengan Retry
        attempts = 0
        last_exception = None
        while attempts < self.max_retries:
            attempts += 1
            try:
                if self.simulate_network_delay > 0:
                    time.sleep(self.simulate_network_delay)

                # Generate record ID di database EMR
                emr_record_id = f"REC-EMR-{uuid.uuid4().hex[:8].upper()}"
                sync_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

                record = {
                    "emr_record_id": emr_record_id,
                    "session_id": session_id,
                    "approved_by": actor,
                    "sync_timestamp": sync_timestamp,
                    "payload_data": payload
                }

                # Simpan ke storage simulasi
                self._emr_database[session_id] = record
                self._transmission_log[session_id] = {
                    "emr_record_id": emr_record_id,
                    "sync_timestamp": sync_timestamp,
                    "attempts": attempts
                }

                # Catat audit trail sukses
                audit_logger.log_event(
                    session_id=session_id,
                    action=AuditAction.EMR_EXPORT_SUCCESS,
                    actor=actor,
                    role="medical_staff",
                    status="SUCCESS",
                    details={"emr_record_id": emr_record_id, "attempts": attempts}
                )

                return {
                    "status": "SUCCESS",
                    "message": "Data rekam medis berhasil dikirim ke sistem EMR Rumah Sakit.",
                    "emr_record_id": emr_record_id,
                    "sync_timestamp": sync_timestamp,
                    "is_duplicate": False
                }

            except Exception as e:
                last_exception = e
                time.sleep(0.05 * attempts)

        # Jika seluruh percobaan gagal
        err_sanitized = mask_pii(str(last_exception) if last_exception else "Max retries reached")
        audit_logger.log_event(
            session_id=session_id,
            action=AuditAction.EMR_EXPORT_FAILED,
            actor=actor,
            role="medical_staff",
            status="FAILED",
            details={"attempts": attempts, "error": err_sanitized}
        )
        raise EMRTransmissionError(f"Gagal mengirim ke EMR setelah {attempts} percobaan: {err_sanitized}")

    def get_sync_status(self, session_id: str) -> Dict[str, Any]:
        if session_id in self._transmission_log:
            tx = self._transmission_log[session_id]
            return {
                "synced": True,
                "session_id": session_id,
                "emr_record_id": tx["emr_record_id"],
                "sync_timestamp": tx["sync_timestamp"]
            }
        return {
            "synced": False,
            "session_id": session_id,
            "message": "Data belum pernah disinkronkan ke EMR."
        }


class FHIRAdapter(BaseEMRAdapter):
    """
    Adapter Konversi HL7 FHIR Bundle (R4/R5 standard format).
    Mengubah payload Voice-to-EMR menjadi FHIR Bundle yang berisi:
    - Encounter
    - Condition (Keluhan Utama & Gejala Tambahan)
    - Observation (Tanda-tanda Vital)
    - AllergyIntolerance (Status Alergi)
    """

    def __init__(self, mock_adapter: Optional[MockHospitalEMRAdapter] = None):
        self.mock_adapter = mock_adapter or MockHospitalEMRAdapter()

    def to_fhir_bundle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mengonversi payload terstruktur ke standar HL7 FHIR Bundle."""
        session_id = payload.get("session_metadata", {}).get("session_id", f"ENC-{uuid.uuid4().hex[:6]}")
        bundle_id = f"bundle-{uuid.uuid4().hex[:8]}"

        entries = []

        # 1. Encounter Resource
        encounter_id = f"enc-{session_id}"
        entries.append({
            "fullUrl": f"urn:uuid:{encounter_id}",
            "resource": {
                "resourceType": "Encounter",
                "id": encounter_id,
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB",
                    "display": "ambulatory"
                },
                "serviceType": {
                    "text": "Konsultasi Rawat Jalan / Voice EMR"
                }
            }
        })

        # 2. Condition Resource: Chief Complaint
        chief = payload.get("chief_complaint", {})
        chief_val = chief.get("value") or payload.get("keluhan_utama")
        if chief_val:
            cond_id = f"cond-chief-{uuid.uuid4().hex[:6]}"
            entries.append({
                "fullUrl": f"urn:uuid:{cond_id}",
                "resource": {
                    "resourceType": "Condition",
                    "id": cond_id,
                    "clinicalStatus": {
                        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
                    },
                    "category": [{
                        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis", "display": "Chief Complaint"}]
                    }],
                    "code": {"text": chief_val},
                    "subject": {"display": "Patient"},
                    "encounter": {"reference": f"urn:uuid:{encounter_id}"},
                    "onsetAge": {"text": chief.get("duration") or ""} if chief.get("duration") else None
                }
            })

        # 3. Observation Resources: Vital Signs
        vitals = payload.get("vitals", {})
        if vitals.get("blood_pressure") and vitals["blood_pressure"].get("value"):
            bp_val = vitals["blood_pressure"]["value"]
            entries.append({
                "fullUrl": f"urn:uuid:obs-bp-{uuid.uuid4().hex[:6]}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "preliminary" if vitals["blood_pressure"].get("review_required") else "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure panel"}]},
                    "valueString": f"{bp_val} mmHg"
                }
            })

        if vitals.get("temperature") and vitals["temperature"].get("value"):
            temp_val = vitals["temperature"]["value"]
            entries.append({
                "fullUrl": f"urn:uuid:obs-temp-{uuid.uuid4().hex[:6]}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8310-5", "display": "Body temperature"}]},
                    "valueQuantity": {"value": temp_val, "unit": "Cel", "system": "http://unitsofmeasure.org", "code": "Cel"}
                }
            })

        # 4. AllergyIntolerance Resource
        allergies = payload.get("allergies", {})
        allergy_status = allergies.get("status", "not_asked")
        if allergy_status in ["reported", "denied"]:
            entries.append({
                "fullUrl": f"urn:uuid:allergy-{uuid.uuid4().hex[:6]}",
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "clinicalStatus": {
                        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active" if allergy_status == "reported" else "resolved"}]
                    },
                    "verificationStatus": {
                        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed" if allergy_status == "reported" else "refuted"}]
                    },
                    "note": [{"text": f"Status Alergi Pasien: {allergy_status}. Item: {', '.join(allergies.get('items', [])) or 'None'}"}]
                }
            })

        return {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "transaction",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "entry": entries
        }

    def send_emr(self, payload: Dict[str, Any], actor: str = "doctor") -> Dict[str, Any]:
        # Validasi schema & approval melalui mock adapter
        res = self.mock_adapter.send_emr(payload, actor=actor)
        fhir_bundle = self.to_fhir_bundle(payload)
        res["fhir_bundle"] = fhir_bundle
        return res

    def get_sync_status(self, session_id: str) -> Dict[str, Any]:
        return self.mock_adapter.get_sync_status(session_id)


def get_emr_adapter(adapter_type: str = "mock") -> BaseEMRAdapter:
    """Factory helper untuk mendapatkan adapter EMR yang diinginkan."""
    if adapter_type.lower() == "fhir":
        return FHIRAdapter()
    return MockHospitalEMRAdapter()
