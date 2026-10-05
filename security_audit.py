"""
Security & Audit Trail Module for Hospital Voice-to-EMR (MedVoice AI)
Menyediakan audit trail standar rumah sakit, sanitasi/masking PII (data pasien sensitif),
serta logging aman tanpa membocorkan PHI (Protected Health Information).
"""

import os
import json
import re
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class AuditAction:
    AUDIO_INGESTED = "AUDIO_INGESTED"
    TRANSCRIPTION_COMPLETED = "TRANSCRIPTION_COMPLETED"
    TEXT_CORRECTED = "TEXT_CORRECTED"
    CLINICAL_EXTRACTED = "CLINICAL_EXTRACTED"
    VALIDATION_EXECUTED = "VALIDATION_EXECUTED"
    DRAFT_GENERATED = "DRAFT_GENERATED"
    STAFF_REVIEW_RECORDED = "STAFF_REVIEW_RECORDED"
    STAFF_APPROVED = "STAFF_APPROVED"
    STAFF_REJECTED = "STAFF_REJECTED"
    EMR_EXPORT_ATTEMPTED = "EMR_EXPORT_ATTEMPTED"
    EMR_EXPORT_SUCCESS = "EMR_EXPORT_SUCCESS"
    EMR_EXPORT_FAILED = "EMR_EXPORT_FAILED"


# Regex untuk pola PII (NIK, nomor telepon, tanggal lahir)
_NIK_PATTERN = re.compile(r'\b\d{16}\b')
_PHONE_PATTERN = re.compile(r'\b(?:\+?62|08)\d{8,11}\b')
_DOB_PATTERN = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')


def mask_pii(text: str) -> str:
    """
    Masking Protected Health Information (PII/PHI) dari string teks
    sebelum dicatat ke audit log atau error reporting.
    """
    if not text:
        return ""
    masked = _NIK_PATTERN.sub("[NIK-REDACTED]", text)
    masked = _PHONE_PATTERN.sub("[PHONE-REDACTED]", masked)
    masked = _DOB_PATTERN.sub("[DOB-REDACTED]", masked)
    return masked


def sanitize_dict_for_audit(data: Dict[str, Any], max_depth: int = 3) -> Dict[str, Any]:
    """
    Menghapus atau memotong data sensitif dari payload dictionary untuk logging audit.
    """
    if max_depth <= 0 or not isinstance(data, dict):
        return {}

    sanitized = {}
    sensitive_keys = {"patient_name", "nik", "phone", "raw_audio_bytes", "ktp", "address"}

    for k, v in data.items():
        if k in sensitive_keys:
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, str):
            sanitized[k] = mask_pii(v[:200] + "..." if len(v) > 200 else v)
        elif isinstance(v, dict):
            sanitized[k] = sanitize_dict_for_audit(v, max_depth - 1)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_dict_for_audit(item, max_depth - 1) if isinstance(item, dict)
                else (mask_pii(str(item)[:100]) if isinstance(item, str) else item)
                for item in v[:5]
            ]
        else:
            sanitized[k] = v
    return sanitized


class SecurityAuditLogger:
    """
    Logger Audit Trail Rumah Sakit berbasis JSONL dengan integrity hashing.
    Menjamin rekam jejak setiap tahapan dari audio input hingga EMR sync.
    """

    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit_trail.jsonl"

    def _generate_checksum(self, record_data: Dict[str, Any]) -> str:
        """Menghasilkan SHA256 checksum untuk deteksi tamper pada record audit."""
        serialized = json.dumps(record_data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def log_event(self,
                  session_id: str,
                  action: str,
                  actor: str = "system",
                  role: str = "system",
                  status: str = "SUCCESS",
                  details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Mencatat event audit trail baru.

        Args:
            session_id: ID sesi EMR
            action: Konstanta AuditAction
            actor: ID user/tenaga medis pembuat aksi (misal: "dr_andi", "system")
            role: Peran aktor (doctor, nurse, admin, system)
            status: SUCCESS, WARNING, FAILED
            details: Informasi tambahan yang relevan (akan otomatis disanitasi)
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sanitized_details = sanitize_dict_for_audit(details or {})

        record = {
            "timestamp": timestamp,
            "session_id": session_id,
            "action": action,
            "actor": actor,
            "role": role,
            "status": status,
            "details": sanitized_details
        }
        record["checksum"] = self._generate_checksum(record)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # Fallback jika terjadi IO error (tidak membocorkan data)
            print(f"[AUDIT_ERROR] Failed to write audit record: {type(e).__name__}")

        return record

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Mengambil seluruh riwayat audit trail untuk session tertentu."""
        if not self.log_file.exists():
            return []

        history = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        history.append(entry)
        except Exception as e:
            print(f"[AUDIT_ERROR] Failed to read audit records: {type(e).__name__}")

        return history


# Global singleton instance
audit_logger = SecurityAuditLogger()
