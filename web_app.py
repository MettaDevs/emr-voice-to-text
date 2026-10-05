import os
import shutil
import time
import json
import datetime
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from transcriber import SpeechTranscriber
from medical_extractor import MedicalComplaintExtractor
from security_audit import audit_logger, AuditAction, mask_pii
from emr_integration import (
    get_emr_adapter,
    validate_emr_payload_schema,
    UnapprovedSubmissionError,
    EMRTransmissionError
)
from evaluation import run_benchmark_suite
from audio_processor import inspect_audio_file, AudioQualityStatus
from audio_chunker import session_manager

app = FastAPI(title="MedVoice AI — Voice-to-EMR Hospital Grade Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "audio_samples")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
HISTORY_DIR = os.path.join(BASE_DIR, "session_history")

for d in [AUDIO_DIR, STATIC_DIR, UPLOADS_DIR, HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)

print("[Inisialisasi] Memuat engine Faster-Whisper Large-v3-Turbo & Medical Extractor...")
transcriber = SpeechTranscriber(model_size="large-v3-turbo")
extractor = MedicalComplaintExtractor()


def _save_session(session_data: dict) -> str:
    """Simpan sesi ke file JSON untuk riwayat."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{ts}.json"
    path = os.path.join(HISTORY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    return filename


def _build_response(stt_res: dict, ext_res: dict, audio_file: str,
                    audio_url: str | None, transcription_time: float,
                    total_time: float, auto_save: bool = True) -> dict:
    """Bangun response standar dari hasil STT + extraction dengan hospital POC schema."""
    session_id = ext_res.get("session_metadata", {}).get("session_id") or f"VA-{int(time.time())}"

    session = {
        # ── Status & Metadata ──
        "status": "success",
        "session_id": session_id,
        "session_metadata": ext_res.get("session_metadata", {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_file": audio_file,
            "audio_duration": stt_res.get("duration", 0.0),
            "language": stt_res.get("detected_language", "id")
        }),
        "audio_file": audio_file,
        "audio_url": audio_url,
        "audio_duration": stt_res.get("duration", 0.0),
        "detected_language": stt_res.get("detected_language", "id"),
        "language_probability": stt_res.get("language_probability", 0.0),
        "transcription_time": transcription_time,
        "total_time": total_time,

        # ── Audio Quality & Diagnostics (Sections 8, 9, 10) ──
        "audio_status": stt_res.get("audio_status", "OPTIMAL"),
        "transcription_status": stt_res.get("transcription_status", "success"),
        "audio_diagnostics": stt_res.get("audio_diagnostics", {}),
        "warning": stt_res.get("warning"),
        "missing_segments": stt_res.get("missing_segments", []),
        "vad_fallback_used": stt_res.get("vad_fallback_used", False),

        # ── Transkrip Asli & Terkoreksi (P0) ──
        "original_text": ext_res.get("original_text", stt_res.get("text", "")),
        "corrected_text": ext_res.get("corrected_text", ext_res.get("cleaned_transcript", "")),
        "corrections": ext_res.get("corrections", []),

        # ── Speaker Segments (P0) ──
        "segments": ext_res.get("segments", ext_res.get("labeled_segments", [])),

        # ── Clinical Extraction (P1) ──
        "chief_complaint": ext_res.get("chief_complaint", {}),
        "symptoms": ext_res.get("symptoms", []),
        "duration": ext_res.get("duration"),
        "vitals": ext_res.get("vitals", {}),
        "allergies": ext_res.get("allergies", {}),
        "medical_history": ext_res.get("medical_history", []),
        "medications_mentioned": ext_res.get("medications_mentioned", []),
        "procedures_mentioned": ext_res.get("procedures_mentioned", []),
        "uncategorized": ext_res.get("uncategorized", []),

        # ── Confidence & Clinical Validation (P0/P1) ──
        "confidence": ext_res.get("confidence", {}),
        "review": ext_res.get("review", {}),
        "clinical_validation": ext_res.get("clinical_validation", {}),
        "review_status": ext_res.get("review_status", "draft"),

        # ── Legacy Keys (100% Backward Compatibility) ──
        "raw_transcript": stt_res.get("text", ""),
        "layer1_phonetic": ext_res.get("layer1_phonetic", ""),
        "cleaned_transcript": ext_res.get("cleaned_transcript", ""),
        "labeled_segments": ext_res.get("labeled_segments", []),
        "clinical_entities": ext_res.get("clinical_entities", {}),
        "pipeline_layers": ext_res.get("pipeline_layers", {}),
        "fields": {
            "keluhan_utama": ext_res.get("keluhan_utama", ""),
            "keluhan_tambahan": ext_res.get("keluhan_tambahan", ""),
            "lama_sakit": ext_res.get("lama_sakit", ""),
        },
        "ml_classification": ext_res.get("ml_predicted_class", "KLINIS"),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Audit logging (P2)
    audit_logger.log_event(
        session_id=session_id,
        action=AuditAction.TRANSCRIPTION_COMPLETED,
        actor="system",
        role="system",
        status="SUCCESS",
        details={"audio_duration": stt_res.get("duration", 0.0), "words": len(stt_res.get("text", "").split())}
    )
    audit_logger.log_event(
        session_id=session_id,
        action=AuditAction.CLINICAL_EXTRACTED,
        actor="system",
        role="system",
        status="SUCCESS",
        details={"chief": ext_res.get("keluhan_utama", ""), "review_required": ext_res.get("review", {}).get("required", False)}
    )

    if auto_save and stt_res.get("text", "").strip():
        session["history_file"] = _save_session(session)
    return session


# ─── Samples ────────────────────────────────────────────────────────────────

@app.get("/api/samples")
def get_samples():
    return [
        {
            "id": "sample_1_kepala_pipi.mp3",
            "title": "Kasus 1: Sakit Kepala + Pipi",
            "description": "Dokter: 'Keluhannya apa?' | Pasien: 'Eee kepala saya sakit, terus di atas pipi saya sakit'",
            "file_url": "/audio/sample_1_kepala_pipi.mp3"
        },
        {
            "id": "sample_2_perut_meriang.mp3",
            "title": "Kasus 2: Perut Melilit + Meriang & Mual",
            "description": "Dokter: 'Ada keluhan apa bu?' | Pasien: 'Anu dok, perut saya melilit dari kemarin malam, terus meriang dan mual'",
            "file_url": "/audio/sample_2_perut_meriang.mp3"
        },
        {
            "id": "sample_3_batuk_sesak.mp3",
            "title": "Kasus 3: Batuk Kering + Tenggorokan Gatal",
            "description": "Dokter: 'Keluhan utamanya apa?' | Pasien: 'Eee batuk kering sudah tiga hari, tenggorokan gatal dan agak sesak'",
            "file_url": "/audio/sample_3_batuk_sesak.mp3"
        },
        {
            "id": "test_rontgen.mp3",
            "title": "Kasus 4: Rontgen Dada",
            "description": "'Dok, kemarin saya sudah rontgen dada di rumah sakit'",
            "file_url": "/audio/test_rontgen.mp3"
        },
        {
            "id": "test_word.mp3",
            "title": "Pengujian Mic: 'Test'",
            "description": "Cek mic — tidak ada keluhan medis",
            "file_url": "/audio/test_word.mp3"
        }
    ]


# ─── Audio Serving ──────────────────────────────────────────────────────────

@app.get("/audio/{filename}")
def serve_audio(filename: str):
    for directory in [AUDIO_DIR, UPLOADS_DIR]:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return FileResponse(path)
    return JSONResponse({"error": "Audio file not found"}, status_code=404)


# ─── Process Sample ─────────────────────────────────────────────────────────

@app.post("/api/process-sample")
def process_sample(sample_id: str = Form(...)):
    file_path = os.path.join(AUDIO_DIR, sample_id)
    if not os.path.exists(file_path):
        return JSONResponse({"error": f"File {sample_id} tidak ditemukan"}, status_code=404)

    t0 = time.time()
    stt_res = transcriber.transcribe(file_path)
    t_stt = round(time.time() - t0, 2)

    ext_res = extractor.extract(stt_res["text"], stt_res.get("segments", []))
    t_total = round(time.time() - t0, 2)

    return _build_response(stt_res, ext_res, sample_id, f"/audio/{sample_id}",
                           t_stt, t_total)


# ─── Process Upload ─────────────────────────────────────────────────────────

@app.post("/api/process-upload")
async def process_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".webm"
    filename = f"rec_{int(time.time())}{ext}"
    save_path = os.path.join(UPLOADS_DIR, filename)

    with open(save_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    t0 = time.time()
    stt_res = transcriber.transcribe(save_path)
    t_stt = round(time.time() - t0, 2)

    ext_res = extractor.extract(stt_res["text"], stt_res.get("segments", []))
    t_total = round(time.time() - t0, 2)

    return _build_response(stt_res, ext_res, filename, f"/audio/{filename}",
                           t_stt, t_total)


# ─── Audio Chunking & Streaming Endpoints (Sections 3, 7, 8, 9, 11) ─────────

@app.post("/api/audio/session/start")
async def start_audio_session(request: Request):
    """Inisialisasi sesi perekaman audio chunking baru dengan metadata klien."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = body.get("session_id") or f"SESS-{int(time.time() * 1000)}"
    return session_manager.start_session(session_id, client_metadata=body)


@app.post("/api/audio/chunk")
async def upload_audio_chunk(
    session_id: str = Form(...),
    sequence_id: int = Form(...),
    start_time: float = Form(0.0),
    end_time: float = Form(0.0),
    file: UploadFile = File(...)
):
    """Menerima audio chunk dengan nomor urut (sequence ID), deteksi gap & duplikasi."""
    ext = os.path.splitext(file.filename)[1] or ".webm"
    audio_bytes = await file.read()

    res = session_manager.add_chunk(
        session_id=session_id,
        sequence_id=sequence_id,
        audio_bytes=audio_bytes,
        start_time=start_time,
        end_time=end_time,
        file_ext=ext
    )
    return res


@app.post("/api/audio/session/complete")
def complete_audio_session(session_id: str = Form(...)):
    """Menggabungkan seluruh audio chunk, menjalankan Whisper GPU & ekstraksi klinis."""
    t0 = time.time()
    assembled_wav, diag = session_manager.assemble_session_audio(session_id)

    if not assembled_wav or not diag["valid"]:
        return JSONResponse({
            "status": "error",
            "error": diag.get("error", "Gagal merakit potongan audio."),
            "audio_status": "corrupted",
            "transcription_status": "failed",
            "diagnostics": diag
        }, status_code=400)

    stt_res = transcriber.transcribe(assembled_wav)
    t_stt = round(time.time() - t0, 2)

    ext_res = extractor.extract(stt_res["text"], stt_res.get("segments", []))
    t_total = round(time.time() - t0, 2)

    stt_res["missing_segments"] = diag.get("missing_chunks_count", 0)
    if "audio_diagnostics" in stt_res:
        stt_res["audio_diagnostics"]["chunking"] = diag

    return _build_response(stt_res, ext_res, os.path.basename(assembled_wav),
                           None, t_stt, t_total)


@app.get("/api/audio/session/{session_id}/status")
def get_session_status(session_id: str):
    """Mendapatkan status sesi rekaman, urutan chunk, dan partial transcript."""
    session = session_manager.get_session(session_id)
    if not session:
        return JSONResponse({"error": "Sesi tidak ditemukan"}, status_code=404)

    partials = session_manager.get_accumulated_partial_transcript(session_id)
    return {
        "session_id": session_id,
        "status": session["status"],
        "total_chunks_received": len(session["chunks"]),
        "highest_sequence_id": session["highest_sequence_id"],
        "partial_transcripts": partials
    }


@app.post("/api/audio/diagnostics")
async def audio_diagnostics(file: UploadFile = File(...)):
    """Inspeksi akustik cepat (RMS, peak, clipping, durasi) terhadap file audio."""
    ext = os.path.splitext(file.filename)[1] or ".webm"
    temp_path = os.path.join(UPLOADS_DIR, f"diag_{int(time.time()*1000)}{ext}")
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    diag = inspect_audio_file(temp_path)
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return diag


# ─── Session History ────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history(limit: int = 20):
    """Ambil daftar sesi yang tersimpan (terbaru di atas)."""
    files = sorted(
        [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")],
        reverse=True
    )[:limit]
    sessions = []
    for fn in files:
        path = os.path.join(HISTORY_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "file": fn,
                "session_id": data.get("session_id", "-"),
                "timestamp": data.get("timestamp", "-"),
                "audio_file": data.get("audio_file", "-"),
                "keluhan_utama": data.get("fields", {}).get("keluhan_utama", ""),
                "confidence_level": data.get("confidence", {}).get("level", "-"),
            })
        except Exception:
            pass
    return sessions


# ─── Pydantic Models for Hospital Review & Export ───────────────────────────

class ReviewRequest(BaseModel):
    session_id: str
    action: str = Field(..., description="'approve', 'reject', or 'update'")
    actor: str = "dr_evaluator"
    role: str = "doctor"
    notes: Optional[str] = ""
    updated_fields: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    session_id: str
    adapter_type: str = "mock"  # "mock" or "fhir"
    actor: str = "dr_evaluator"


def _find_session_file(session_id: str) -> tuple[Optional[str], Optional[dict]]:
    """Cari file sesi berdasarkan session_id atau nama file."""
    for fn in os.listdir(HISTORY_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("session_id") == session_id or fn == session_id:
                return path, data
        except Exception:
            continue
    return None, None


# ─── Human-in-the-Loop Review Endpoints (Section 11) ────────────────────────

@app.post("/api/emr/review")
def review_emr_draft(req: ReviewRequest):
    """
    Endpoint Human-in-the-Loop untuk tenaga medis:
    - Menyetujui (approve) draft EMR
    - Menolak (reject) draft EMR
    - Memperbarui (update) isi field EMR sebelum persetujuan
    """
    path, session_data = _find_session_file(req.session_id)
    if not session_data:
        return JSONResponse({"error": f"Sesi {req.session_id} tidak ditemukan"}, status_code=404)

    prev_status = session_data.get("review_status", "draft")

    if req.action == "approve":
        session_data["review_status"] = "approved"
        session_data["approved_by"] = req.actor
        session_data["approved_at"] = datetime.datetime.now().isoformat()
        if req.notes:
            session_data["review_notes"] = req.notes

        audit_logger.log_event(
            session_id=req.session_id,
            action=AuditAction.STAFF_APPROVED,
            actor=req.actor,
            role=req.role,
            status="SUCCESS",
            details={"notes": req.notes, "prev_status": prev_status}
        )

    elif req.action == "reject":
        session_data["review_status"] = "rejected"
        session_data["rejected_by"] = req.actor
        session_data["rejected_at"] = datetime.datetime.now().isoformat()
        session_data["reject_reason"] = req.notes or "Ditolak oleh tenaga medis"

        audit_logger.log_event(
            session_id=req.session_id,
            action=AuditAction.STAFF_REJECTED,
            actor=req.actor,
            role=req.role,
            status="SUCCESS",
            details={"reason": req.notes}
        )

    elif req.action == "update":
        if req.updated_fields:
            if "staff_corrections" not in session_data:
                session_data["staff_corrections"] = []

            # 1. Simpan data awal tanpa menimpa riwayat (Section 7)
            if "initial_draft" not in session_data:
                session_data["initial_draft"] = {
                    "fields": dict(session_data.get("fields", {})),
                    "created_at": session_data.get("timestamp")
                }

            field_diffs = {}
            now_iso = datetime.datetime.now().isoformat()
            for k, v in req.updated_fields.items():
                old_v = session_data.get("fields", {}).get(k, "")
                if str(old_v).strip() != str(v).strip():
                    field_diffs[k] = {"before": old_v, "after": v}
                    correction_entry = {
                        "field": k,
                        "original_value": old_v,
                        "corrected_value": v,
                        "corrected_by": req.actor,
                        "correction_reason": req.notes or "Transcription correction",
                        "timestamp": now_iso,
                        "used_for_training": False  # Dilarang otomatis pakai sbg data training tanpa kurasi
                    }
                    session_data["staff_corrections"].append(correction_entry)

                if "fields" in session_data:
                    session_data["fields"][k] = v

            # Sinkronisasi ke chief_complaint jika keluhan utama diedit
            if "keluhan_utama" in req.updated_fields:
                if "chief_complaint" in session_data and isinstance(session_data["chief_complaint"], dict):
                    session_data["chief_complaint"]["value"] = req.updated_fields["keluhan_utama"]

            session_data["last_edited_by"] = req.actor
            session_data["last_edited_at"] = now_iso

            audit_logger.log_event(
                session_id=req.session_id,
                action=AuditAction.STAFF_REVIEW_RECORDED,
                actor=req.actor,
                role=req.role,
                status="SUCCESS",
                details={"diffs": field_diffs, "notes": req.notes}
            )

    elif req.action == "flag":
        session_data["review_status"] = "flagged_inappropriate"
        session_data["flagged_by"] = req.actor
        session_data["flagged_at"] = datetime.datetime.now().isoformat()
        session_data["flag_reason"] = req.notes or "Ditandai tidak sesuai oleh tenaga medis"

        audit_logger.log_event(
            session_id=req.session_id,
            action=AuditAction.STAFF_REVIEW_RECORDED,
            actor=req.actor,
            role=req.role,
            status="FLAGGED",
            details={"reason": req.notes}
        )

    else:
        return JSONResponse({"error": f"Aksi '{req.action}' tidak valid. Gunakan 'approve', 'reject', 'update', atau 'flag'."}, status_code=400)

    # Simpan pembaruan ke file sesi
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "session_id": req.session_id,
        "review_status": session_data.get("review_status"),
        "message": f"Sesi berhasil diperbarui dengan aksi '{req.action}'."
    }


@app.get("/api/emr/session/{session_id}")
def get_emr_session(session_id: str):
    """Mengambil data draft EMR lengkap berserta status review & validasi klinis."""
    path, session_data = _find_session_file(session_id)
    if not session_data:
        return JSONResponse({"error": f"Sesi {session_id} tidak ditemukan"}, status_code=404)
    return session_data


@app.get("/api/emr/session/{session_id}/corrections")
def get_session_corrections(session_id: str):
    """Mengambil riwayat feedback dan koreksi tenaga medis (Section 7)."""
    path, session_data = _find_session_file(session_id)
    if not session_data:
        return JSONResponse({"error": f"Sesi {session_id} tidak ditemukan"}, status_code=404)
    return {
        "session_id": session_id,
        "initial_draft": session_data.get("initial_draft"),
        "staff_corrections": session_data.get("staff_corrections", []),
        "total_corrections": len(session_data.get("staff_corrections", []))
    }


# ─── EMR Export Endpoints (Section 12) ──────────────────────────────────────

@app.post("/api/emr/export")
def export_to_emr(req: ExportRequest):
    """
    Mengirimkan data rekam medis ke sistem EMR rumah sakit.
    Diproteksi ketat: Wajib berstatus 'approved' sebelum pengiriman.
    """
    path, session_data = _find_session_file(req.session_id)
    if not session_data:
        return JSONResponse({"error": f"Sesi {req.session_id} tidak ditemukan"}, status_code=404)

    try:
        adapter = get_emr_adapter(req.adapter_type)
        result = adapter.send_emr(session_data, actor=req.actor)
        return {
            "status": "success",
            "session_id": req.session_id,
            "export_result": result
        }
    except UnapprovedSubmissionError as e:
        return JSONResponse({
            "status": "blocked",
            "error": "UNAPPROVED_SUBMISSION",
            "message": str(e)
        }, status_code=403)
    except EMRTransmissionError as e:
        return JSONResponse({
            "status": "failed",
            "error": "TRANSMISSION_ERROR",
            "message": str(e)
        }, status_code=500)


@app.get("/api/emr/audit/{session_id}")
def get_audit_trail(session_id: str):
    """Mengambil riwayat audit trail lengkap untuk suatu sesi (Section 13)."""
    history = audit_logger.get_session_history(session_id)
    return {
        "session_id": session_id,
        "total_records": len(history),
        "audit_trail": history
    }


# ─── System Evaluation Benchmark Endpoint (Section 15) ─────────────────────

@app.get("/api/evaluation/benchmark")
def run_evaluation():
    """Menjalankan evaluasi metrik akurasi (WER/CER, F1, Negasi, TTV, Alergi)."""
    metrics = run_benchmark_suite(extractor)
    return {
        "status": "success",
        "benchmark_report": metrics
    }


# ─── Health Check ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    import ctranslate2
    cuda = ctranslate2.get_cuda_device_count()
    return {
        "status": "ok",
        "whisper": "ready",
        "device": "cuda" if cuda > 0 else "cpu",
        "cuda_devices": cuda,
        "model": "large-v3-turbo",
    }


# ─── Frontend ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MedVoice AI — Voice-to-EMR</h1>"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)
