"""
audio_chunker.py — Audio Buffering, Sequencing & Chunking Manager (Section 3)
=============================================================================
Menyediakan mekanisme buffering yang aman:
  - Sequence ID unik & berurutan (seq_0, seq_1, ...)
  - Timestamp start & end setiap potongan audio
  - Deteksi chunk hilang / terlewat (gap detection)
  - Pencegahan pemrosesan chunk duplikat (idempotensi)
  - Penyusunan (assembly) potongan audio menjadi file lengkap tanpa artefak perbatasan
  - Pelacakan partial transcript per chunk
"""

import os
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class AudioChunk:
    def __init__(self,
                 session_id: str,
                 sequence_id: int,
                 start_time: float,
                 end_time: float,
                 raw_bytes: bytes,
                 file_ext: str = ".webm"):
        self.session_id = session_id
        self.sequence_id = sequence_id
        self.start_time = round(start_time, 2)
        self.end_time = round(end_time, 2)
        self.duration = round(end_time - start_time, 2) if end_time >= start_time else 0.0
        self.raw_bytes = raw_bytes
        self.byte_size = len(raw_bytes)
        self.file_ext = file_ext
        self.checksum = hashlib.sha256(raw_bytes).hexdigest()[:16]
        self.status = "received"  # received | processing | transcribed | failed
        self.partial_transcript = ""
        self.created_at = time.time()


class AudioSessionManager:
    """
    Manajer Sesi Audio & Penyangga Chunking Real-time untuk Lingkungan Rumah Sakit.
    Menjamin data ucapan tidak hilang antar-potongan dan mencatat diagnostik urutan.
    """

    def __init__(self, storage_dir: str = "uploads/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # Sesi aktif di memori: session_id -> dict
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(self,
                      session_id: str,
                      client_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Inisialisasi sesi perekaman audio baru."""
        session_folder = self.storage_dir / session_id
        session_folder.mkdir(parents=True, exist_ok=True)

        meta = client_metadata or {}
        session_record = {
            "session_id": session_id,
            "status": "recording",  # recording | completed | aborted
            "created_at": time.time(),
            "client_metadata": meta,
            "chunks": {},            # sequence_id -> AudioChunk
            "processed_checksums": set(),
            "highest_sequence_id": -1,
            "session_folder": str(session_folder),
            "final_audio_path": None,
            "final_transcript": None
        }

        self._sessions[session_id] = session_record
        return {
            "session_id": session_id,
            "status": "recording",
            "message": "Sesi audio berhasil diinisialisasi."
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def add_chunk(self,
                  session_id: str,
                  sequence_id: int,
                  audio_bytes: bytes,
                  start_time: float = 0.0,
                  end_time: float = 0.0,
                  file_ext: str = ".webm") -> Dict[str, Any]:
        """
        Menerima dan memvalidasi audio chunk dengan deteksi urutan dan idempotensi.
        """
        if session_id not in self._sessions:
            self.start_session(session_id)

        session = self._sessions[session_id]
        chunk = AudioChunk(session_id, sequence_id, start_time, end_time, audio_bytes, file_ext)

        # 1. Proteksi Duplikasi: Jika checksum chunk sudah diproses, tolak duplikasi secara aman
        if chunk.checksum in session["processed_checksums"]:
            existing_chunk = session["chunks"].get(sequence_id)
            return {
                "session_id": session_id,
                "sequence_id": sequence_id,
                "is_duplicate": True,
                "status": "cached",
                "checksum": chunk.checksum,
                "partial_transcript": existing_chunk.partial_transcript if existing_chunk else ""
            }

        # Simpan chunk ke disk
        chunk_filename = f"chunk_{sequence_id:04d}_{chunk.checksum}{file_ext}"
        chunk_path = Path(session["session_folder"]) / chunk_filename
        with open(chunk_path, "wb") as f:
            f.write(audio_bytes)

        chunk.file_path = str(chunk_path)
        session["chunks"][sequence_id] = chunk
        session["processed_checksums"].add(chunk.checksum)

        if sequence_id > session["highest_sequence_id"]:
            session["highest_sequence_id"] = sequence_id

        # 2. Deteksi Urutan yang Hilang (Missing Sequence Gap Detection)
        expected_seqs = set(range(session["highest_sequence_id"] + 1))
        current_seqs = set(session["chunks"].keys())
        missing_seqs = sorted(list(expected_seqs - current_seqs))

        return {
            "session_id": session_id,
            "sequence_id": sequence_id,
            "is_duplicate": False,
            "status": "stored",
            "byte_size": chunk.byte_size,
            "checksum": chunk.checksum,
            "total_chunks_received": len(session["chunks"]),
            "missing_sequences": missing_seqs,
            "has_sequence_gap": len(missing_seqs) > 0
        }

    def update_partial_transcript(self, session_id: str, sequence_id: int, text: str):
        """Menyimpan partial transcript yang dihasilkan untuk chunk tertentu."""
        session = self._sessions.get(session_id)
        if session and sequence_id in session["chunks"]:
            session["chunks"][sequence_id].partial_transcript = text
            session["chunks"][sequence_id].status = "transcribed"

    def get_accumulated_partial_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        """Mengumpulkan partial transcript berurutan dari seluruh chunk yang ada."""
        session = self._sessions.get(session_id)
        if not session:
            return []

        result = []
        for seq_id in sorted(session["chunks"].keys()):
            c = session["chunks"][seq_id]
            if c.partial_transcript:
                result.append({
                    "sequence_id": seq_id,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "text": c.partial_transcript,
                    "status": "partial"
                })
        return result

    def assemble_session_audio(self, session_id: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Menggabungkan seluruh audio chunk secara berurutan menjadi satu file WAV terpadu.
        Memeriksa gap antar-chunk dan mencegah kata awal/akhir terpotong.
        """
        import av
        import numpy as np

        session = self._sessions.get(session_id)
        if not session or not session["chunks"]:
            return None, {
                "valid": False,
                "error": "Sesi tidak ditemukan atau tidak memiliki potongan audio.",
                "total_chunks": 0
            }

        sorted_chunks: List[AudioChunk] = [
            session["chunks"][k] for k in sorted(session["chunks"].keys())
        ]

        expected_count = session["highest_sequence_id"] + 1
        received_count = len(sorted_chunks)
        missing_count = expected_count - received_count

        combined_np = None
        first_chunk = sorted_chunks[0]
        ext = os.path.splitext(first_chunk.file_path)[1].lower()

        # ── 1. Pendekatan Stream Byte Concatenation (Khusus WebM/Opus Browser) ──
        # Browser MediaRecorder menghasilkan 1 file WebM kontinu yang dipotong menjadi chunks.
        # Hanya chunk 0 yang memiliki EBML Header, sedangkan chunk 1..N adalah Clusters.
        # Menggabungkan raw bytes memungkinkan PyAV membaca seluruh percakapan tanpa terpotong di 2 detik pertama!
        if ext in [".webm", ".ogg", ".opus", ".mkv"]:
            try:
                raw_bytes = bytearray()
                for c in sorted_chunks:
                    if os.path.exists(c.file_path):
                        with open(c.file_path, "rb") as cf:
                            raw_bytes.extend(cf.read())

                if raw_bytes:
                    import io
                    container = av.open(io.BytesIO(raw_bytes))
                    if container.streams.audio:
                        stream = container.streams.audio[0]
                        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
                        frames = []
                        for frame in container.decode(stream):
                            for r_frame in resampler.resample(frame):
                                frames.append(r_frame.to_ndarray())
                        container.close()
                        if frames:
                            combined_np = np.concatenate(frames, axis=1)[0].astype(np.float32)
            except Exception as e:
                # Jika stream concat gagal, lanjut ke per-chunk decoding fallback
                combined_np = None

        # ── 2. Pendekatan Per-Chunk Decoding (Untuk WAV/PCM Mandiri) ──
        if combined_np is None:
            all_audio_frames = []
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

            for c in sorted_chunks:
                if not os.path.exists(c.file_path) or os.path.getsize(c.file_path) == 0:
                    continue
                try:
                    container = av.open(c.file_path)
                    if container.streams.audio:
                        for frame in container.decode(audio=0):
                            for r_frame in resampler.resample(frame):
                                all_audio_frames.append(r_frame.to_ndarray())
                    container.close()
                except Exception as e:
                    # Catat peringatan jika decode chunk gagal
                    c.status = "failed"
                    c.error = str(e)

            if not all_audio_frames:
                return None, {
                    "valid": False,
                    "error": "Seluruh chunk audio gagal didecode atau kosong.",
                    "total_chunks": received_count,
                    "missing_chunks": missing_count
                }

            combined_np = np.concatenate(all_audio_frames, axis=1)[0].astype(np.float32)

        total_duration = round(len(combined_np) / 16000.0, 3)

        # Simpan file WAV gabungan
        from audio_processor import save_pcm_wav
        output_wav = os.path.join(session["session_folder"], f"{session_id}_assembled.wav")
        save_pcm_wav(combined_np, output_wav, sample_rate=16000)

        session["final_audio_path"] = output_wav
        session["status"] = "completed"

        diagnostics = {
            "valid": True,
            "session_id": session_id,
            "total_chunks_received": received_count,
            "missing_chunks_count": missing_count,
            "total_duration_seconds": total_duration,
            "assembled_wav_path": output_wav,
            "sequence_integrity": missing_count == 0
        }
        return output_wav, diagnostics


# Global singleton instance
session_manager = AudioSessionManager()
