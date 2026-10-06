# Voice-to-EMR Medical Intake Assistant (MedVoice AI)

Layanan AI Voice Recognition & Medical Entity Extraction berbasis Faster-Whisper Large-v3-Turbo pada GPU NVIDIA (CUDA FP16) untuk pengisian otomatis formulir Rekam Medis (Electronic Medical Record / EMR).

---

## Fitur Utama & Keunggulan
1. **Audio Preprocessing & Acoustic Filtering:** High-pass filter 80Hz Butterworth dan *Speech-Aware Adaptive Loudness Leveling* (-19 dBFS) pada `audio_processor.py`.
2. **Faster-Whisper STT (CUDA FP16):** Prompt prior klinis deskriptif anti-prompt bleeding, VAD dual-pass fallback, decoding deterministik pada `transcriber.py`.
3. **Medical Entity Extractor & Diarization:** 3 field utama EMR (*Keluhan Utama, Keluhan Tambahan, Lama Sakit*), deteksi tanda vital, alergi obat, serta pemisahan giliran bicara Dokter vs Pasien secara cerdas pada `medical_extractor.py`.
4. **Web UI & Streaming Durasi Panjang:** Rekaman percakapan durasi panjang via streaming chunk audio WebM/Opus, Live VU Meter dinamis, approval queue & audit trail pada `static/index.html` dan `web_app.py`.
5. **Evaluasi & Keandalan:** Benchmark WER 3.64%, CER 2.90%, coverage 100%, RTF 0.111x (latensi ~1.08s). Seluruh rangkaian 50+ pengujian unit regresi 100% PASS.

---

## Struktur Direktori Modul
- `web_app.py` : Server REST API FastAPI & Web UI (Port 8050)
- `transcriber.py` : Engine Faster-Whisper Large-v3-Turbo + CUDA FP16
- `audio_processor.py` : Filter akustik, audio leveling, dan VAD preprocessing
- `audio_chunker.py` : Perekam dan perakit potongan streaming audio WebM/Opus
- `medical_extractor.py` : Ekstraksi entitas klinis & heuristik giliran bicara (Diarization)
- `static/index.html` : Web UI interaktif formulir EMR dengan Live VU meter
- `test_modules.py` : Uji 14 modul klinis terpisah
- `test_clinical_fidelity.py` : Uji 11 aturan keselamatan makna klinis & pemisahan penutur
- `test_hospital_poc_p2.py` : Uji pipeline POC rumah sakit
- `test_audio_stt.py` : Uji integrasi audio dan Faster-Whisper STT
- `start.bat` : Shortcut launcher Windows

---

## Cara Menjalankan (How to Run)

### 1. Prasyarat Sistem
* **OS:** Windows / Linux
* **Python:** 3.10 atau 3.11
* **Hardware GPU (Rekomendasi):** NVIDIA GPU dengan CUDA Compute Capability >= 6.0 (misal: RTX 3050, RTX 3060, T4) dengan Driver CUDA 12.x. *(Tersedia fallback otomatis ke CPU int8 jika GPU tidak terdeteksi).*

### 2. Instalasi Dependensi
```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi virtual environment:
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependensi pustaka
pip install -r requirements.txt
```

*(Opsional untuk GPU NVIDIA):*
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

### 3. Menjalankan Server Web & API
```bash
python web_app.py
```
Atau klik dua kali `start.bat` (Windows).

Buka di peramban (browser):
* **Web UI (Formulir Rekam Medis):** [http://localhost:8050](http://localhost:8050) atau [http://127.0.0.1:8050](http://127.0.0.1:8050)
* **Health Check API:** `http://localhost:8050/api/health`

### 4. Menjalankan Pengujian (Testing)
```bash
# Uji 14 modul klinis:
python test_modules.py

# Uji Keselamatan Makna Klinis & Diarization:
python test_clinical_fidelity.py

# Uji Hospital POC:
python test_hospital_poc_p2.py

# Uji Audio STT:
python test_audio_stt.py
```
