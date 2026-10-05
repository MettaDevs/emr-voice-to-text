# Voice-to-EMR Medical Intake Assistant

Layanan AI ekstraksi suara percakapan dokter-pasien (*Speech-to-Text* + *Information Extraction*) untuk pengisian otomatis formulir rekam medis (Electronic Medical Record / EMR).

## Fitur Utama
1. **Speech-to-Text (Faster-Whisper):** Mentranskripsikan audio percakapan klinis ke teks Bahasa Indonesia secara cepat dan hemat komputasi (CPU int8 / GPU CUDA).
2. **Layer Pembersih Noise (Noise Cleaner):** Membuang kata jeda / *filler words* seperti "eee", "eh", "anu", "hmm", "mmm".
3. **Layer Pemilah Klinis (Clinical Extractor):** Memisahkan percakapan secara cerdas ke dalam dua field formulir rekam medis:
   - `KELUHAN UTAMA *` (Chief Complaint)
   - `KELUHAN TAMBAHAN / ANAMNESA *` (Secondary Symptoms)
4. **Web UI & REST API:** Antarmuka interaktif dan endpoint HTTP untuk integrasi ke sistem web / EMR apa pun.

## Struktur Direktori
- `web_app.py` : Server REST API FastAPI & Web UI (Port 8050)
- `transcriber.py` : Engine Faster-Whisper
- `noise_cleaner.py` : Modul pembersih noise verbal
- `medical_extractor.py` : Layer ekstraksi keluhan medis
- `run_pipeline.py` : Script CLI untuk pengujian otomatis semua kasus
- `audio_samples/` : Sampel rekaman suara dokter-pasien
- `static/index.html` : Web UI interaktif
- `start.bat` : Shortcut untuk menjalankan server langsung

## Cara Menjalankan
Cukup klik dua kali `start.bat` atau jalankan dari PowerShell:
```powershell
cd c:\PKL\voice-emr-service
.\venv\Scripts\python.exe web_app.py
```
Lalu buka browser di: `http://localhost:8050`
