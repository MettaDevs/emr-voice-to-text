# Voice-to-EMR Service (MedVoice AI)

Proyek AI Voice Recognition & Medical Entity Extraction mandiri berbasis Faster-Whisper Large-v3-Turbo untuk pengisian otomatis formulir Rekam Medis (EMR).

## Ringkasan Konteks & Status
- Repo ini terpisah dari CoreERP sesuai arahan atasan (Divavava).
- Model berjalan di GPU NVIDIA GeForce RTX 3050 (CUDA FP16).
- Bahasa utama: Bahasa Indonesia medis dengan dukungan istilah klinis bilingual (English medical terms).
- Tiga field utama formulir rekam medis:
  1. `KELUHAN UTAMA *`
  2. `KELUHAN TAMBAHAN / ANAMNESA *`
  3. `LAMA SAKIT / DURASI`

Rincian lengkap riwayat pengembangan, parameter teknis, dan catatan penting tercatat di `PROJECT_HISTORY.md`.
