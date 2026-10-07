# RIWAYAT & KONTEKS PROYEK VOICE-TO-EMR (MEDVOICE AI)

## 1. Latar Belakang & Instruksi Atasan (Divavava)
- **Instruksi:** Mengembangkan sistem *voice recognition* berbasis [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) yang terpisah dari CoreERP.
- **Tujuan Utama:** Membuat 1 layer di bawah model untuk membersihkan noise ("eee", "anu", "hmm") dan memilah percakapan konsultasi dokter-pasien secara otomatis ke dalam field formulir rekam medis:
  1. `KELUHAN UTAMA *` (Chief Complaint)
  2. `KELUHAN TAMBAHAN / ANAMNESA *` (Secondary Symptoms / Anamnesis)
  3. `LAMA SAKIT / DURASI` (Onset / Duration)

## 2. Arsitektur & Teknologi yang Sudah Aktif
- **Runtime:** Native Python 3.11 virtual environment (`venv`) di `C:\PKL\voice-emr-service`.
- **Akselerasi Hardware:** GPU NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) menggunakan CUDA 12.
- **Model STT:** `faster-whisper` **Large-v3-Turbo** (float16 pada CUDA GPU) dengan latensi ~1 detik.
- **Transkripsi Medis:**
  - Bahasa dikunci ke Bahasa Indonesia (`language="id"`) dengan kamus prompt prior klinis.
  - Proteksi anti-looping deret angka: `repetition_penalty=1.25`, `no_repeat_ngram_size=3`.
- **Layer Pemilah Klinis (`medical_extractor.py`) — v2:**
  - Parsing semantik dialog multi-turn (mendeteksi keluhan primer, gejala tambahan, durasi sakit).
  - **Confidence scoring:** TINGGI / SEDANG / RENDAH (0–100) berdasarkan kualitas ekstraksi.
  - **Warning system:** memberikan peringatan jika transcript terlalu pendek atau keluhan tidak terdeteksi.
  - **Speaker labeling:** heuristik Dokter/Pasien berdasarkan pola pertanyaan.
  - **Penanganan non-keluhan:** field kosong saat rekaman hanya berisi suara tes mic ("tes 1,2,3").
  - Normalisasi fonetik otomatis (`pilot` → `pilek`, `nganes` → `panas/anget`, `malum/malem` → `malam`).
- **Antarmuka & Web API:**
  - FastAPI server (`web_app.py`) di port 8050.
  - Endpoint baru: `/api/history` (riwayat sesi), `/api/health`.
  - UI Web modern putih klinis (`static/index.html`) — didesain ulang v2.

## 3. Berkas-Berkas Inti
- `web_app.py`           : Server REST API & Web UI (FastAPI)
- `transcriber.py`       : Engine Faster-Whisper Large-v3-Turbo + CUDA (Prompt dialog klinis & istilah sehari-hari)
- `noise_cleaner.py`     : Pembersih kata jeda & normalisasi fonetik
- `medical_extractor.py` : Pipeline berlapis klinis (L1 Fonetik, L2 Bahasa Indonesia & Sehari-hari, L3 Konteks Kalimat & Speaker Diarization, L4 Konteks Medis & Confidence)
- `static/index.html`    : Tampilan antarmuka EMR modern (Manual voice recording, transcript berlabel pembicara, diagnostik 5 tahap, riwayat sesi)
- `session_history/`     : Folder penyimpanan JSON sesi yang sudah diproses
- `start.bat`            : Shortcut menjalankan server

## 4. Fitur yang Sudah Berjalan
- [x] Input audio manual: Rekam mikrofon langsung & Upload file (sample cards dihapus sesuai instruksi)
- [x] Speech-to-text akurat (GPU RTX 3050 CUDA, Faster-Whisper Large-v3-Turbo)
- [x] Prompt Prior Klinis Rumah Sakit Indonesia (<= 210 token): mengarahkan Whisper memahami kosakata RS (IGD, BPJS, rontgen dada, tensi, paracetamol, dll) tanpa truncating
- [x] Layer 1: Koreksi fonetik audio otomatis (`tanas` → `panas`, `keluarnya` → `keluhannya`, `kau belah tahu` → `kalau boleh tahu`, `pikir` → `pilek`, `tengsi` → `tensi`, `ronsen` → `rontgen`, `parasetamol` → `paracetamol`)
- [x] Layer 2: Normalisasi Bahasa Indonesia & tutur sehari-hari pasien tanpa polusi kamus (transkrip tetap natural dan baku, bebas duplikasi kata seperti `sudah sudah` atau `diare diare`)
- [x] Layer 3 (Maximal Conversation Detection): 
  - Dekomposisi SETIAP segmen akustik Whisper menjadi sub-kalimat tutur individu dengan interpolasi timestamp proporsional
  - State Machine Diarization Dokter vs Pasien tingkat tinggi (mencakup sapaan, anamnesis, pemeriksaan fisik, pengukuran TTV, peresepan, dan closing)
- [x] Layer 4 (Maximal Clinical Intelligence):
  - **Negation Detection Engine:** Mendeteksi dan menapis gejala yang disangkal/dinegasikan pasien ("tidak ada sesak napas", "nggak muntah", "alergi disangkal") agar tidak keliru masuk EMR
  - **Patient-Isolated Extraction:** Mengekstrak keluhan pasien dari tuturan pasien saja, bukan pertanyaan dokter
  - **Redundancy Suppression:** Mencegah gejala umum menduplikasi gejala spesifik (contoh: jika sudah ada `Batuk berdahak`, tidak menambahkan `Batuk`)
  - **Ekstraksi Tanda-Tanda Vital (TTV):** Otomatis mendeteksi Tensi (mmHg), Suhu Tubuh (°C), SpO2 (%), dan Nadi (x/menit)
  - **Ekstraksi Status Alergi:** Mendeteksi riwayat alergi obat/makanan maupun status alergi disangkal
- [x] Dukungan durasi angka kata bahasa Indonesia ("dari dua hari yang lalu", "sudah tiga hari", "sejak semalam", "kemarin lusa")
- [x] Penanganan audio non-keluhan/tes mic: keluhan utama dikosongkan dengan warning jelas
- [x] Confidence scoring (TINGGI/SEDANG/RENDAH) & Warning sistem klinis
- [x] Formulir EMR yang dapat diedit langsung oleh tenaga medis sebelum disimpan
- [x] Penyimpanan riwayat sesi otomatis ke JSON & panel peninjauan riwayat
- [x] **Peredam Noise & Hembusan Nafas (Breath & Noise Suppression):**
  - Parameter Silero VAD diperketat (`threshold=0.48`, `min_speech_duration_ms=250`, `speech_pad_ms=200`) sehingga desah nafas, hembusan mik, batuk kecil, dan gumaman terpotong otomatis di tingkat audio sebelum inferensi Whisper.
  - Eliminasi total kata jeda/filler (`hmm`, `hmmm`, `mmm`, `amin`, `aamiin`, `[sigh]`, `(hembusan napas)`) tanpa disubstitusi atau diubah menjadi kata lain.
  - Segmen akustik yang hanya berisi suara nafas/noise otomatis digugurkan dari daftar dialog pembicara.
- [x] **Dukungan Perekaman Durasi Panjang (Long-Duration Recording):**
  - Streaming chunking browser via `MediaRecorder.start(1000)` (timeslice 1 detik) mencegah lonjakan memori atau file korup pada konsultasi berdurasi 5–15+ menit.
  - Timer perekaman visual real-time (`🔴 00:00`) menampilkan durasi live konsultasi.
  - Audio hardware DSP browser aktif (`channelCount: 1`, `sampleRate: 16000`, `echoCancellation: true`, `noiseSuppression: true`, `autoGainControl: true`).
- [x] **Penyederhanaan UI Medis Fokus:**
  - Menghapus panel "Riwayat Sesi" dan "Rincian Pipeline AI (5 Tahap)" dari antarmuka agar tampilan bersih dan fokus pada input percakapan serta formulir EMR.
  - Layout satu kolom terpusat yang lapang dan mudah dibaca oleh dokter dan tenaga medis.
  - Menambahkan tombol **📋 Salin Formulir** dengan umpan balik visual untuk kemudahan copy-paste ke sistem SIMRS.

## 5. Cara Menjalankan
```powershell
cd C:\PKL\voice-emr-service
.\venv\Scripts\python.exe web_app.py
```
Akses melalui browser: **http://127.0.0.1:8050**

---

## 6. Hospital POC Enhancement — Tingkat Rumah Sakit (v3.0)

Berdasarkan spesifikasi resmi POC Rumah Sakit, sistem MedVoice AI telah ditingkatkan secara menyeluruh pada 3 tingkat prioritas (P0, P1, P2) dengan fokus pada **keamanan klinis (anti-malapraktik AI), pelacakan jejak (traceability), human-in-the-loop approval, dan standar integrasi HL7 FHIR**:

### A. Prioritas P0 — Keamanan & Ketepatan Informasi Klinis
1. **Traceable Phonetic Correction (`correct_with_trace`) & Protected Entity Sets:**
   - Menghasilkan output terpisah antara `original_text`, `corrected_text`, dan array `corrections`:
     `[{"original": "tanas", "corrected": "panas", "confidence": 0.88, "review_required": true, "reason": "..."}]`.
   - **Safeguard Mutasi Medis:** Dilarang keras memutasi nama obat (`PROTECTED_DRUG_NAMES`), dosis obat (`DOSAGE_PATTERN`), angka tanda vital (`VITAL_NUM_PATTERN`), dan diagnosis klinis (`PROTECTED_DIAGNOSES`).
2. **Multi-Role Speaker Diarization & Isolasi Sumber:**
   - Mendukung 5 peran pembicara: `doctor`, `patient`, `nurse`, `companion`, `unknown`.
   - Output segmen dilengkapi: `speaker_id` (`speaker_1`, `speaker_2`, dll), `speaker_role`, `start_time`, `end_time`, `confidence`, `source`.
   - Pertanyaan dokter dan tuturan pendamping tidak otomatis masuk sebagai keluhan langsung pasien.
3. **Preservasi Negasi Gejala & Alergi Terstandarisasi:**
   - Menangkap negasi kata ("tidak panas", "bebas batuk") dan menandainya sebagai `status: "absent"`.
   - Status alergi distandarisasi ke 5 status: `reported`, `denied`, `unknown`, `not_asked`, `uncertain`.
4. **Clinical Data Validator (`ClinicalDataValidator`):**
   - Memvalidasi konsistensi klinis pra-EMR: keluhan utama wajib ada jika pasien mengeluh, pemeriksaan batas fisiologis ekstrem, penelusuran kata durasi ke teks sumber, dan pencegahan submit data cacat.

### B. Prioritas P1 — Kelengkapan Ekstraksi & Struktur Schema Konsisten
1. **Keluhan Utama Terstruktur (`chief_complaint`):**
   - Atribut: `value`, `body_part` (pemetaan anatomis), `onset`, `duration`, `severity` (berat/ringan), `source`, `review_required`.
2. **Normalisasi Durasi & Waktu (`duration`):**
   - Atribut: `original_text`, `value` (integer), `unit` (hour/day/week/month/year), `is_approximate` (boolean), `event_time`, `source`.
3. **Ekstraksi Tanda Vital Lengkap (8 Parameter):**
   - Tekanan Darah (mmHg), Suhu Tubuh (°C), SpO2 (%), Nadi (bpm), Frekuensi Napas (x/min), Berat Badan (kg), Tinggi Badan (cm), Skala Nyeri (0–10). Dilengkapi validasi rentang normal.
4. **Decoupled Multi-Dimensional Confidence & Review Trigger:**
   - Memisahkan confidence skor: `transcription`, `speaker`, `clinical_extraction`, `normalization`.
   - Daftar `review.reasons` eksplisit untuk setiap item yang membutuhkan konfirmasi dokter.
5. **Struktur EMR Terpadu (16 Bagian Section 10):**
   - Mempertahankan 100% backward compatibility untuk field lama (`fields`, `keluhan_utama`, `keluhan_tambahan`, `lama_sakit`, `raw_transcript`, `cleaned_transcript`).

### C. Prioritas P2 — Kesiapan Operasional, Integrasi EMR & Keamanan
1. **EMR Integration Abstraction (`emr_integration.py`):**
   - `BaseEMRAdapter` (interface).
   - `MockHospitalEMRAdapter`: Dilengkapi **Approval Gatekeeper** (menolak transmisi jika status bukan `approved`), proteksi **Idempotency** (mencegah duplikasi data saat re-submit), dan **Safe Retry Mechanism**.
   - `FHIRAdapter`: Mengonversi draft yang disetujui menjadi standar internasional **HL7 FHIR Bundle** (resources: `Encounter`, `Condition`, `Observation`, `AllergyIntolerance`).
2. **Security & Audit Trail Module (`security_audit.py`):**
   - Mencatat seluruh lifecycle (`AUDIO_INGESTED`, `TRANSCRIPTION_COMPLETED`, `CLINICAL_EXTRACTED`, `STAFF_APPROVED`, `EMR_EXPORT_SUCCESS`).
   - Dilengkapi fungsi `mask_pii()` untuk menyamarkan NIK, nomor telepon, dan data pribadi di log audit.
   - SHA256 integrity checksum pada setiap record log.
3. **Modul Evaluasi Kualitas Sistem (`evaluation.py`):**
   - Kalkulasi metrik terstandarisasi: WER, CER, Precision, Recall, F1-Score ekstraksi gejala, Negation Accuracy, Vitals Exact Match, Allergy Status Match, Field Completeness.
4. **Human-in-the-Loop Web API (`web_app.py`):**
   - `POST /api/emr/review`: Dokter dapat menyetujui (`approve`), menolak (`reject`), atau memperbarui (`update`) isi draft.
   - `POST /api/emr/export`: Mengirimkan draft yang disetujui ke EMR adapter (FHIR / Mock).
   - `GET /api/emr/session/{session_id}`: Mengambil draft lengkap.
   - `GET /api/emr/audit/{session_id}`: Menampilkan audit trail sesi.
   - `GET /api/evaluation/benchmark`: Menjalankan evaluasi benchmark real-time.

### D. Hasil Pengujian Otomatis (100% Passed)
Seluruh modul telah diuji melalui unit test otomatis:
- `test_hospital_poc_p0.py` : **10 test PASSED** (Koreksi berjejak, proteksi obat/dosis, speaker multi-role, negasi, alergi terstandarisasi, validator klinis).
- `test_hospital_poc_p1.py` : **6 test PASSED** (Keluhan utama terstruktur, normalisasi durasi, TTV 8 parameter, decoupled confidence, schema 16 bagian).
- `test_hospital_poc_p2.py` : **8 test PASSED** (EMR gatekeeper, review approval lifecycle, idempotency, konversi FHIR bundle, audit logging PII masking, evaluasi WER/F1).
- `test_suite_all.py` & `test_conversation_maximal.py` : **Semua test kompatibilitas lama PASSED**.

## 7. Audit Menyeluruh, Dataset Sintetis, dan Evaluasi Kualitas (v3.1)

Menindaklanjuti audit sistem rumah sakit:
1. **Perbaikan Rekaman Audio & Penanganan Mutasi Kata:**
   - Prompt Whisper disederhanakan menjadi prompt klinis Bahasa Indonesia alami (`INDONESIAN_CLINICAL_PROMPT`).
   - VAD diatur seimbang (threshold 0.38, padding 350ms, min speech 150ms) agar konsonan halus di akhir kata (`-nya`, `panas`, `batuk`, `dok`) tidak terpotong.
   - Menghapus aturan fonetik berbahaya (`\bilang\b -> hilang`, `\bdokter\b -> dok`, `\bcari obat\b -> minum obat`).
   - Mengganti regex deduplikasi liar dengan `_clean_stutters()` yang melestarikan kata ulang sah (*tiba-tiba, kira-kira, kadang-kadang, batuk-batuk, gatal-gatal, muntah-muntah, bentol-bentol*).
2. **Koreksi Ucapan Pasien (Self-Correction):**
   - Parser durasi kini mendukung deteksi ralat waktu (*"Bukan sejak kemarin, tetapi sejak tadi pagi"* -> durasi *"Sejak Tadi Pagi"*).
3. **Negasi Antar-Giliran (Cross-Turn Dialogue Anamnesis):**
   - Pertanyaan dokter ("Apakah ada batuk?") yang disangkal pasien ("Tidak ada dok") berhasil dipetakan sebagai gejala negatif (*absent*).
4. **Dataset Sintetis & Test Suite Komprehensif:**
   - `synthetic_dataset.py`: 17 skenario percakapan sintetis (A–H) sesuai standar UU PDP / HIPAA.
   - `test_modules.py`: 14 pengujian unit terpisah untuk setiap tahap pipeline (14/14 PASS).
   - `test_clinical_fidelity.py`: 10 aturan keselamatan klinis (10/10 PASS).
   - `test_end_to_end.py`: Pengujian alur audio -> EMR draft pada GPU CUDA (2/2 PASS).
   - Total pengujian otomatis: **50/50 test PASSED (0 FAIL)**.
5. **Evaluasi Kualitas Klinis Objektif (`evaluate_quality.py`):**
   - Precision: 83.33% | Recall: 71.43% | F1-Score: 76.92%
   - Negation Accuracy: 100.00% (6/6)
   - Vital Signs Accuracy: 100.00% (2/2)
   - Allergy Accuracy: 100.00% (4/4)
   - Duration & Self-Correction Accuracy: 100.00% (9/9)
   - EMR Transmit & Idempotency: 100.00%
6. **Laporan Audit Resmi:**
   - Rincian lengkap terdokumentasi di `AUDIT_AND_EVALUATION_REPORT.md`.

## 8. Perbaikan Penangkapan Suara, Chunking & STT Pipeline (v3.2)

Berdasarkan investigasi hilangnya ucapan dokter/pasien pada rekaman:
1. **Audio Inspection & Adaptive Software Gain Boost (`audio_processor.py`):**
   - Mendeteksi RMS (dBFS), peak level, clipping, hening, dan durasi audio sebelum diserahkan ke STT.
   - Mengaplikasikan Adaptive Gain Boost otomatis (+18dB max) untuk audio bersuara sangat pelan (-50 s.d. -38 dBFS) agar tertangkap jelas oleh Faster-Whisper.
2. **Optimalisasi VAD & Dual-Pass VAD Fallback (`transcriber.py`):**
   - VAD lebih sensitif (threshold 0.30, speech duration 120ms, silence duration 800ms, padding 400ms).
   - Relaksasi `no_speech_threshold=0.80` dan `log_prob_threshold=-1.5`.
   - **Dual-Pass Fallback:** Jika VAD menyaring audio tetapi sinyal memiliki energi akustik (RMS > -48 dBFS), otomatis menjalankan pass kedua tanpa VAD filter agar ucapan pasien tidak hilang.
3. **Penyangga Audio Chunking & Idempotent Streaming (`audio_chunker.py`):**
   - Mengelola chunk audio berurutan dengan sequence ID, gap detection, dan SHA256 checksum duplicate protection.
   - Perakitan (assembly) chunk menjadi file WAV terpadu tanpa distorsi sambungan.
4. **Pembaruan Antarmuka Capture & Visualizer (`static/index.html`):**
   - Dropdown pemilihan mikrofon via `enumerateDevices()`.
   - Live VU Meter bar & dBFS indicator via Web Audio API (`AudioContext` + `AnalyserNode`).
   - Peringatan instan saat mikrofon ter-mute atau volume terlalu rendah.
   - 4-Step Pipeline Status Grid (Audio Terdeteksi, Audio Terkirim, STT Diproses, Teks Dihasilkan).
   - Streaming chunk upload setiap 2000ms dengan fallback otomatis ke full blob.
5. **Hasil Pengujian & Evaluasi Kualitas STT:**
   - `test_audio_stt.py`: **10 / 10 PASS (100%)**.
   - `evaluate_audio_stt.py`: WER **9.09%**, CER **3.62%**, Audio Coverage **100.00%**, Empty Transcription Rate **0.00%**, Speech Detection **100.00%**, Latensi **1.26s** (RTF 0.129x pada RTX 3050 CUDA FP16).
   - Dokumentasi lengkap tercatat di `AUDIO_CAPTURE_AND_STT_REPORT.md`.

## 9. Peningkatan Kejelasan Penangkapan Suara & Akurasi STT Maksimal (v3.3)

Fokus utama penyempurnaan penangkapan suara menjadi teks yang jernih, sensitif, dan bebas halusinasi:
1. **Penyaringan Akustik 80Hz High-Pass Filter (`audio_processor.py`):**
   - Mengaplikasikan filter Butterworth order 2 (cut-off 80Hz) via `scipy.signal`.
   - Mengeliminasi DC offset, hum listrik AC (50Hz/60Hz), gesekan mikrofon, dan hembusan napas/wind pop di bawah 80Hz tanpa meredam frekuensi dasar vokal manusia (85–255Hz).
2. **Speech-Aware Adaptive Loudness Leveling (`audio_processor.py`):**
   - Menghitung RMS khusus pada frame bicara aktif (*active speech frames*, > -46 dBFS) dan bukan pada jeda hening.
   - Melakukan normalisasi adaptif ke target -19.0 dBFS (titik optimal sensitivitas log-mel filterbank Whisper).
   - Soft limiter peak anti-distortion (dibatasi di level 0.92 / -0.7 dBFS) agar suara keras tidak mengalami clipping.
3. **Peningkatan Kualitas Audio Browser (`static/index.html`):**
   - Mengganti `sampleRate: 16000` menjadi `{ ideal: 48000 }` (menghindari distorsi aliasing dari downsampler browser; downsampling ke 16kHz dilakukan secara presisi oleh PyAV libswresample di backend).
   - Menonaktifkan `noiseSuppression: false` pada WebRTC (mencegah pemotongan suku kata awal/akhir konsonan lunak akibat spectral gating browser).
   - Meningkatkan Opus bitrate dari 64 kbps ke **128 kbps** untuk kejernihan sibilan dan frikatif (s, f, t, c, sy).
   - Memasang badge kejelasan suara dinamis pada VU Meter:
     - 🟢 *Level Optimal (Sangat Jelas)*: > -26 dBFS
     - 🟡 *Agak Pelan (Bisa Lebih Dekat)*: -42 s.d. -27 dBFS
     - 🔴 *Terlalu Pelan / Hening*: < -43 dBFS
4. **Prompt Prior Klinis Deskriptif & Anti-Prompt Bleeding (`transcriber.py`):**
   - Mengganti format skrip dialog dengan prompt perbendaharaan kata klinis deskriptif:
     `"Transkripsi percakapan klinis dokter dan pasien di rumah sakit Indonesia. Keluhan utama, anamnesa, riwayat penyakit, gejala demam, pusing, batuk pilek, sesak napas, mual muntah, nyeri perut, lemas, pemeriksaan fisik, tensi darah, suhu tubuh, denyut nadi, dan riwayat alergi obat."`
   - Mencegah fenomena *prompt bleeding* dan *looping* (misal repetisi "Pasien 3.5 C, nadi...").
   - Parameter decoding stabil: `temperature=0.0`, `beam_size=5`, `repetition_penalty=1.08`.
5. **Hasil Benchmark STT Terverifikasi (`evaluate_audio_stt.py`):**
   - **Word Error Rate (WER): 3.64%** (Turun signifikan dari 9.09%).
   - **Character Error Rate (CER): 2.90%** (Turun dari 3.62%).
   - **Audio Coverage Rate: 100.00%** (Seluruh ucapan terdeteksi utuh).
   - **Empty Transcription Rate: 0.00%**.
   - **Kecepatan Proses: 1.08 detik** (RTF: 0.111x real-time di GPU RTX 3050 CUDA FP16).

## 10. Perbaikan Ekstraksi Gejala Tenggorokan & Penanganan Kata Keterangan (v3.4)

Berdasarkan temuan kasus dialog medis di mana keluhan `"tenggorokan juga agak sakit"` tidak tercatat di Keluhan Tambahan:
1. **Perbaikan Substring Negasi pada Kata Keterangan Adverbia (`medical_extractor.py`):**
   - Fungsi `_is_symptom_negated` sebelumnya melakukan pencarian substring sederhana `neg in t` untuk kata negasi seperti `"gak"`.
   - Karena `"gak"` merupakan substring dari kata `"agak"` (`a-gak`), setiap frasa keluhan yang mengandung adverbia `"agak"` (misalnya `"agak sakit"`, `"tenggorokan juga agak sakit"`, `"agak lemas"`) keliru diklasifikasikan sebagai keluhan dinegasikan (*absent*).
   - Diperbaiki menggunakan batas kata regex `\b(?:tidak|nggak|gak|ngga|bukan|tanpa|bebas|belum)\b` sehingga kata `"agak"` tidak lagi memicu negasi.
2. **Fleksibilitas Pola Gejala Tenggorokan (`SECONDARY_PATTERNS`):**
   - Regex keluhan tenggorokan sebelumnya kaku (`\bsakit\s+tenggorokan\b|\btenggorokan\s+sakit\b`), sehingga gagal mencocokkan frase dengan kata sisipan/adverbia seperti `"tenggorokan juga agak sakit"`.
   - Diperluas menjadi `\b(?:sakit|nyeri|radang|perih)\s+(?:pada\s+|di\s+)?tenggorokan\b|\btenggorokan(?:\s+\w+){0,3}\s+(?:sakit|nyeri|perih|radang)\b`.
3. **Penyelarasan Cross-Turn Anamnesis:**
   - Memastikan jawaban pasien yang mengonfirmasi gejala (misal *"Ada sedikit batuk..."*) tidak keliru dianggap penolakan cross-turn hanya karena mengandung kata *"tidak sering/tidak parah"*.
   - Melindungi gejala berstatus `present` agar tidak ditimpa menjadi `absent`.
4. **Hasil Pengujian & Regresi Klinis:**
   - `test_clinical_fidelity.py`: **12/12 PASS (100%)**.
   - Dialog uji coba berhasil mengekstrak `Keluhan Tambahan: Batuk, Badan lemas, Sakit tenggorokan` dan mual/muntah berstatus disangkal (*absent*).

## 11. Parsing Angka Numerik Durasi / Lama Sakit (Tahun / Bulan / Hari) untuk CoreERP EMR (v3.5)

Berdasarkan kebutuhan integrasi formulir rekam medis CoreERP (arahan Divavava):
1. **Dukungan Segmentasi Numerik Durasi (`medical_extractor.py`):**
   - Formulir rekam medis rumah sakit CoreERP membagi input `LAMA SAKIT` menjadi 3 kolom angka terpisah: `[ ] Thn  [ ] Bln  [ ] Hari`.
   - Mengimplementasikan parser klinis `_parse_duration_to_numbers()` yang secara otomatis mengonversi frasa durasi percakapan pasien/dokter ke angka:
     - *"Sejak kemarin sore"* / *"semalam"* / *"tadi pagi"* → `Hari: 1` (`thn: ""`, `bln: ""`, `hari: "1"`)
     - *"Kemarin lusa"* → `Hari: 2`
     - *"Sudah 3 hari"* → `Hari: 3`
     - *"2 minggu"* → `Hari: 14` (konversi minggu ke hari)
     - *"1 bulan"* → `Bln: 1`
     - *"1 tahun 2 bulan"* → `Thn: 1`, `Bln: 2`
2. **Dukungan Durasi Gabungan (*Compound Duration*):**
   - Menambahkan pola regex durasi majemuk di `DURATION_PATTERNS` untuk menangkap kombinasi multi-satuan seperti *"1 tahun 2 bulan"*, *"1 bulan 5 hari"*, dsb.
3. **Penyelarasan API & Payload EMR (`web_app.py`):**
   - Field `lama_sakit_thn`, `lama_sakit_bln`, `lama_sakit_hari`, serta `duration_parsed` disertakan pada response JSON `fields` dan `duration` dengan 100% backward compatibility.
4. **Pembaruan Antarmuka Web (`static/index.html`):**
   - Menambahkan segmented input group `[ ] Thn  [ ] Bln  [ ] Hari` persis sesuai desain form CoreERP.
   - Mengisi nilai secara otomatis (*auto-populate*) saat audio selesai dianalisis.
   - Mendukung penyimpanan koreksi manual tenaga medis (`saveStaffCorrections`).
5. **Hasil Pengujian & Regresi:**
   - `test_clinical_fidelity.py`: **13/13 PASS (100%)**.
   - `test_modules.py`: **14/14 PASS (100%)**.
   - `evaluate_quality.py`: Duration & Self-Correction Accuracy **100.00% (9/9)**.

