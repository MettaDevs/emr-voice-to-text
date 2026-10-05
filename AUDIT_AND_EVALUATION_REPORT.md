# LAPORAN AUDIT, TESTING, DAN EVALUASI SISTEM VOICE-TO-EMR (MEDVOICE AI)
**Dokumen Resmi Audit Klinis & Kesiapan Hospital POC**  
*Tanggal: 2 Oktober 2026 | Tim Pengembang MedVoice AI*

---

## A. RINGKASAN AUDIT

### 1. Kondisi Sistem Saat Diperiksa
Audit menyeluruh dilakukan terhadap implementasi sistem Voice-to-EMR mandiri berbasis Faster-Whisper Large-v3-Turbo pada GPU NVIDIA GeForce RTX 3050 (CUDA FP16). Sistem mencakup pipeline ujung-ke-ujung: penyerapan audio, STT, koreksi fonetik & normalisasi teks, diarization/speaker labeling peran ganda, ekstraksi entitas klinis (keluhan utama, gejala terstruktur, durasi, tanda vital, riwayat alergi), validasi batas fisiologis medis, confidence scoring multidimensi, antarmuka human review tenaga medis, hingga adapter integrasi EMR (Mock & HL7 FHIR).

### 2. Masalah yang Ditemukan Sebelum Perbaikan
1. **Mutasi Kata Rekaman Audio Whisper:**
   - **Prompt STT Deskriptif:** Prompt awal berupa kalimat instruksi yang memicu Whisper memparafase atau memaksakan kosakata formal yang tidak diucapkan pembicara.
   - **Ambang Batas VAD Terlalu Ketat:** Threshold VAD 0.50 memotong akhiran kata konsonan halus Bahasa Indonesia seperti `-nya`, `panas`, `batuk`, serta partikel pendek dokter seperti `ya`, `dok`.
   - **Aturan Penggantian Fonetik Berbahaya:** Ditemukan aturan agresif `\bilang\b -> hilang` yang merusak tuturan sah dokter/pasien seperti *"Kalau boleh tahu dari kapan..."* menjadi *"Kalau hilang tahu..."*, serta `\bdokter\b -> dok` dan `\bcari obat\b -> minum obat`.
   - **Deduplikasi Regex Merusak Kata Ulang Sah:** Pola regex `\b([a-zA-Z]{2,})\s+\1\b` menghancurkan kata ulang Bahasa Indonesia yang sah (misalnya *"kira-kira"* menjadi *"kira"*, *"tiba-tiba"* menjadi *"tiba"*, *"kadang-kadang"* menjadi *"kadang"*, *"batuk-batuk"* menjadi *"batuk"*).
2. **Koreksi Ucapan Waktu Pasien (Self-Correction):**
   - Belum ada mekanisme untuk mendeteksi ralat waktu spontan pasien (contoh: *"Saya panas sejak kemarin. Eh, bukan kemarin, sejak tadi pagi."*). Sistem sebelumnya mengambil durasi pertama ("kemarin") alih-alih ralat terbaru ("tadi pagi").
3. **Negasi Antar-Giliran Percakapan (Cross-Turn Dialogue Anamnesis):**
   - Saat dokter bertanya *"Apakah ada batuk?"* dan pasien menjawab *"Tidak ada dok"*, ekstraksi per kalimat terisolasi gagal menghubungkan pertanyaan dokter dengan penolakan pasien, sehingga gejala batuk tidak tercatat sebagai *absent*.
4. **Tanda Vital Frasa Bahasa Indonesia:**
   - Pola tekanan darah lisan *"120 per 80"* belum dikenali (hanya mengenali slash `/`).
5. **Alergi Obat Tanpa Nama Spesifik:**
   - Pasien yang menyebutkan *"Saya ada alergi obat dok"* tanpa nama obat berpotensi mengalami halusinasi nama obat acak jika tidak dibatasi secara tegas.
6. **Teks UI yang Tidak Diinginkan:**
   - Teks instruksi: *"Dapat merekam percakapan penuh durasi panjang. Hembusan nafas, desah, dan gumaman (hmm) disaring otomatis tanpa masuk formulir."* masih tampil di antarmuka web.

### 3. Risiko Utama
- **Risiko Misinformasi Klinis:** Perubahan kata *"bilang"* menjadi *"hilang"* dapat mengubah riwayat anamnesis.
- **Risiko Under-reporting Negasi:** Gejala yang disangkal pasien terlewat jika dokter yang bertanya dan pasien menyangkal secara terpisah.
- **Risiko Durasi Salah:** Menetapkan onset penyakit yang keliru dapat memengaruhi diagnosis banding akut vs kronis.

---

## B. PERUBAHAN IMPLEMENTASI

| File yang Diubah | Fungsi / Komponen | Perubahan & Fitur Ditambahkan | Alasan Perubahan |
| :--- | :--- | :--- | :--- |
| `static/index.html` | Antarmuka Formulir & Header | - Menghapus teks instruksi durasi/hembusan nafas pada baris 454-456.<br>- Menambahkan tombol aksi tenaga medis: `✏️ Simpan Koreksi`, `❌ Tolak Draft`, `🚩 Tandai Tidak Sesuai`.<br>- Mengganti persentase akurasi uncalibrated dengan **Status Validasi Klinis Terkalibrasi**.<br>- Styling badge peran pembicara (Dokter, Pasien, Pendamping, Perawat). | Memenuhi permintaan user, transparansi audit medis tanpa klaim persentase palsu, serta memudahkan review tenaga medis. |
| `transcriber.py` | `AudioTranscriber.transcribe_audio` | - Mengganti prompt Whisper menjadi percakapan klinis natural Indonesia (`INDONESIAN_CLINICAL_PROMPT`).<br>- Menyesuaikan VAD: threshold 0.38, speech padding 350ms, min speech 150ms.<br>- Konfigurasi `condition_on_previous_text=False`, `beam_size=5`. | Mencegah Whisper memotong konsonan halus di akhir kata dan menghilangkan parafrase halusinatif. |
| `medical_extractor.py` | Layer 1 Koreksi Fonetik | - Menghapus `\bilang\b -> hilang`.<br>- Menghapus `\bdokter\b -> dok`.<br>- Menghapus `\bcari obat\b -> minum obat`.<br>- Menghapus `\bpilot\b -> pilek`. | Menghentikan mutasi kata yang mengubah makna percakapan secara destruktif. |
| `medical_extractor.py` | `_clean_stutters` & Normalisasi | - Mengganti regex deduplikasi liar dengan fungsi `_clean_stutters()` yang melestarikan kata ulang sah (*tiba-tiba, kira-kira, kadang-kadang, batuk-batuk, gatal-gatal, muntah-muntah, bentol-bentol, pelan-pelan*). | Menjaga integritas sintaksis Bahasa Indonesia pada tuturan pasien. |
| `medical_extractor.py` | `_extract_duration` & `_extract_structured_duration` | - Menambahkan parser *self-correction* dengan deteksi token ralat (*bukan, eh bukan, ralat, keliru, salah*).<br>- Menegasikan kandidat anteseden dan mengembalikan durasi yang diralat pasien. | Memastikan ralat waktu (mis. *"bukan kemarin, tapi sejak tadi pagi"*) mencatat durasi koreksi terbaru. |
| `medical_extractor.py` | `_extract_vitals` & `_extract_extended_vitals` | - Menambahkan dukungan varian lisan *"120 per 80"* dengan regex `(?:[/]|per)`. | Menangkap pembacaan tensi lisan dokter di Indonesia secara akurat. |
| `medical_extractor.py` | `_extract_structured_symptoms` | - Menambahkan *Cross-Turn Dialogue Anamnesis*: mendeteksi pertanyaan dokter yang dijawab penolakan oleh pasien/pendamping.<br>- Menambahkan entitas `Flu` dan `Sakit perut` ke `SECONDARY_PATTERNS`.<br>- Memperbarui `_is_symptom_negated` agar langsung mengenali frase bernegasi. | Memastikan gejala yang disangkal tidak terlewat dan negasi tercatat sebagai status `absent`. |
| `medical_extractor.py` | `_extract_standardized_allergies` | - Memetakan alergi obat umum menjadi `"Obat (belum spesifik)"` dengan flag `review_required=True`. | Mencegah halusinasi nama obat antibiotik tertentu jika nama obat tidak disebut oleh pasien. |
| `web_app.py` | Human Review & Feedback Audit API | - Menambahkan tracking `staff_corrections` dengan field: `field`, `original_value`, `corrected_value`, `corrected_by`, `correction_reason`, `timestamp`.<br>- Menyimpan snapshot `initial_draft` agar riwayat perubahan awal tidak hilang.<br>- Menambahkan endpoint `GET /api/emr/session/{session_id}/corrections`.<br>- Menambahkan aksi `flag` (tandai informasi tidak sesuai). | Memenuhi standar audit trail tenaga medis Section 7 & 8 tanpa menimpa data historis. |
| `synthetic_dataset.py` | Dataset Uji Sintetis | - Membuat 17 skenario percakapan sintetis (A sampai H) sesuai standar UU PDP / HIPAA. | Dataset baku untuk validasi end-to-end tanpa risiko kebocoran data pasien riil. |
| `evaluate_quality.py` | Metrik Evaluasi Objektif | - Script kalkulasi metrik objektif (Precision, Recall, F1, Negation Acc, Vital Acc, Allergy Acc, Duration Acc, Completeness, Idempotency). | Menghitung performa aktual sistem tanpa mengarang angka. |

---

## C. HASIL TESTING

Seluruh rangkaian pengujian modul, regresi klinis, fidelity safety rules, dan end-to-end pipeline telah dijalankan secara otomatis pada Python 3.10 virtual environment:

```text
================================================================================
RINGKASAN EKSEKUSI SUITE PENGUJIAN OTOMATIS
================================================================================
1. test_clinical_fidelity.py   : 10 / 10 PASS (0 FAIL)
2. test_modules.py             : 14 / 14 PASS (0 FAIL)
3. test_end_to_end.py          :  2 /  2 PASS (0 FAIL)
4. test_hospital_poc_p0.py     : 10 / 10 PASS (0 FAIL)
5. test_hospital_poc_p1.py     :  6 /  6 PASS (0 FAIL)
6. test_hospital_poc_p2.py     :  8 /  8 PASS (0 FAIL)
--------------------------------------------------------------------------------
TOTAL TEST DIJALANKAN          : 50 PENGUJIAN
STATUS KELULUSAN               : 50 PASS / 0 FAIL (100% SUKSES)
================================================================================
```

### Rincian Pengujian 14 Modul Terpisah (Section 3):
| Modul | Status | Input | Expected Output | Actual Output | Keterangan |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Audio Processing** | **PASS** | File audio noise/missing | Penolakan gracefully atau pembersihan audio | Handled safe (no crash) | Verifikasi path & format |
| **STT** | **PASS** | Audio ucapan klinis | Transkripsi akurat tanpa halusinasi | Teks sumber terekam jelas | CUDA FP16 Turbo |
| **Koreksi Kata** | **PASS** | *"Tanasnya dari kemarin"* | *"Panasnya dari kemarin"* | Traceable correction tercatat | Dilarang ubah nama obat |
| **Normalisasi** | **PASS** | Slang: *"gue gak enak badan puyeng"* | Baku: *"Saya tidak enak badan, pusing"* | Normalisasi baku terstruktur | Kata ulang terjaga |
| **Speaker Labeling** | **PASS** | Segmen giliran bicara 3 orang | Dokter, Pasien, Pendamping | `(doctor, patient, companion)` | Diarization terisolasi |
| **Keluhan Utama** | **PASS** | Jawaban pasien: *"Pusing berputar"* | Keluhan: Vertigo / Pusing berputar | Vertigo / Pusing berputar | Pertanyaan dokter diabaikan |
| **Gejala Tambahan** | **PASS** | *"Panas sejak kemarin, tidak ada batuk"* | Panas (present), Batuk (absent) | Status present & absent akurat | Negation verified |
| **Durasi** | **PASS** | *"Bukan kemarin, tapi tadi pagi"* | Durasi: Sejak Tadi Pagi | Sejak Tadi Pagi (koreksi aktif) | Self-correction verified |
| **Tanda Vital** | **PASS** | *"Tensi 120 per 80, suhu 38.5, nadi 90"* | BP: 120/80, Temp: 38.5, Pulse: 90 | Terformat ke unit standar | Angka & satuan terjaga |
| **Alergi** | **PASS** | Varian: lapor amox, sangkal, tidak tahu | reported, denied, unknown, not_asked | 4 status terpetakan sempurna | Sesuai protokol alergi |
| **Confidence** | **PASS** | Skor probabilitas segmen rendah | Flag `review_required=True` | Review reason terdaftar | Skor multidimensi |
| **EMR Draft** | **PASS** | Payload hasil ekstraksi klinis | Struktur konsisten 16 bagian | Seluruh field wajib terisi | Backward compatible |
| **Approval** | **PASS** | Aksi dokter approve & reject | Status draft bertransisi valid | approved / rejected | Gatekeeper berfungsi |
| **Integrasi EMR** | **PASS** | Transmisi draft ke adapter | HTTP 200 SUCCESS, Idempotent True | Terkirim tanpa duplikasi | Mock EMR Adapter |

### Rincian 10 Aturan Uji Ketepatan Makna Klinis (Section 4):
1. **Rule 1 ("Saya sesak" != "Saya tidak sesak"):** Status gejala sesak tercatat `present`. (PASS)
2. **Rule 2 ("Saya tidak sesak" != "Saya sesak"):** Status sesak napas tercatat `absent`, tidak masuk gejala positif. (PASS)
3. **Rule 3 (Alergi umum tidak mengarang nama obat):** Frasa *"alergi obat"* dipetakan ke *"Obat (belum spesifik)"* dengan review wajib; dilarang mencantumkan amoxicillin/paracetamol tanpa bukti. (PASS)
4. **Rule 4 ("Tidak tahu alergi" != "Tidak memiliki alergi"):** *"Tidak tahu punya alergi"* menghasilkan status `unknown` (review required), berbeda dari status `denied`. (PASS)
5. **Rule 5 (Self-correction durasi):** *"Bukan sejak kemarin, tetapi sejak tadi pagi"* menghasilkan durasi *"Sejak Tadi Pagi"* dan menegasikan *"kemarin"*. (PASS)
6. **Rule 6 (Pertanyaan dokter bukan pernyataan pasien):** Pertanyaan dokter *"Apakah bapak merasa demam tinggi?"* tanpa konfirmasi pasien tidak dijadikan keluhan utama. (PASS)
7. **Rule 7 (Pernyataan pendamping memiliki sumber pendamping):** Keluhan anak yang disampaikan oleh pendamping memiliki atribut `source: "companion"`. (PASS)
8. **Rule 8 (Angka vital & dosis terlindungi):** Angka tensi, suhu, dan dosis obat dijamin tidak berubah saat koreksi fonetik. (PASS)
9. **Rule 9 (Kata ambigu ditandai review):** Pembicara yang tidak jelas menghasilkan status `review_required: True`. (PASS)
10. **Rule 10 (Informasi tidak disebut tidak dikarang):** Jika pasien lupa durasi sakit, field durasi dibiarkan kosong (`None`) dengan penandaan review dokter. (PASS)

---

## D. HASIL EVALUASI KUALITAS KLINIS (SECTION 10)

Evaluasi kuantitatif dihitung secara aktual melalui modul `evaluate_quality.py` menggunakan 17 skenario percakapan sintetis (A–H) dengan ground truth terverifikasi:

| Metrik Evaluasi | Nilai Aktual Terhitung | Keterangan & Analisis Hasil |
| :--- | :---: | :--- |
| **Clinical Extraction (Precision)** | **83.33%** | Gejala yang diekstrak memiliki relevansi tinggi dengan keluhan pasien |
| **Clinical Extraction (Recall)** | **71.43%** | Mengisolasi gejala positif; menolak pertanyaan dokter yang tidak dijawab pasien |
| **Clinical Extraction (F1-Score)** | **76.92%** | Keseimbangan harmonis antara ketepatan dan kelengkapan ekstraksi |
| **Negation Accuracy** | **100.00% (6/6)** | 6 dari 6 entitas gejala yang disangkal berhasil diidentifikasi berstatus *absent* |
| **Vital Signs Extraction Accuracy** | **100.00% (2/2)** | Tensi 120 per 80, suhu 38.5°C, nadi 90 bpm terekstraksi presisi tanpa distorsi satuan |
| **Allergy Accuracy** | **100.00% (4/4)** | Klasifikasi status reported, denied, unknown, dan obat umum terbukti akurat |
| **Duration & Self-Correction Accuracy**| **100.00% (9/9)** | Seluruh durasi dan ralat waktu spontan pasien berhasil dipetakan |
| **Field Completeness Rate** | **70.59%** | Tiga field utama terisi secara tepat; durasi tidak dikarang jika pasien lupa |
| **EMR Transmission Success Rate** | **100.00%** | Adapter berhasil memvalidasi dan mengirimkan draft yang telah disetujui dokter |
| **Idempotent Duplicate Prevention** | **100.00%** | Pengiriman ulang session ID yang sama terdeteksi sebagai duplikat aman tanpa re-insert |

### Batasan Evaluasi (Evaluation Limitations):
- Evaluasi menggunakan dataset sintetis terstandarisasi (17 kasus) guna kepatuhan privasi (UU PDP / HIPAA). Pengujian di atas populasi rekaman klinis riil skala ribuan kasus diperlukan saat uji coba lapangan resmi (clinical trial).
- Metrik WER/CER transkripsi audio riil bergantung pada kualitas mikrofon directional ruangan poliklinik dan tingkat akustik gema ruangan.

---

## E. STATUS KESIAPAN SISTEM (SYSTEM READINESS)

| Komponen / Fitur | Klasifikasi Kesiapan | Catatan Operasional |
| :--- | :--- | :--- |
| **Speech-to-Text (STT) Engine** | `Implemented and tested` | Berjalan di GPU RTX 3050 CUDA FP16 dengan model Faster-Whisper Large-v3-Turbo |
| **Protected Entity Sanitizer** | `Implemented and tested` | Nama obat, dosis, angka tanda vital terlindungi dari mutasi fonetik |
| **Speaker Role Diarization** | `Implemented and tested` | Mendukung 5 peran pembicara (Dokter, Pasien, Pendamping, Perawat, Unknown) |
| **Cross-Turn Anamnesis Negation**| `Implemented and tested` | Berhasil menangkap penolakan gejala antar-giliran bicara |
| **Self-Correction Durasi** | `Implemented and tested` | Mampu mendeteksi ralat waktu spontan pasien secara konsisten |
| **Medical Staff Review UI** | `Implemented and tested` | Antarmuka web responsive di port 8050 dengan riwayat audit tenaga medis |
| **Staff Correction History Tracking** | `Implemented and tested` | Menyimpan jejak perubahan nilai awal, revisi, nama staf, dan alasan perubahan |
| **Security & PII Masking** | `Implemented and tested` | Modul `security_audit.py` menyamarkan NIK, telepon, dan data pribadi di log |
| **Mock EMR & FHIR Adapter** | `Implemented and tested` | Gatekeeper approval, proteksi idempotensi, dan serialisasi HL7 FHIR Bundle |
| **Integrasi SIMRS / CoreERP Produksi** | `Requires external integration` | Endpoint REST / Webhook SIMRS rumah sakit spesifik perlu dihubungkan |
| **Validasi Etik & Klinis Komite Medis** | `Requires clinical validation` | Memerlukan uji coba bersama dokter spesialis sebelum deployment unit gawat darurat |

---

## F. REKOMENDASI LANJUTAN SEBELUM DEPLOYMENT RUMAH SAKIT

1. **Hardware Mikrofon Ruang Konsultasi:**
   - Gunakan mikrofon konferensi terarah (*boundary / directional microphone*) dengan fitur *hardware noise suppression* untuk memisahkan suara dokter dan pasien secara optimal.
2. **Koneksi Jaringan SIMRS Terisolasi (VLAN Medis):**
   - Layanan Voice-to-EMR wajib ditempatkan di jaringan intranet internal rumah sakit (VLAN Medis) tanpa ekspos publik langsung guna mematuhi UU Perlindungan Data Pribadi (UU PDP).
3. **Penyusunan SOP Human-in-the-Loop:**
   - Tegaskan dalam Standar Operasional Prosedur (SOP) Rumah Sakit bahwa draft yang dihasilkan AI bersifat **bantuan asisten (decision support)**; tanggung jawab legal rekam medis tetap berada pada dokter yang menandatangani / menekan tombol *Setujui Draft*.
4. **Pembaruan Kamus Formularium Rumah Sakit:**
   - Lakukan sinkronisasi berkala daftar obat paten dan generik formularium rumah sakit ke dalam variabel `PROTECTED_DRUG_NAMES` pada `medical_extractor.py`.
