"""
synthetic_dataset.py — MedVoice AI Hospital POC Synthetic Test Dataset
=====================================================================
Dataset percakapan sintetis medis untuk pengujian kepatuhan klinis Voice-to-EMR.
Dilarang menggunakan data pasien asli (HIPAA / UU PDP compliant).

Mencakup 8 variasi klinis wajib:
  A. Percakapan normal
  B. Percakapan dengan koreksi ucapan (self-correction)
  C. Percakapan dengan negasi gejala
  D. Percakapan dengan alergi obat / makanan / denial / unknown
  E. Percakapan dengan tanda vital (TTV lengkap)
  F. Percakapan dengan informasi tidak lengkap
  G. Percakapan multi-speaker (Dokter, Pasien, Pendamping, Perawat)
  H. Percakapan dengan istilah rawan salah dengar / fonetik akustik
"""

SYNTHETIC_DATASET = [
    # ── A. Percakapan Normal ──────────────────────────────────────────────────
    {
        "id": "SYN-NORM-01",
        "category": "A_normal",
        "description": "Konsultasi demam standar dokter-pasien",
        "dialogue": (
            "Dokter: Selamat pagi pak, apa keluhan yang dirasakan hari ini? "
            "Pasien: Selamat pagi dok, saya merasa panas sejak kemarin. "
            "Dokter: Apakah ada batuk? "
            "Pasien: Tidak ada dok."
        ),
        "expected": {
            "chief_complaint": "Demam / Panas",
            "duration": "Kemarin",
            "present_symptoms": ["Demam / Panas"],
            "absent_symptoms": ["Batuk"],
            "vitals": {},
            "allergy_status": "not_asked"
        }
    },
    {
        "id": "SYN-NORM-02",
        "category": "A_normal",
        "description": "Konsultasi sakit kepala berdenyut dengan durasi 3 hari",
        "dialogue": (
            "Dokter: Selamat siang bu, keluhannya apa ya? "
            "Pasien: Kepala saya sakit berdenyut dok, sudah tiga hari ini. "
            "Dokter: Ada mual atau muntah bu? "
            "Pasien: Tidak ada mual sama sekali dok."
        ),
        "expected": {
            "chief_complaint": "Sakit kepala berdenyut",
            "duration": "Tiga Hari Ini",
            "present_symptoms": ["Sakit kepala berdenyut"],
            "absent_symptoms": ["Mual"],
            "vitals": {},
            "allergy_status": "not_asked"
        }
    },

    # ── B. Percakapan dengan Koreksi Ucapan (Self-Correction) ───────────────────
    {
        "id": "SYN-CORR-01",
        "category": "B_self_correction",
        "description": "Pasien mengoreksi durasi waktu sakit dari kemarin ke tadi pagi",
        "dialogue": (
            "Dokter: Sudah dari kapan mulai terasa tidak enak badannya? "
            "Pasien: Saya panas sejak kemarin. Eh, bukan kemarin, sejak tadi pagi."
        ),
        "expected": {
            "chief_complaint": "Demam / Panas",
            "duration": "Sejak Tadi Pagi",
            "negated_duration": "kemarin",
            "present_symptoms": ["Demam / Panas"]
        }
    },
    {
        "id": "SYN-CORR-02",
        "category": "B_self_correction",
        "description": "Pasien meralat keluhan dari flu menjadi pusing berputar",
        "dialogue": (
            "Dokter: Ada keluhan apa yang paling mengganggu? "
            "Pasien: Awalnya saya kira flu dok, tapi bukan flu dok, pusing kliyengan muter-muter sejak semalam."
        ),
        "expected": {
            "chief_complaint": "Pusing kliyengan",
            "duration": "Sejak Semalam",
            "absent_symptoms": ["Flu"],
            "present_symptoms": ["Pusing kliyengan"]
        }
    },
    {
        "id": "SYN-CORR-03",
        "category": "B_self_correction",
        "description": "Koreksi eksplisit bukan sejak kemarin tetapi sejak tadi pagi",
        "dialogue": (
            "Dokter: Mulai kapan demamnya? "
            "Pasien: Bukan sejak kemarin, tetapi sejak tadi pagi dok."
        ),
        "expected": {
            "chief_complaint": "Demam / Panas",
            "duration": "Sejak Tadi Pagi",
            "negated_duration": "kemarin"
        }
    },

    # ── C. Percakapan dengan Negasi Gejala ─────────────────────────────────────
    {
        "id": "SYN-NEG-01",
        "category": "C_negation",
        "description": "Pasien menyatakan tidak merasa sesak tetapi lemas",
        "dialogue": (
            "Dokter: Bagaimana napasnya pak? Ada rasa sesak atau sesak napas? "
            "Pasien: Saya tidak merasa sesak, tetapi badan terasa lemas sekali dok."
        ),
        "expected": {
            "chief_complaint": "Badan lemas",
            "present_symptoms": ["Badan lemas"],
            "absent_symptoms": ["Sesak napas"],
            "forbid_positive": ["Sesak", "Sesak napas"]
        }
    },
    {
        "id": "SYN-NEG-02",
        "category": "C_negation",
        "description": "Penyangkalan nyeri perut dan muntah pada kasus demam",
        "dialogue": (
            "Dokter: Apakah ada rasa sakit perut atau muntah bu? "
            "Pasien: Perut tidak sakit dan tidak ada muntah dok, hanya demam tinggi sejak kemarin lusa."
        ),
        "expected": {
            "chief_complaint": "Demam tinggi",
            "duration": "Sejak Kemarin Lusa",
            "absent_symptoms": ["Sakit perut", "Muntah"],
            "present_symptoms": ["Demam tinggi"]
        }
    },

    # ── D. Percakapan dengan Alergi Obat / Makanan ─────────────────────────────
    {
        "id": "SYN-ALG-01",
        "category": "D_allergy",
        "description": "Alergi spesifik obat amoxicillin yang dilaporkan pasien",
        "dialogue": (
            "Dokter: Apakah bapak ada riwayat alergi obat sebelumnya? "
            "Pasien: Saya alergi amoxicillin dok, kalau minum itu langsung gatal-gatal."
        ),
        "expected": {
            "allergy_status": "reported",
            "allergen": "Amoxicillin",
            "category": "drug",
            "reaction": "Gatal-Gatal",
            "review_required": False
        }
    },
    {
        "id": "SYN-ALG-02",
        "category": "D_allergy",
        "description": "Pasien menyangkal riwayat alergi",
        "dialogue": (
            "Dokter: Ada alergi obat atau makanan bu? "
            "Pasien: Tidak ada alergi dok, aman semua."
        ),
        "expected": {
            "allergy_status": "denied",
            "items": [],
            "review_required": False
        }
    },
    {
        "id": "SYN-ALG-03",
        "category": "D_allergy",
        "description": "Pasien menyatakan tidak tahu apakah ada alergi atau tidak",
        "dialogue": (
            "Dokter: Sebelumnya pernah alergi obat suntik atau tablet? "
            "Pasien: Saya tidak tahu punya alergi atau tidak dok, belum pernah tes."
        ),
        "expected": {
            "allergy_status": "unknown",
            "items": [],
            "review_required": True
        }
    },
    {
        "id": "SYN-ALG-04",
        "category": "D_allergy",
        "description": "Pasien menyebutkan alergi obat tanpa nama obat spesifik",
        "dialogue": (
            "Dokter: Apakah ada alergi obat pak? "
            "Pasien: Saya ada alergi obat dok, waktu itu dikasih antibiotik kapsul merah."
        ),
        "expected": {
            "allergy_status": "reported",
            "allergen_contains": "Obat",
            "forbid_hallucination": ["Amoxicillin", "Paracetamol", "Cefixime"],
            "review_required": True
        }
    },

    # ── E. Percakapan dengan Tanda Vital (TTV Lengkap) ────────────────────────
    {
        "id": "SYN-VIT-01",
        "category": "E_vitals",
        "description": "Tanda vital lisan: tensi 120 per 80, suhu 38.5 C, nadi 90",
        "dialogue": (
            "Dokter: Baik, hasil pemeriksaan fisik: Tekanan darah 120 per 80, "
            "suhu 38,5 derajat, dan nadi 90 kali per menit."
        ),
        "expected": {
            "blood_pressure": "120/80",
            "systolic": 120,
            "diastolic": 80,
            "temperature": 38.5,
            "pulse": 90,
            "source": "doctor"
        }
    },
    {
        "id": "SYN-VIT-02",
        "category": "E_vitals",
        "description": "Tanda vital perawat: SpO2 dan laju napas",
        "dialogue": (
            "Perawat: Lapor dok, tensi pasien 130/85 mmHg, saturasi oksigen 98 persen, "
            "dan frekuensi napas 20 kali per menit."
        ),
        "expected": {
            "blood_pressure": "130/85",
            "spo2": 98,
            "respiratory_rate": 20,
            "source": "nurse"
        }
    },

    # ── F. Percakapan dengan Informasi Tidak Lengkap ──────────────────────────
    {
        "id": "SYN-INC-01",
        "category": "F_incomplete",
        "description": "Pasien lupa waktu sakit — sistem dilarang mengarang durasi",
        "dialogue": (
            "Dokter: Sudah berapa lama merasa sakitnya? "
            "Pasien: Saya sudah merasa tidak enak badan, tapi lupa mulai kapan dok."
        ),
        "expected": {
            "chief_complaint": "Badan tidak enak",
            "duration": None,
            "review_required": True
        }
    },

    # ── G. Percakapan Multi-Speaker (Dokter, Pasien, Pendamping) ──────────────
    {
        "id": "SYN-SPK-01",
        "category": "G_multi_speaker",
        "description": "Pendamping menyampaikan keluhan anak ke dokter",
        "dialogue": (
            "Dokter: Selamat pagi bu, siapa yang sakit? "
            "Pendamping: Anak saya badannya panas dok sejak kemarin malam. "
            "Dokter: Ada batuk atau muntah adiknya? "
            "Pasien: Batuk sedikit dok. "
            "Pendamping: Iya dok, batuknya tadi pagi baru muncul."
        ),
        "expected": {
            "companion_turns": 2,
            "doctor_turns": 2,
            "patient_turns": 1,
            "chief_source": "companion",
            "chief_complaint": "Demam / Panas"
        }
    },

    # ── H. Istilah Berpotensi Salah Dikenali (Phonetic Challenge) ─────────────
    {
        "id": "SYN-PHON-01",
        "category": "H_phonetic",
        "description": "Fonetik mic berderau: tanasnya, kau belah tahu, batu pilek",
        "dialogue": (
            "Dokter: Kalau boleh tahu keluhannya apa? "
            "Pasien: Tanasnya dari kemarin malam dok, terus batu pilek juga."
        ),
        "expected": {
            "chief_complaint": "Demam / Panas",
            "secondary": ["Batuk pilek"],
            "duration": "Dari Kemarin Malam"
        }
    },
    {
        "id": "SYN-PHON-02",
        "category": "H_phonetic",
        "description": "Perut melilit dan mual muntah dengan kata sehari-hari",
        "dialogue": (
            "Dokter: Apa yang dirasakan bapak? "
            "Pasien: Perut saya melilit dok, terus enek mau muntah sudah dua hari."
        ),
        "expected": {
            "chief_complaint": "Perut melilit",
            "secondary": ["Mual", "Muntah"],
            "duration": "Dua Hari"
        }
    }
]

def get_dataset_by_category(category_prefix: str) -> list:
    """Ambil subset dataset berdasarkan prefix kategori (mis. 'A', 'B', 'C')."""
    return [d for d in SYNTHETIC_DATASET if d["category"].startswith(category_prefix)]

def get_all_dataset() -> list:
    """Ambil seluruh dataset sintetis."""
    return SYNTHETIC_DATASET
