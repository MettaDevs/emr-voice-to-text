"""
test_hospital_cases.py — Verifikasi lengkap alur klinis Rumah Sakit di Indonesia
"""
from medical_extractor import MedicalComplaintExtractor

e = MedicalComplaintExtractor()
SEP = "=" * 70

cases = [
    {
        "id": "RS-1",
        "title": "Konsultasi Poli Umum: Demam, Pilek, Batuk Berdahak",
        "text": "Selamat pagi dok. Pagi bu, ada keluhan apa? Badan saya panas dari kemarin malam dok, terus pilek sama batuk berdahak grok-grok. Tenggorokan juga sakit buat menelan. Sudah dua hari yang lalu mulai terasa.",
    },
    {
        "id": "RS-2",
        "title": "Pasien IGD / Penyakit Dalam: Maag Kambuh, Nyeri Ulu Hati & Mual",
        "text": "Dok, ulu hati perih banget kayaknya maag kambuh dari semalam, terus mual mau muntah dan perut kembung begah.",
    },
    {
        "id": "RS-3",
        "title": "Pasien Poli Saraf: Vertigo & Pusing Kliyengan Berputar",
        "text": "Keluhannya apa Pak? Kepala saya puyeng kliyengan muter-muter dok kalau bangun tidur, sejak 3 hari yang lalu. Badan lemes sekali.",
    },
    {
        "id": "RS-4",
        "title": "Rekaman Riil User: Mang Kimun Sakit Panas & Pilek (Word Number Duration)",
        "text": "Mang Kimun sakit panas dan pilek dari dua hari yang lalu.",
    },
    {
        "id": "RS-5",
        "title": "Rekaman Riil User: Pilek sama pileksnya, tanas, kau belah tahu",
        "text": "Malam Dok, kemarin saya pusing. Hari ini panas juga sama pilek. Kalau udah tau dari kapan ya pilek sama pileksnya? Dari 2 hari lalu sih dok, tapi ini sekarang sudah mendingan.",
    },
    {
        "id": "RS-6",
        "title": "Kasus Diare / Gastroenteritis di Rumah Sakit",
        "text": "Ada keluhan apa? Perut melilit mulas dok, terus buang-buang air mencret BAB cair sudah 5 kali dari tadi pagi.",
    },
    {
        "id": "RS-7",
        "title": "Kasus Anyang-anyangan / Infeksi Saluran Kemih",
        "text": "Keluhannya apa Bu? Saya anyang-anyangan dok dari kemarin, kencing perih dan panas rasanya.",
    },
    {
        "id": "RS-8",
        "title": "Kasus Prosedur RS: Rontgen Dada & Cek Lab",
        "text": "Dok, kemarin saya sudah rontgen dada dan cek darah lengkap di laboratorium rumah sakit, hasilnya sudah keluar.",
    },
    {
        "id": "RS-9",
        "title": "Mic Testing di RS: Tes 1 2 3 (Harus Bersih Kosong)",
        "text": "Tes, tes 1 2 3 testing mic satu dua tiga.",
    }
]

for c in cases:
    print(SEP)
    print(f"[{c['id']}] {c['title']}")
    res = e.extract(c["text"])
    print("TRANSKRIP BERSIH :", res["cleaned_transcript"])
    print("KELUHAN UTAMA    :", res["keluhan_utama"] or "(KOSONG)")
    print("KELUHAN TAMBAHAN :", res["keluhan_tambahan"] or "(KOSONG)")
    print("LAMA SAKIT       :", res["lama_sakit"] or "(KOSONG)")
    print("CONFIDENCE       :", f"{res['confidence']['level']} ({res['confidence']['score']}/100)")
    if res["confidence"]["warnings"]:
        for w in res["confidence"]["warnings"]:
            print("  [PERINGATAN]   :", w)
    if res["labeled_segments"]:
        print("SPEAKER TURNS    :")
        for s in res["labeled_segments"]:
            print(f"  [{s['speaker']}] {s['text']}")

print(SEP)
