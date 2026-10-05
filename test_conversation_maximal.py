"""
test_conversation_maximal.py — Pengujian Komprehensif Deteksi Percakapan Tingkat Maksimal
Menguji:
  1. Dekomposisi segmen akustik Whisper multi-kalimat menjadi giliran bicara individu
  2. Identifikasi pembicara Dokter vs Pasien tingkat tinggi (state machine & aksi klinis)
  3. Isolasi tuturan pasien & penapisan gejala yang dinegasikan (negation detection)
  4. Ekstraksi entitas klinis (TTV / Vital Signs, Riwayat Alergi, Resep)
"""
from medical_extractor import MedicalComplaintExtractor

e = MedicalComplaintExtractor()
SEP = "=" * 75

test_dialogues = [
    {
        "id": "MAX-1",
        "title": "Konsultasi Lengkap RS: Multi-turn, Negasi Sesak Napas, TTV & Alergi Disangkal",
        "segments": [
            {
                "start": 0.0, "end": 5.2,
                "text": "Selamat pagi dok. Pagi bu, ada keluhan apa? Badan saya panas dari kemarin malam dok, terus pilek."
            },
            {
                "start": 5.2, "end": 9.8,
                "text": "Ada batuk atau sesak napas bu? Tidak ada batuk maupun sesak napas dok."
            },
            {
                "start": 9.8, "end": 17.5,
                "text": "Baik, coba saya periksa dulu ya bu. Tensinya 120/80 mmHg, suhunya 38.5 derajat celcius, saturasi 98%."
            },
            {
                "start": 17.5, "end": 22.0,
                "text": "Ada alergi obat bu? Tidak ada alergi dok."
            },
            {
                "start": 22.0, "end": 28.5,
                "text": "Nanti saya resepkan paracetamol ya bu, banyak istirahat dan banyak minum air hangat."
            }
        ]
    },
    {
        "id": "MAX-2",
        "title": "Percakapan IGD: Nyeri Dada & Mual (Muntah Dinegasikan, TTV Tensi 140/90)",
        "segments": [
            {
                "start": 0.0, "end": 4.5,
                "text": "Dok, dada saya sakit nyeri sejak tadi pagi. Ada mual atau muntah pak? Mual ada dok tapi tidak muntah."
            },
            {
                "start": 4.5, "end": 9.0,
                "text": "Coba saya cek tensinya dulu ya pak. Tensinya 140/90 mmHg, agak tinggi ya pak."
            }
        ]
    },
    {
        "id": "MAX-3",
        "title": "Poli Anak (Ibu membawa anak): Demam Menggigil, Tanpa Batuk",
        "segments": [
            {
                "start": 0.0, "end": 6.0,
                "text": "Selamat pagi dokter. Pagi bu, sakit apa anaknya? Anak saya badannya panas tinggi menggigil dari kemarin lusa dok. Nggak ada batuk sama sekali."
            }
        ]
    }
]

print(SEP)
print("PENGUJIAN TINGKAT MAKSIMAL: DETEKSI PERCAKAPAN MEDIS RS")
print(SEP)

all_passed = True

for td in test_dialogues:
    full_transcript = " ".join([s["text"] for s in td["segments"]])
    res = e.extract(full_transcript, td["segments"])

    print(f"\n[{td['id']}] {td['title']}")
    print("-" * 75)
    print("KELUHAN UTAMA    :", res["keluhan_utama"] or "(KOSONG)")
    print("KELUHAN TAMBAHAN :", res["keluhan_tambahan"] or "(KOSONG)")
    print("LAMA SAKIT       :", res["lama_sakit"] or "(KOSONG)")
    print("CONFIDENCE       :", f"{res['confidence']['level']} ({res['confidence']['score']}/100)")
    print(f"GILIRAN BICARA TERDETEKSI ({len(res['labeled_segments'])} turns):")
    for s in res["labeled_segments"]:
        ts = f"{s['start']}s–{s['end']}s" if 'start' in s else ""
        print(f"  [{s['speaker']}] {ts:<14} : {s['text']}")

    # Verifikasi Kasus MAX-1
    if td["id"] == "MAX-1":
        # 1. Sesak napas & batuk harus TIDAK masuk keluhan utama maupun keluhan tambahan (karena dinegasikan)
        has_sesak = "sesak" in res["keluhan_utama"].lower() or "sesak" in res["keluhan_tambahan"].lower()
        has_batuk = "batuk" in res["keluhan_utama"].lower() or "batuk" in res["keluhan_tambahan"].lower()
        if has_sesak or has_batuk:
            print("  [FAIL] Gejala yang dinegasikan (sesak/batuk) keliru diekstrak!")
            all_passed = False
        else:
            print("  [PASS] Gejala yang dinegasikan (sesak/batuk) berhasil disaring!")

        # 2. TTV harus terdeteksi
        vitals = res["clinical_entities"]["vitals"]
        if "120/80 mmHg" in vitals.get("tensi", "") and "38.5°C" in vitals.get("suhu", ""):
            print("  [PASS] TTV berhasil diekstrak (Tensi 120/80, Suhu 38.5°C)!")
        else:
            print("  [FAIL] TTV gagal diekstrak:", vitals)
            all_passed = False

        # 3. Alergi disangkal harus terdeteksi
        allergies = res["clinical_entities"]["allergies"]
        if allergies.get("disangkal"):
            print("  [PASS] Status alergi berhasil diekstrak (Alergi: Disangkal)!")
        else:
            print("  [FAIL] Status alergi gagal diekstrak:", allergies)
            all_passed = False

    # Verifikasi Kasus MAX-2
    elif td["id"] == "MAX-2":
        # Muntah dinegasikan (tidak muntah), Mual ada
        has_muntah = "muntah" in res["keluhan_utama"].lower() or "muntah" in res["keluhan_tambahan"].lower()
        if has_muntah:
            print("  [FAIL] Muntah yang dinegasikan keliru diekstrak!")
            all_passed = False
        else:
            print("  [PASS] Muntah yang dinegasikan berhasil disaring!")

print("\n" + SEP)
print(f"HASIL AKHIR: {'SEMUA PENGUJIAN MAKSIMAL BERHASIL!' if all_passed else 'ADA PENGUJIAN GAGAL!'}")
print(SEP)
