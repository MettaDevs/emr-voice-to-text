"""
Test pipeline 5 tahap:
  Audio → STT → L1 Fonetik → L2 Bahasa Indonesia (Sehari-hari/Baku) → L3 Konteks Kalimat → L4 Konteks Medis
"""
from medical_extractor import (
    MedicalComplaintExtractor, _apply_layer1, _apply_layer2, _label_speakers
)

e = MedicalComplaintExtractor()
SEP = "=" * 65

cases = [
    {
        "desc": "Kasus Riil dari Rekaman User (tanas, kau belah tahu, malu, mendingan)",
        "text": "Malam, bang. Malam. Kau belah tahu kenapa keluhannya ya bu? Ini tuh saya tanas sama pikir juga. Tanasnya sih kemarin malu enggak dong ya. Sekarang udah mendingan sebenarnya coba gue cari obat aja. Oke bu, coba saya cek dulu tanasnya ya bu...",
        "segs": []
    },
    {
        "desc": "Bahasa Sehari-hari: Greges-greges, Pusing Kliyengan, Enek",
        "text": "Ada keluhan apa Pak? Dok, badan saya greges-greges dari kemarin malam, pusing kliyengan sama enek perutnya.",
        "segs": []
    },
    {
        "desc": "Bahasa Sehari-hari: Anyang-anyangan, Kencing Perih",
        "text": "Keluhannya apa Bu? Saya anyang-anyangan dok dari tadi pagi, kencing perih banget.",
        "segs": []
    },
    {
        "desc": "Bahasa Sehari-hari: Mencret / BAB Cair, Mulas",
        "text": "Sudah berapa lama sakitnya? Udah 2 hari dok. Perut melilit, mulas, terus mencret BAB cair terus.",
        "segs": []
    },
    {
        "desc": "Bahasa Sehari-hari: Maag / Ulu Hati Perih",
        "text": "Ada keluhan apa? Dok, ulu hati perih banget kayaknya maag kambuh dari semalam, terus mual.",
        "segs": []
    },
    {
        "desc": "Bahasa Sehari-hari: Batuk Grok-grok, Nelen Sakit, Suara Serak",
        "text": "Keluhannya apa? Batuk berdahak grok-grok dok sudah 3 hari, nelen sakit sama suara serak.",
        "segs": []
    },
    {
        "desc": "Audio Non-Medis: Tes Mikrofon (Harus Kosong)",
        "text": "tes 1 2 3 tes satu dua tiga halo",
        "segs": []
    },
]

for c in cases:
    print(SEP)
    print("KASUS :", c["desc"])
    res = e.extract(c["text"], c["segs"])
    print("RAW   :", res["raw_transcript"][:75] + ("..." if len(res["raw_transcript"]) > 75 else ""))
    print("L1    :", res["layer1_phonetic"][:75] + ("..." if len(res["layer1_phonetic"]) > 75 else ""))
    print("L2    :", res["cleaned_transcript"][:75] + ("..." if len(res["cleaned_transcript"]) > 75 else ""))
    print("--- HASIL FORMULIR EMR ---")
    print("KELUHAN UTAMA    :", res["keluhan_utama"] or "(KOSONG)")
    print("KELUHAN TAMBAHAN :", res["keluhan_tambahan"] or "(KOSONG)")
    print("LAMA SAKIT       :", res["lama_sakit"] or "(KOSONG)")
    print("CONFIDENCE       :", f"{res['confidence']['level']} ({res['confidence']['score']}/100)")
    if res["confidence"]["warnings"]:
        for w in res["confidence"]["warnings"]:
            print("  [PERINGATAN]   :", w)
    if res["labeled_segments"]:
        print("GILIRAN BICARA (SPEAKER):")
        for s in res["labeled_segments"]:
            print(f"  [{s['speaker']}] {s['text']}")

print(SEP)
