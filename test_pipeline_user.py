import re

def parse_clinical_dialogue(raw_transcript: str):
    # 1. Normalisasi fonetik & typo audio umum
    normalized = raw_transcript
    fixes = {
        r'\bpilot\b': 'pilek',
        r'\bkemarim\b': 'kemarin',
        r'\bkemaren\b': 'kemarin',
        r'\bmalum\b': 'malam',
        r'\bmendengan\b': 'mendingan',
        r'\bnganes\b': 'anget',
        r'\bronggen|rongsen|monsan\b': 'rontgen',
        r'\bbatu\s+kering\b': 'batuk kering',
        r'\btemurokan|tengurokan\b': 'tenggorokan',
    }
    for pat, rep in fixes.items():
        normalized = re.sub(pat, rep, normalized, flags=re.I)

    # 2. Ekstraksi Durasi / Lama Sakit
    dur_match = re.search(r'\b((?:dari|sejak|sudah)\s+(?:kemarin(?:\s*(?:malam|pagi|siang|sore))?|tadi|\d+\s+hari|\d+\s+minggu|\d+\s+bulan))\b', normalized, re.I)
    lama_sakit = dur_match.group(1).title() if dur_match else ""

    # 3. Ekstraksi Keluhan Utama
    # Cari gejala primer yang pertama kali diungkapkan pasien saat ditanya keluhan
    keluhan_utama = ""
    primary_symptoms = [
        "pilek", "batuk kering", "batuk berdahak", "batuk", "demam tinggi", "demam",
        "pusing berputar", "pusing", "sakit kepala", "migrain", "vertigo", "sesak napas",
        "sesak", "nyeri dada", "sakit perut", "perut melilit", "mual muntah", "mual",
        "sakit tenggorokan", "rontgen", "asam lambung", "gerd", "diare"
    ]

    # Cari di segmen awal pasien
    first_part = normalized.split('?')[1] if '?' in normalized else normalized
    for s in primary_symptoms:
        if re.search(r'\b' + s + r'\b', first_part, re.I):
            keluhan_utama = s.title()
            break

    if not keluhan_utama:
        # Fallback jika tidak ada kata gejala khusus, ambil klausa pertama
        keluhan_utama = "Pemeriksaan Medis"

    # 4. Ekstraksi Keluhan Tambahan / Anamnesa
    keluhan_tambahan_items = []

    # Deteksi klausa gejala sekunder (nggak enak badan, panas, meriang, lemas, dll)
    secondary_patterns = [
        (r'\b(?:nggak|tidak)\s+enak\s+badan(?:nya)?\b', 'Badan tidak enak'),
        (r'\b(?:panas|anget)\s+badan(?:nya)?\b', 'Badan panas / hangat'),
        (r'\bmeriang\b', 'Meriang'),
        (r'\bmual\b', 'Mual'),
        (r'\blemas\b', 'Badan lemas'),
        (r'\bgatal\b', 'Gatal'),
        (r'\bmenggigil\b', 'Menggigil'),
        (r'\bsusah\s+tidur\b', 'Susah tidur'),
        (r'\bpipi\s+sakit\b', 'Di atas pipi sakit'),
    ]

    for pat, label in secondary_patterns:
        if re.search(pat, normalized, re.I):
            # Pastikan tidak dobel dengan keluhan utama
            if label.lower() not in keluhan_utama.lower() and label not in keluhan_tambahan_items:
                keluhan_tambahan_items.append(label)

    # Deteksi klausa tambahan dari kata sambung (cuman, terus, selain itu)
    connector_matches = re.findall(r'(?:cuman|terus|selain\s+itu|sama)\s+([^.,?]+)', normalized, re.I)
    for m in connector_matches:
        c = re.sub(r'\b(?:ini|tuh|kayak|sih)\b', '', m, flags=re.I).strip()
        # Jika bukan sekadar angka atau sapaan dan panjangnya cukup
        if len(c) > 4 and not re.search(r'^\d+$', c):
            formatted_c = c.capitalize()
            # Hindari duplikasi
            if not any(item.lower() in formatted_c.lower() for item in keluhan_tambahan_items):
                keluhan_tambahan_items.append(formatted_c)

    keluhan_tambahan = ", ".join(keluhan_tambahan_items) if keluhan_tambahan_items else "-"

    return {
        "keluhan_utama": keluhan_utama,
        "keluhan_tambahan": keluhan_tambahan,
        "lama_sakit": lama_sakit,
        "normalized_text": normalized
    }

raw_example = "Malam, malam. Apa keluhan ya bu? Kalau boleh tahu kemarin saya pilek, hari ini udah mendingan sih. Udah berapa lama pilek ya bu ? Dari kemarim, dari kemaren malum, sekarang udah mendengan. Cuman ini setiap malam kayak nggak enak badannya tuh. Panas badannya."
res = parse_clinical_dialogue(raw_example)

print("HASIL PARSING KLINIS MODERN:")
print("[KELUHAN UTAMA]             :", res["keluhan_utama"])
print("[KELUHAN TAMBAHAN/ANAMNESA] :", res["keluhan_tambahan"])
print("[LAMA SAKIT/DURASI]         :", res["lama_sakit"])
