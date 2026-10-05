import re

def parse_clinical_dialogue(raw_transcript: str):
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

    # 1. Ekstraksi Durasi / Lama Sakit (ID & EN)
    dur_match = re.search(r'\b((?:dari|sejak|sudah|for|since|past)\s+(?:kemarin(?:\s*(?:malam|pagi|siang|sore))?|yesterday(?:\s*night)?|tadi|\d+\s+hari|\d+\s+minggu|\d+\s+days|\d+\s+weeks))\b', normalized, re.I)
    lama_sakit = dur_match.group(1).title() if dur_match else ""

    # 2. Ekstraksi Keluhan Utama
    primary_symptoms = [
        ("pilek", "Pilek"),
        ("batuk kering", "Batuk kering"),
        ("batuk berdahak", "Batuk berdahak"),
        ("batuk", "Batuk"),
        ("demam tinggi", "Demam tinggi"),
        ("demam", "Demam"),
        ("pusing berputar", "Pusing berputar / Vertigo"),
        ("pusing", "Pusing"),
        ("sakit kepala", "Sakit kepala"),
        ("kepala saya sakit", "Sakit kepala"),
        ("migrain", "Migrain"),
        ("vertigo", "Vertigo"),
        ("sesak napas", "Sesak napas"),
        ("sesak", "Sesak napas"),
        ("nyeri dada", "Nyeri dada"),
        ("chest pain", "Chest pain"),
        ("headache", "Headache"),
        ("fever", "Fever"),
        ("cough", "Cough"),
        ("sakit perut", "Sakit perut"),
        ("perut saya melilit", "Perut melilit"),
        ("perut melilit", "Perut melilit"),
        ("mual muntah", "Mual muntah"),
        ("mual", "Mual"),
        ("sakit tenggorokan", "Sakit tenggorokan"),
        ("rontgen dada", "Pemeriksaan rontgen dada"),
        ("rontgen", "Pemeriksaan rontgen"),
        ("asam lambung", "Asam lambung / GERD"),
        ("gerd", "GERD"),
    ]

    keluhan_utama = ""
    # Cari di segmen respon pertama pasien
    first_part = normalized.split('?')[1] if '?' in normalized else normalized
    for trigger, label in primary_symptoms:
        if re.search(r'\b' + re.escape(trigger) + r'\b', first_part, re.I):
            keluhan_utama = label
            break

    if not keluhan_utama:
        # Cari di seluruh teks jika belum ketemu
        for trigger, label in primary_symptoms:
            if re.search(r'\b' + re.escape(trigger) + r'\b', normalized, re.I):
                keluhan_utama = label
                break

    # 3. Ekstraksi Keluhan Tambahan / Anamnesa
    secondary_patterns = [
        (r'\b(?:di\s+atas\s+)?pipi\s+(?:saya\s+)?sakit\b', 'Di atas pipi sakit'),
        (r'\b(?:nggak|tidak)\s+enak\s+badan(?:nya)?\b', 'Badan tidak enak'),
        (r'\b(?:panas|anget)\s+badan(?:nya)?\b', 'Badan panas / anget'),
        (r'\bmeriang\b', 'Meriang'),
        (r'\bmual\b', 'Mual'),
        (r'\blemas\b', 'Badan lemas'),
        (r'\bgatal\b', 'Tenggorokan gatal'),
        (r'\bmenggigil\b', 'Menggigil'),
        (r'\bsusah\s+tidur\b', 'Susah tidur'),
        (r'\bdifficulty\s+breathing\b', 'Difficulty breathing'),
        (r'\bdizzy\b', 'Dizziness / Pusing'),
    ]

    keluhan_tambahan_items = []
    for pat, label in secondary_patterns:
        if re.search(pat, normalized, re.I):
            if label.lower() not in keluhan_utama.lower() and label not in keluhan_tambahan_items:
                keluhan_tambahan_items.append(label)

    # Deteksi klausa tambahan dari kata sambung (cuman, terus, selain itu, sama)
    connector_matches = re.findall(r'(?:cuman|terus|selain\s+itu|sama|along\s+with|and\s+also)\s+([^.,?]+)', normalized, re.I)
    for m in connector_matches:
        c = re.sub(r'\b(?:ini|tuh|kayak|sih)\b', '', m, flags=re.I).strip()
        if len(c) > 4 and not re.search(r'^\d+$', c):
            formatted_c = c.capitalize()
            if not any(item.lower() in formatted_c.lower() for item in keluhan_tambahan_items):
                if keluhan_utama.lower() not in formatted_c.lower():
                    keluhan_tambahan_items.append(formatted_c)

    keluhan_tambahan = ", ".join(keluhan_tambahan_items) if keluhan_tambahan_items else "-"

    return {
        "keluhan_utama": keluhan_utama or "Pemeriksaan Umum",
        "keluhan_tambahan": keluhan_tambahan,
        "lama_sakit": lama_sakit or "-",
        "normalized_text": normalized
    }

# Uji semua kasus:
test_cases = [
    ("Kasus Rekaman User (Terbaru)", "Malam, malam. Apa keluhan ya bu? Kalau boleh tahu kemarin saya pilot, hari ini ada mendingan sih. Udah berapa lama pilot ya bu? Dari kemarin malum. Cuman ini setiap malam kayak nggak enak badannya tuh nganes badannya."),
    ("Kasus 1 Atasan", "Dokter: Keluhannya apa? Pasien: eee kepala saya sakit, terus di atas pipi saya sakit"),
    ("Kasus 2 Perut", "Ada keluhan apa bu hari ini? Anu dok, perut saya melilit dari kemarin malam, mmm terus badan agak meriang dan mual"),
    ("Kasus 3 Batuk", "Bisa diceritakan keluhan utamanya apa? Eee ini dok, batuk kering sudah tiga hari, terus tenggorokan rasanya gatal dan agak sesak"),
    ("Kasus Rontgen", "Dok, kemarin saya sudah rontgen dada di rumah sakit")
]

for title, text in test_cases:
    print("\n" + "=" * 60)
    print(f"TEST: {title}")
    print("=" * 60)
    r = parse_clinical_dialogue(text)
    print("KELUHAN UTAMA :", r["keluhan_utama"])
    print("TAMBAHAN      :", r["keluhan_tambahan"])
    print("LAMA SAKIT    :", r["lama_sakit"])
