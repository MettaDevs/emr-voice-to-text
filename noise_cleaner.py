import re

# Daftar filler words / kata jeda bilingual
FILLER_PATTERNS = [
    r'\b(?:e{1,6}|eh|ee|eee|eeee|eeh|ehm)\b',
    r'\b(?:anu|apa\s*tuh|apa\s*ya)\b',
    r'\b(?:h{1,4}m{1,4}|e{1,4}m{1,4}|m{2,6})\b',
    r'\b(?:aduh|wah|waduh)\b',
    r'\b(?:kayak|kayaknya|gitu|gitu\s*deh)\b',
    r'\b(?:sih|deh|kan|loh)\b',
    r'\b(?:uh|uhm|um|umm|er|ah|like|you\s*know)\b',
]

# Kamus Normalisasi Fonetik Medis (Koreksi salah dengar fonem medis)
MEDICAL_PHONETIC_CORRECTIONS = {
    r'\b(?:ronggen|rongsen|ronsen|monsan|on\s+sunday)\b': 'rontgen',
    r'\b(?:batu\s+kering)\b': 'batuk kering',
    r'\b(?:temurokan|tengurokan)\b': 'tenggorokan',
    r'\b(?:siti\s+sken|city\s+scan)\b': 'CT Scan',
    r'\b(?:u\s+es\s+ge)\b': 'USG',
    r'\b(?:e\s+ka\s+ge)\b': 'EKG',
    r'\b(?:pe\s+ce\s+er)\b': 'PCR',
}

def clean_verbal_noise(text: str) -> str:
    """
    1. Membersihkan filler words (eee, anu, um, uh)
    2. Menormalkan salah dengar fonetik istilah medis (ronggen/monsan -> rontgen)
    3. Merapikan tanda baca dan kapitalisasi
    """
    if not text:
        return ""

    cleaned = text

    # 1. Normalisasi fonetik istilah medis terlebih dahulu
    for pattern, replacement in MEDICAL_PHONETIC_CORRECTIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # 2. Hapus filler words
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # 3. Rapikan tanda baca
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    cleaned = re.sub(r'\?\s*,\s*', '? ', cleaned)
    cleaned = re.sub(r'^[,\s\.\-]+', '', cleaned)
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    cleaned = re.sub(r'\s*\.\s*', '. ', cleaned)
    cleaned = re.sub(r',\s*,+', ', ', cleaned)
    cleaned = re.sub(r'\.\s*\.+', '.', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


if __name__ == "__main__":
    test_str = "I go to hospital, I am a monsan 321 dan ronggen dada"
    print("Sebelum :", test_str)
    print("Sesudah :", clean_verbal_noise(test_str))
