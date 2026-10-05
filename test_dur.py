import re

def get_duration(text):
    # Prioritaskan durasi yang lebih spesifik (kemarin malam / X hari / X minggu) daripada sekadar "tadi"
    specific_patterns = [
        r'\b((?:dari|sejak|sudah)\s+kemar[ei]n\s*(?:mal[ei]m|pagi|siang|sore)?)\b',
        r'\b((?:dari|sejak|sudah|for|since)\s+\d+\s+(?:hari|minggu|bulan|days|weeks))\b',
        r'\b((?:dari|sejak)\s+(?:tadi\s*(?:mal[ei]m|pagi|siang)?|semalam))\b'
    ]
    for pat in specific_patterns:
        m = re.search(pat, text, re.I)
        if m:
            dur = m.group(1).title()
            # Standarisasi malem / malum -> Malam
            dur = re.sub(r'mal[eu]m', 'Malam', dur, flags=re.I)
            dur = re.sub(r'kemar[ei]n', 'Kemarin', dur, flags=re.I)
            return dur
    return "-"

text = "Dari tadi, dari kemaren malem. Cuman ini..."
print("Durasi:", get_duration(text))
