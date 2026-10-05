import re

text = "Malam, malam. Apa keluhan ya bu? Kalau boleh tahu kemarin saya pilek, hari ini udah mendingan sih. Udah berapa lama pilek ya bu ? Dari kemarim, dari kemaren malum, sekarang udah mendengan. Cuman ini setiap malam kayak nggak enak badannya tuh. Panas badannya."

# 1. Deteksi Durasi / Lama Sakit
durasi_match = re.search(r'\b((?:dari|sejak|sudah)\s+(?:kemarin(?:\s+(?:malam|pagi|siang|sore))?|tadi|\d+\s+hari|\d+\s+minggu))\b', text, re.I)
lama_sakit = durasi_match.group(1).title() if durasi_match else ""

# 2. Deteksi Keluhan Utama:
# Cari respon terhadap pertanyaan "apa keluhan" atau deteksi kata gejala primer
keluhan_utama = ""
q_keluhan = re.search(r'(?:apa\s+keluhan(?:nya)?|keluhan(?:nya)?\s+apa)[^?]*\?\s*([^?]+)', text, re.I)
if q_keluhan:
    raw_answer = q_keluhan.group(1).strip()
    # Bersihkan sapaan dan ambil klausa gejala
    symptom_match = re.search(r'\b(pilek|batuk|flu|demam|pusing|sakit\s+kepala|sesak|nyeri|mual|rontgen)\b', raw_answer, re.I)
    if symptom_match:
        keluhan_utama = symptom_match.group(1).title()
    else:
        keluhan_utama = raw_answer

# 3. Deteksi Keluhan Tambahan:
# Ambil klausa gejala penyerta (cuman, terus, selain itu, badan...)
keluhan_tambahan = []
additional_match = re.findall(r'(?:cuman|terus|selain\s+itu|sama)\s+([^.]+)', text, re.I)
if additional_match:
    for m in additional_match:
        # Bersihkan kata filler
        c = re.sub(r'\b(?:ini|tuh|kayak)\b', '', m, flags=re.I).strip()
        keluhan_tambahan.append(c.capitalize())

# Cek apakah ada gejala panas/anget/meriang di kalimat akhir
if re.search(r'\b(panas\s+badannya|anget|meriang|nggak\s+enak\s+badan)\b', text, re.I):
    extra = re.findall(r'([^\.\?]*\b(?:panas|anget|meriang|nggak\s+enak)\s+badan[^\.\?]*)', text, re.I)
    for e in extra:
        cln = e.strip().capitalize()
        if cln not in keluhan_tambahan:
            keluhan_tambahan.append(cln)

print("KELUHAN UTAMA :", keluhan_utama)
print("TAMBAHAN      :", ", ".join(keluhan_tambahan))
print("LAMA SAKIT    :", lama_sakit)
