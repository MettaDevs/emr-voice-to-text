"""
medical_extractor.py  — MedVoice AI Clinical Extraction Pipeline
=================================================================

Arsitektur Pipeline 5 Tahap:
  Audio  →  STT  →  Bahasa Indonesia  →  Konteks Kalimat  →  Konteks Medis
               ↑           ↑                     ↑                   ↑
          (transcriber) (Layer 1 & 2)        (Layer 3)           (Layer 4)

Layer 1 – Koreksi Fonetik     : Kata-kata yang sering disalah-dengar oleh Whisper pada audio mic
Layer 2 – Bahasa Indonesia    : Slang/gaul → baku; bahasa sehari-hari/daerah → istilah medis
Layer 3 – Konteks Kalimat     : Segmentasi per kalimat & giliran bicara Dokter vs Pasien
Layer 4 – Konteks Medis       : Ekstraksi keluhan utama, keluhan tambahan, durasi, scoring & warning
"""

import re
import datetime
import uuid

# ══════════════════════════════════════════════════════════════════════════════
# PROTECTED CLINICAL ENTITIES (Anti-Coruption Safeguards)
# Entitas yang DILARANG diubah/dikoreksi sembarangan untuk mencegah malapraktik AI
# ══════════════════════════════════════════════════════════════════════════════
PROTECTED_DRUG_NAMES = {
    "paracetamol", "amoxicillin", "amoksisilin", "ibuprofen", "asam mefenamat",
    "omeprazole", "lansoprazole", "cetirizine", "setirisin", "ambroxol",
    "domperidone", "salbutamol", "ciprofloxacin", "cefixime", "cefadroxil",
    "antasida", "metformin", "amlodipine", "captopril", "simvastatin", "azithromycin",
    "dexamethasone", "deksametason", "prednisone", "oralit", "diazepam"
}

PROTECTED_DIAGNOSES = {
    "hipertensi", "diabetes", "gastritis", "pneumonia", "tb paru", "vertigo",
    "migrain", "diare", "gerd", "faringitis", "bronkitis", "asma", "covid", "dispepsia"
}

DOSAGE_PATTERN = re.compile(r'\b(?:\d+(?:[.,]\d+)?\s*(?:mg|gr|gram|ml|cc|tablet|kapsul|tetes|sendok|x\s*\d+|\/\s*hari))\b', re.I)
VITAL_NUM_PATTERN = re.compile(r'\b(?:\d{2,3}\s*\/\s*\d{2,3}|\d{2}(?:[.,]\d+)?\s*(?:°\s*c|c\b|derajat)|\d{2,3}\s*%(?:\s*spo2)?|\d{2,3}\s*(?:x\/menit|bpm))\b', re.I)

PHONETIC_CORRECTIONS = {
    # ── Temuan Kritis dari Rekaman Nyata Pasien & Dokter (Longest Match First) ──
    r'\btanasnya\b':                        'panasnya',
    r'\btanas\b':                           'panas',
    r'\bnganes\b':                          'panas',
    r'\bkau\s+belah\s+tah?u\b':             'kalau boleh tahu',
    r'\bkalo\s+bol(?:eh)?\s+tah?u\b':       'kalau boleh tahu',
    r'\bkenapa\s+keluarnya\b':              'kenapa keluhannya',
    r'\bapa\s+keluarnya\b':                 'apa keluhannya',
    r'\bada\s+keluarnya\b':                 'ada keluhannya',
    r'\bkeluarnya\s+apa\b':                 'keluhannya apa',
    r'\bpilek\s+sama\s+pilek(?:nya|snya)?\b': 'panas sama pileknya',
    r'\bkemarin\s+malu\b':                  'kemarin malam',
    r'\bsetiap\s+malu\b':                   'setiap malam',
    r'\bmalang\b(?=\s+(?:hari|ini|kemarin))': 'malam',
    r'\bmalum\b|\bmalem\b':                 'malam',
    r'\bkemarim\b|\bkemaren\b|\bkemal[ei]n\b|\bkemaring\b': 'kemarin',
    r'\balergia\b':                         'alergi',
    r'\bkalau\s+boleh\s+tau\b|\bkalo\s+boleh\s+tau\b': 'kalau boleh tahu',
    r'\bmendengan\b':                       'mendingan',

    # ── Fonetik Akustik Standar & Tes Mic ──
    r'\bters,\s*ters\b':                    'tes, tes',
    r'\bters[-\s]+ters\b':                  'tes tes',
    r'\bters\b':                            'tes',
    r'\btest\b':                            'tes',
    r'\bdesu\b|\bdess\b':                   'tes',
    r'\bpidak\b|\bpikil\b':                 'pilek',
    r'\bronggen\b|\brongsen\b|\bronsen\b|\broentgen\b|\broncen\b|\bmonsan\b': 'rontgen',
    r'\bfoto\s+torak[s]?\b|\bfoto\s+thorax\b': 'rontgen dada',
    r'\bbatu\s+kering\b':                   'batuk kering',
    r'\bbatu\s+berdahak\b':                 'batuk berdahak',
    r'\bbatu\s+darah\b':                   'batuk darah',
    r'\bbatu\s+pilek\b':                    'batuk pilek',
    r'\bbatu[-\s]batu\b':                   'batuk-batuk',
    r'\bbatu\s+terus\b':                    'batuk terus',
    r'\btemurokan\b|\btengurokan\b':        'tenggorokan',
    r'\bpinggang\s+saya\s+sakit\b':         'sakit pinggang',
    r'\bdog\b|\bdock\b|\bduk\b':            'dok',
    r'\btengsi(?:nya)?\b|\btenshi\b|\btenci\b|\btengsin\b': 'tensi',
    r'\bsaturashi\b':                       'saturasi',
    r'\bi\s*ge\s*de\b|\bi\.g\.d\b|\bu\s*ge\s*de\b': 'IGD',
    r'\bi\s*se\s*u\b|\bi\.c\.u\b':          'ICU',
    r'\bbepejes\b|\bbe\s*pe\s*je\s*es\b|\bb\.p\.j\.s\b': 'BPJS',
    r'\be\s*ka\s*ge\b|\be\.k\.g\b':          'EKG',
    r'\bu\s*es\s*ge\b|\bu\.s\.g\b':          'USG',
    r'\blaborat\b|\blap\s+darah\b':         'laboratorium',
    r'\bde\s*el\b|\bd\.l\b':                'darah lengkap',
    r'\bparasetamol\b|\bparastamol\b|\bparasitamol\b|\bparaset\b|\bparacet\b': 'paracetamol',
    r'\bamoksilin\b|\bamoksisilin\b|\bamoxilin\b|\bamox\b': 'amoxicillin',
    r'\bsefiksim\b|\bcefiksim\b':           'cefixime',
    r'\bsefadroksil\b|\bcefadrosil\b':      'cefadroxil',
    r'\bsiprofloksasin\b|\bsipro\b':        'ciprofloxacin',
    r'\bantacid\b|\banta\s+sida\b|\bantasid\b': 'antasida',
    r'\bomeprazol\b':                       'omeprazole',
    r'\blansoprazol\b':                     'lansoprazole',
    r'\bsetirisin\b|\bsetirizin\b|\bcetirisin\b': 'cetirizine',
    r'\bambroksol\b|\bambroxsol\b':         'ambroxol',
    r'\bdomperidon\b':                      'domperidone',
    r'\bibu\s+profen\b':                    'ibuprofen',
    r'\bmefenamat\b|\basam\s+penamat\b':    'asam mefenamat',
    r'\bsalbuta\b':                         'salbutamol',
    r'\bnebulaizer\b|\bnebulezer\b|\bnebuleser\b': 'nebulizer',
    r'\bdi\s+uap\b':                        'diuap',
    r'\bopnam\b|\bop\s+namen\b|\bofname\b': 'opname',
    r'\bimpus\b':                           'infus',
    r'\bsesek\s+nafas\b|\bsesak\s+nafas\b|\bnafas\s+sesek\b': 'sesak napas',
    r'\bsesek\b':                           'sesak',
    r'\bkliengan\b|\bkleyengan\b|\bkeliyengan\b': 'kliyengan',
    r'\bmriang\b|\bmeriyang\b':             'meriang',
    r'\bayang[-\s]ayangan\b|\bkencing\s+anyang\b': 'anyang-anyangan',
    r'\bb\.a\.b\b|\bb\s+a\s+b\b':           'BAB',
    r'\bb\.a\.k\b|\bb\s+a\s+k\b':           'BAK',

    # ── Gumaman, Nafas & Hesitation (Dihilangkan total, tidak masuk teks dan tidak diganti kata lain) ──
    r'(?<!\bpak\s)(?<!\bbapak\s)\b(?:amin|aamiin|aamin)\b': '',
    r'\b(?:h+m+|e+h+|e{2,}|m{2,}|u+h+|u+m+|a+h+)\b': '',
    r'\[.*?\]|\(.*?\)|(?:\b(?:hembusan|tarikan)\s+na[fp]as\b)': '',
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — NORMALISASI BAHASA INDONESIA & BAHASA SEHARI-HARI
# Menjembatani bahasa tutur masyarakat Indonesia ke bahasa klinis rekam medis
# ══════════════════════════════════════════════════════════════════════════════

# 2a. Sapaan & Konteks Dialog Awal
DIALOG_NORMALIZATIONS = {
    r'\bpak\s+dok\b|\bbu\s+dok\b':          'dok',
    r'\bok\s+dok\b|\boke\s+dok\b':           'baik dok',
    r'\biya\s+dok\b':                        'iya dok',
}

# 2b. Bahasa Sehari-hari / Bahasa Daerah → Istilah Baku & Natural
COLLOQUIAL_TO_MEDICAL = {
    # ── Demam, Panas & Suhu ──
    r'\bpanas\s+dingin\b':                              'demam',
    r'\bgreges[-\s]?greges\b|\bgreges\b|\bnggreges\b': 'meriang',
    r'\bsumeng\b|\bsumer\b':                            'demam ringan',
    r'\bbadan\s+anget\b|\banget\s+badan\b':             'badan agak panas',
    r'\banget\b':                                       'agak panas',
    r'\bmeriang\b|\bmriang\b':                          'meriang',
    r'\bbadan\s+nggak\s+enak\b|\bbadan\s+tidak\s+enak\b': 'badan tidak enak',
    r'\bnggak\s+enak\s+badan\b|\btidak\s+enak\s+badan\b': 'badan tidak enak',
    r'\bdemam\s+tinggi\b|\bpanas\s+tinggi\b':           'demam tinggi',
    r'\bpanas\s+dalam\b':                               'panas dalam',
    r'\bmenggigil\b':                                   'menggigil',

    # ── Kepala & Saraf ──
    r'\bpuyeng\b|\bpuying\b|\bmumet\b':                 'pusing',
    r'(?<!\bpusing\s)\bkliyengan\b':                    'pusing kliyengan',
    r'\bliyur\b':                                       'pusing',
    r'\bmuter[-\s]?muter\b|\bberputar[-\s]?putar\b':     'berputar',
    r'\bpusing\s+tujuh\s+keliling\b':                   'vertigo berat',
    r'\bcekot[-\s]?cekot\b|\bnyut[-\s]?nyut(?:an)?\b|\bcenut[-\s]?cenut\b': 'sakit kepala berdenyut',
    r'\bkepala\s+berat\b':                              'sakit kepala berat',
    r'\bkesemutan\b|\bkebas\b|\bbaal\b':                'kesemutan',
    r'\bkram[-\s]?kram\b|\bkram\b':                     'kram otot',

    # ── Saluran Cerna / Pencernaan ──
    r'\bmencret\b|\bmurus\b':                           'diare',
    r'\bbuang[-\s]?buang\s+air\b':                      'buang air besar cair',
    r'\bbab\s+cair\b|\bbab\s+encer\b':                  'BAB cair',
    r'\bsembelit\b|\bsusah\s+bab\b|\bbebelen\b':        'sembelit (susah BAB)',
    r'\bmules\b|\bmulas\b':                             'mulas',
    r'\bperut\s+(?:saya\s+)?melilit\b':                 'perut melilit',
    r'\bperut\s+sakit\b|\bsakit\s+perut\b':             'sakit perut',
    r'(?<=\bkembung\s)begah\b':                         '',
    r'\bperut\s+kembung\b|\bkembung\b':                 'perut kembung',
    r'\bbegah\b|\bsebah\b':                             'kembung',
    r'\beneg\b|\benek\b|\benek\s+perutnya\b|\bpengen\s+muntah\b|\bmau\s+muntah\b': 'mual',
    r'\bmual\s+muntah\b':                               'mual dan muntah',
    r'\bmuntah[-\s]?muntah\b':                          'muntah-muntah',
    r'\bulu\s+hati\s+perih\b|\bulu\s+hati\s+sakit\b|\bnyeri\s+ulu\s+hati\b': 'nyeri ulu hati',
    r'\bmaag\s+kambuh\b|\bsakit\s+maag\b':              'sakit maag',
    r'\basam\s+lambung(?:\s+naik)?\b|\bgerd\b':         'asam lambung',
    r'\bmasuk\s+angin\b':                               'masuk angin',

    # ── Pernapasan & THT ──
    r'\bsesek\b|\bengap\b|\bngos[-\s]?ngosan\b|\bmegap[-\s]?megap\b': 'sesak napas',
    r'\bnafas\s+sesak\b|\bnapas\s+sesak\b':             'sesak napas',
    r'\bhidung\s+mampet\b|\bhidung\s+buntu\b|\bhidung\s+tersumbat\b|\bmeler\b|\bingusan\b': 'hidung tersumbat',
    r'\btenggorokan\s+gatal\b|\bgatal\s+tenggorokan\b': 'tenggorokan gatal',
    r'\btenggorokan\s+sakit\b|\bsakit\s+tenggorokan\b': 'sakit tenggorokan',
    r'\bnelen\s+sakit\b|\bsakit\s+buat\s+menelan\b|\bsakit\s+menelan\b': 'nyeri menelan',
    r'\bsuara\s+serak\b|\bsuara\s+parau\b|\bsuara\s+hilang\b': 'suara serak',
    r'\bbatuk[-\s]?batuk\b':                            'batuk-batuk',
    r'\bbatuk\s+berdahak\b|\bbatuk\s+grok[-\s]?grok\b': 'batuk berdahak',
    r'\bbatuk\s+kering\b':                              'batuk kering',

    # ── Nyeri, Otot & Tulang ──
    r'\bpegel[-\s]?linu\b|\bpegal[-\s]?linu\b|\bpegel[-\s]?pegel\b|\bpegal[-\s]?pegal\b|\bpegel\b|\bpegal\b': 'pegal-pegal linu',
    r'\bngilu\b|\bsendi\s+ngilu\b|\btulang\s+ngilu\b':  'nyeri sendi',
    r'\bencok\b|\bboyok\s+sakit\b|\bsakit\s+boyok\b':   'sakit pinggang',
    r'\blemes\s+banget\b|\bbadan\s+lemes\b|\bbadan\s+lemas\b|\bloyo\b|\blunglai\b|\bgak\s+bertenaga\b|\bnggak\s+ada\s+tenaga\b': 'badan lemas',

    # ── Saluran Kemih ──
    r'\banyang[-\s]?anyangan\b':                        'anyang-anyangan',
    r'\bpipis\s+perih\b|\bpipis\s+sakit\b':             'kencing perih',
    r'\bkencing\s+panas\b':                             'kencing panas',
    r'\bbeser\b|\bsering\s+kencing\b|\bbolak[-\s]?balik\s+pipis\b': 'sering buang air kecil',

    # ── Kulit & Alergi ──
    r'\bbentol[-\s]?bentol\b|\bbiduran\b|\bkalikata\b|\bgatal[-\s]?gatal\b|\bgatel[-\s]?gatel\b': 'biduran (gatal)',

    # ── Jantung & Dada ──
    r'\bdada\s+(?:saya\s+)?sakit(?:\s+nyeri)?\b|\bdada\s+(?:saya\s+)?nyeri\b|\bsakit\s+dada\b': 'nyeri dada',
    r'\bdeg[-\s]?degan\b|\bjantung\s+deg[-\s]?degan\b|\bjantung\s+berdebar\b': 'jantung berdebar',
    r'\bgak\s+nafsu\s+makan\b|\bnggak\s+nafsu\s+makan\b|\btidak\s+nafsu\s+makan\b|\bnafsu\s+makan\s+berkurang\b': 'nafsu makan berkurang',
    r'\bsusah\s+tidur\b|\btidak\s+bisa\s+tidur\b|\binsomnia\b': 'susah tidur',
    r'\bkeringat\s+dingin\b|\bkeringet\s+dingin\b|\bkeringat\s+malam\b': 'keringat dingin',
    r'\b(?:sudah\s+)?mendingan\b|\bsudah\s+membaik\b|\bmulai\s+enakan\b|\bagak\s+mendingan\b': 'sudah membaik',
}

# 2c. Kata Gaul / Slang / Informal → Baku
INFORMAL_TO_FORMAL = {
    # Kata ganti
    r'\bgue\b|\bgw\b|\bgua\b':                          'saya',
    r'\blo\b|\blu\b|\bloe\b':                           'Anda',
    # Negasi informal
    r'\bnggak\b|\bngga\b|\benggak\b|\bgak\b|\bnggk\b|\bndak\b|\bora\b': 'tidak',
    # Kata kerja bantu & keterangan
    r'\budah\b|\budeh\b':                               'sudah',
    r'\bbanget\b|\bbgt\b':                              'sekali',
    r'\bkalo\b|\bklo\b|\bkl\b':                         'kalau',
    r'\bcuman\b|\bcuma\b':                              'hanya',
    r'\baja\b|\baj\b':                                  'saja',
    r'\bsampe\b':                                       'sampai',
    r'\bpake\b':                                        'pakai',
    r'\bdapet\b':                                       'dapat',
    r'\bbener\b':                                       'benar',
    r'\bbikin\b':                                       'membuat',
    r'\bgimana\b|\bgimananya\b':                        'bagaimana',
    r'\bobatin\b':                                      'diobati',
    r'\bngerti\b|\bmengerti\b':                         'mengerti',
    r'\bapotik\b':                                      'apotek',
    r'\bkayak\b':                                       'seperti',
    r'\bkayaknya\b':                                    'sepertinya',
    # Filler & partikel tutur yang dibersihkan secara aman
    r'\bgitu\b|\bgituh\b|\bgini\b':                     '',
    r'\bnih\b|\bneh\b':                                 '',
    r'\bdong\b|\bdeh\b|\bsih\b':                        '',
    r'\bloh\b|\blho\b':                                 '',
    r'\btuh\b':                                         '',
    r'\banu\b':                                         '',
    r'\beee+\b|\bmmm+\b|\bh+m+\b|\buh+\b|\bumm+\b|\bah+\b': '',
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — KONTEKS KALIMAT (Maximal Dialogue Turns & Speaker Diarization)
# ══════════════════════════════════════════════════════════════════════════════

# Pola kalimat tutur dokter (pertanyaan klinis, anamnesa, instruksi pemeriksaan, resep & closing)
DOCTOR_EXPLICIT_ACTIONS = [
    # Sapaan & Pembuka Dokter
    r'\b(?:selamat\s+(?:pagi|siang|sore|malam))\b(?!\s*dok)',
    r'\bsilakan\s+(?:duduk|masuk|cerita|tiduran|berbaring)\b',
    r'\bada\s+yang\s+bisa\s+(?:saya\s+)?bantu\b',
    # Pertanyaan Anamnesis
    r'\bkeluhan(?:nya)?\s+apa\b',
    r'\bada\s+keluhan\s+apa\b',
    r'\bapa\s+yang\s+(?:dirasakan|dikeluhkan)\b',
    r'\bsakitnya?\s+(?:apa|di\s*mana|sebelah\s+mana)\b',
    r'\bkenapa\s+keluhannya\b',
    r'\bapa\s+keluhannya\b',
    r'\bkalau\s+boleh\s+tah?u\b',
    r'\bboleh\s+tah?u\b',
    r'\bsudah\s+berapa\s+lama\b|\budah\s+berapa\s+lama\b',
    r'\bberapa\s+lama\b',
    r'\bdari\s+kapan\b',
    r'\bsejak\s+kapan\b',
    r'\bmulai\s+kapan\b',
    r'\bberapa\s+hari\b',
    r'\batau\s+(?:bagaimana|gimana|apa)\b',
    # Pertanyaan gejala klinis oleh dokter (bukan sambungan keluhan pasien 'sama ada ...' atau 'ada ... juga')
    r'(?<!sama\s)(?<!terus\s)(?<!juga\s)(?<!dan\s)(?<!tidak\s)(?<!nggak\s)(?<!gak\s)\bada\s+(?:panas|demam|batuk|pilek|sesak|mual|muntah|pusing|mencret|diare|darah)\b(?!\s+juga)',
    r'\bada\s+gejala\s+(?:lain|tambahan)\b',
    r'\bada\s+keluhan\s+(?:lain|tambahan)\b',
    r'\bada\s+riwayat\b',
    # Pertanyaan alergi oleh dokter
    r'\b(?:untuk\s+)?alergi(?:\s+obat(?:nya)?)?(?:\s+(?:ibu|bapak|pak|bu|anda))?\s+ada\b',
    r'\b(?:ibu|bapak|pak|bu|anda)\s+ada\s+alergi\b',
    r'\bada\s+alergi\b|\bpunya\s+alergi\b',
    r'\bsudah\s+(?:minum\s+obat|diperiksa|berobat|ke\s+dokter)\b',
    r'\bobat\s+apa\s+yang\s+(?:sudah|pernah)\b',
    r'\bpernah\s+(?:sakit|dirawat|operasi)\b',
    # Pemeriksaan Fisik & Prosedur Medis
    r'\b(?:coba|mari|boleh|kita)\s+(?:saya\s+)?(?:cek|periksa|lihat|dengar)\b',
    r'\bkita\s+(?:cek|periksa)\s+dulu\b',
    r'\b(?:tarik|hembuskan|buang)\s+napas\b',
    r'\bbuka\s+mulutnya\b|\bjulurkan\s+lidah\b|\bbilang\s+aah\b',
    r'\b(?:tiduran|berbaring)\s+di\s+(?:sini|bed|ranjang)\b',
    # Diagnosis, TTV, Resep & Edukasi Pulang
    r'\b(?:tensinya|tekanan\s+darahnya|suhunya|nadinya|saturasinya)\s+(?:normal|tinggi|rendah|\d+)\b',
    r'\bini\s+sepertinya\s+(?:radang|flu|infeksi|maag|diare|alergi)\b',
    r'\bnanti\s+saya\s+(?:resepkan|kasih\s+obat|buatkan\s+resep)\b',
    r'\bsaya\s+(?:buatkan|tuliskan)\s+resep\b',
    r'\btebus\s+(?:obat|resep)\b',
    r'\bminum\s+obatnya\b|\bsesudah\s+makan\b|\bsebelum\s+makan\b',
    r'\bbanyak\s+istirahat\b|\bbanyak\s+minum\s+air\b',
    r'\bkontrol\s+kembali\b|\bkontrol\s+lagi\b',
    r'\bsemoga\s+(?:cepat|lekas)\s+sembuh\b',
]

# Pola kalimat tutur pasien (menyapa dok, respon, pernyataan gejala diri, timeline sakit)
PATIENT_EXPLICIT_CUES = [
    # Panggilan langsung ke dokter
    r'\bdok\b|\bdokter\b',
    # Respon konfirmasi pasien
    r'\b(?:pagi|siang|sore|malam|halo|hai|iya|ya|tidak|nggak|bukan|betul|benar|baik|siap|makasih|terima\s+kasih)\s+dok(?:ter)?\b',
    r'\b(?:kan|gitu|dong|sih|lho|kok|nih)\s+dok(?:ter)?\b',
    # Sambungan keluhan pasien: "sama ada panas juga", "terus ada pilek"
    r'\b(?:sama|terus|dan|juga)\s+ada\s+(?:panas|demam|batuk|pilek|sesak|pusing|mual|nyeri)\b',
    r'\bada\s+(?:panas|demam|batuk|pilek|sesak|pusing|mual|nyeri)\s+juga\b',
    # Pernyataan negasi / afirmasi gejala pasien
    r'\b(?:tidak|nggak|gak|ngga)\s+ada\s+(?:panas|demam|batuk|pilek|sesak|mual|muntah|pusing|darah|alergi)\b',
    r'\b(?:belum|tidak|nggak|gak)\s+(?:pernah|ada)\b',
    r'\balergi[a]?\s+aman\b|\baman\s+sih\s+(?:saya\s+)?dok\b|\baman\s+dok\b|\baman\s+saya\s+dok\b',
    # Subjek orang pertama & keluarga (pediatri / geriatri)
    r'\b(?:saya|aku|badan\s+saya|kepala\s+saya|perut\s+saya|tenggorokan\s+saya|dada\s+saya|pinggang\s+saya|kaki\s+saya)\b',
    r'\b(?:anak\s+saya|anakku|anak\s+ini|ibu\s+saya|suami\s+saya|istri\s+saya|keluarga\s+saya)\b',
    r'\b(?:ini\s+tuh\s+saya|tuh\s+saya|yang\s+saya\s+rasa(?:kan)?)\b',
    # Sensasi keluhan & durasi dari pasien
    r'\b(?:dari\s+kemarin|sejak\s+kemarin|sudah\s+(?:\d+|satu|dua|tiga|empat|lima|enam|tujuh|beberapa)\s+hari|dari\s+tadi|dari\s+semalam|kemarin\s+malam)\b',
    r'\b(?:merasa|kerasa|ngerasain|tersiksa|nggak\s+kuat)\b',
    r'\b(?:belum\s+minum\s+obat|sudah\s+minum\s+obat\s+warung|nggak\s+ada\s+alergi)\b',
    r'\bminta\s+surat\s+(?:sakit|rujukan)\b',
]

# Pola penutur pendamping pasien (keluarga/wali yang menceritakan keluhan pasien lain)
COMPANION_EXPLICIT_CUES = [
    r'\b(?:anak\s+saya|anakku|anak\s+ini|ibu\s+saya|bapak\s+saya|suami\s+saya|istri\s+saya|keluarga\s+saya|bapak\s+ini|ibu\s+ini)\b',
    r'\b(?:dia\s+ngeluh|dia\s+sakit|badannya\s+panas)\b',
]

# Pola penutur perawat / staf klinis
NURSE_EXPLICIT_CUES = [
    r'\b(?:tensi\s+dulu|timbang\s+badan|antrian\s+nomor|panggilan\s+pasien|ruang\s+dokter|suster\s+periksa|suster\s+cek|silakan\s+ke\s+poli)\b',
    r'\b(?:suster|perawat)\s+(?:sudah|sedang|cek|periksa|ukur|lapor)\b',
    r'\b(?:tensi|timbang|suhu)\s+pasien\b',
]


def _split_into_dialogue_sentences(text: str) -> list[str]:
    """
    Pecah paragraf atau transkrip percakapan menjadi kalimat-kalimat tutur utuh.
    Mendeteksi batas pergantian pembicara (turn transitions) meskipun tanpa tanda baca formal.
    """
    if not text or not text.strip():
        return []

    t = text.strip()

    # 1. Transisi: Sapaan Pasien ke Dokter -> Pertanyaan Dokter
    # Contoh: "Selamat pagi dok ada keluhan apa bu" -> "Selamat pagi dok. ada keluhan apa bu"
    t = re.sub(
        r'((?:selamat\s+(?:pagi|siang|sore|malam)|pagi|siang|sore|malam)\s+dok(?:ter)?)[,.]?\s+(?=(?:ada\s+keluhan|keluhan(?:nya)?|silakan|coba|apa\s+yang))',
        r'\1. ', t, flags=re.I
    )

    # 2. Transisi: Pertanyaan Dokter -> Pernyataan Jawaban Pasien
    # Contoh: "ada keluhan apa bu ini tuh saya panas" -> "ada keluhan apa bu? ini tuh saya panas"
    t = re.sub(
        r'((?:ada\s+keluhan\s+apa|keluhan(?:nya)?\s+apa|kenapa\s+keluhannya|dari\s+kapan|sejak\s+kapan|sudah\s+berapa\s+lama)(?:\s+(?:ya|bu|pak|mas|mba))?[.?!]*)(\s+)(?=(?:pagi\s+dok|siang\s+dok|malam\s+dok|iya\s+dok|ya\s+dok|dok\b|saya\b|ini\s+tuh|dari\b|sejak\b|sudah\b|tadi\b|kemarin\b))',
        r'\1? ', t, flags=re.I
    )

    # 3. Transisi tanda tanya yang menempel dengan kata berikutnya
    t = re.sub(r'\?([^\s.?!])', r'? \1', t)
    t = re.sub(r'\?+', '?', t)

    # 4. Pemecahan standar berdasarkan tanda baca kalimat
    parts = re.split(r'(?<=[.?!])\s+', t)
    cleaned = []
    for p in parts:
        p_str = p.strip()
        if p_str:
            cleaned.append(p_str)
    return cleaned


def _classify_sentence_speaker(text: str, last_speaker: str = None,
                                last_was_question: bool = False) -> tuple:
    """
    Klasifikasi pembicara kalimat tutur (Dokter vs Pasien vs Pendamping vs Perawat vs Unknown)
    dengan state machine percakapan RS dan identifikasi role terpisah.
    Returns: (speaker, is_question_turn, role, speaker_id, confidence)
    """
    has_q = '?' in text

    # 0. Deteksi prefix pembicara eksplisit dari transkrip atau rekaman dialog berlabel
    if re.match(r'^(?:dokter|dok)\s*[:\-]', text, re.I):
        return "Dokter", has_q, "doctor", "speaker_1", 0.99
    if re.match(r'^(?:pasien)\s*[:\-]', text, re.I):
        return "Pasien", has_q, "patient", "speaker_2", 0.99
    if re.match(r'^(?:perawat|suster)\s*[:\-]', text, re.I):
        return "Perawat", has_q, "nurse", "speaker_4", 0.99
    if re.match(r'^(?:pendamping|keluarga)\s*[:\-]', text, re.I):
        return "Pendamping", has_q, "companion", "speaker_3", 0.99

    has_dok = bool(re.search(r'(?<!^dok)(?<!^dokter)\bdok(?:ter)?\b(?!\s*[:\-])', text, re.I))
    is_doc_action = any(re.search(p, text, re.I) for p in DOCTOR_EXPLICIT_ACTIONS)
    is_patient_cue = any(re.search(p, text, re.I) for p in PATIENT_EXPLICIT_CUES)
    is_companion_cue = any(re.search(p, text, re.I) for p in COMPANION_EXPLICIT_CUES)
    is_nurse_cue = any(re.search(p, text, re.I) for p in NURSE_EXPLICIT_CUES)

    role = "unknown"
    confidence = 0.85

    # 1. Aksi eksplisit dokter (pertanyaan anamnesis, pemeriksaan, peresepan, instruksi klinis)
    if is_doc_action:
        if has_dok and not re.search(r'\b(?:coba\s+saya|nanti\s+saya|tebus|minum\s+obatnya|tensinya|suhunya|buka\s+mulut)\b', text, re.I):
            if is_companion_cue:
                role = "companion"
                confidence = 0.92
            else:
                role = "patient"
                confidence = 0.90
        else:
            role = "doctor"
            confidence = 0.95
    # 2. Respon langsung pendamping pasien (keluarga yang menceritakan keluhan pasien)
    elif is_companion_cue and (has_dok or re.search(r'\bdia\b|\banak\b', text, re.I)):
        role = "companion"
        confidence = 0.94
    # 3. Respon langsung perawat
    elif is_nurse_cue:
        role = "nurse"
        confidence = 0.92
    # 4. Respon langsung pasien ke dokter
    elif is_patient_cue:
        if has_q and not has_dok and re.search(r'\b(?:atau|bagaimana|gimana|kapan|berapa|kenapa)\b', text, re.I):
            role = "doctor"
            confidence = 0.88
        else:
            if is_companion_cue:
                role = "companion"
                confidence = 0.92
            else:
                role = "patient"
                confidence = 0.95
    # 5. Kata ganti pasien orang pertama tanpa aksi dokter
    elif re.search(r'\b(?:saya|aku|badan|kepala|perut|tenggorokan)\b', text, re.I):
        if is_companion_cue:
            role = "companion"
            confidence = 0.90
        else:
            role = "patient"
            confidence = 0.92
    # 6. Kalimat tanya klinis tanpa sapaan 'dok'
    elif has_q:
        role = "doctor"
        confidence = 0.85
    # 7. Fallback state machine
    else:
        if last_speaker in ["Dokter", "doctor"] and last_was_question:
            role = "patient"
            confidence = 0.80
        elif last_speaker in ["Pasien", "patient", "companion"]:
            role = "doctor" if has_q else "patient"
            confidence = 0.75
        else:
            role = "doctor" if has_q else "patient"
            confidence = 0.70

    # Speaker mapping (Dokter, Pasien, Pendamping, Perawat, Unknown)
    if role == "doctor":
        speaker = "Dokter"
        speaker_id = "speaker_1"
    elif role == "companion":
        speaker = "Pendamping"
        speaker_id = "speaker_3"
    elif role == "nurse":
        speaker = "Perawat"
        speaker_id = "speaker_4"
    elif role == "patient":
        speaker = "Pasien"
        speaker_id = "speaker_2"
    else:
        speaker = "Dokter" if has_q else "Pasien"
        speaker_id = "speaker_unknown"

    is_question_turn = has_q or (role == "doctor" and is_doc_action)
    return speaker, is_question_turn, role, speaker_id, confidence


def _label_speakers(segments: list[dict], fallback_text: str = "") -> list[dict]:
    """
    Layer 3: Identifikasi giliran bicara Dokter vs Pasien vs Pendamping vs Perawat.
    Mendekomposisi SETIAP segmen akustik Whisper menjadi sub-kalimat dialog dengan
    interpolasi timestamp proporsional agar tidak ada tuturan pasien/pendamping yang terlewat.
    """
    raw_segments = segments or []
    expanded_segments = []

    if not raw_segments and fallback_text:
        parts = _split_into_dialogue_sentences(fallback_text)
        total_len = max(len(fallback_text), 1)
        dur = 30.0
        cur_t = 0.0
        for p in parts:
            p_dur = round((len(p) / total_len) * dur, 2)
            expanded_segments.append({
                "start": round(cur_t, 2),
                "end": round(cur_t + p_dur, 2),
                "text": p
            })
            cur_t += p_dur
    else:
        for seg in raw_segments:
            seg_text = seg.get("text", "").strip()
            parts = _split_into_dialogue_sentences(seg_text)
            if not parts:
                continue

            if len(parts) == 1:
                expanded_segments.append(seg)
            else:
                total_len = max(sum(len(p) for p in parts), 1)
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", seg_start + 1.0)
                dur = max(seg_end - seg_start, 0.1)
                cur_t = seg_start

                for p in parts:
                    p_dur = round((len(p) / total_len) * dur, 2)
                    expanded_segments.append({
                        "start": round(cur_t, 2),
                        "end": round(cur_t + p_dur, 2),
                        "text": p
                    })
                    cur_t += p_dur

    labeled = []
    last_speaker = None
    last_was_question = False

    for seg in expanded_segments:
        raw_text = seg.get("text", "").strip()
        cleaned_text = _apply_layer2(_apply_layer1(raw_text))

        # Jika segmen hanya berupa noise/hembusan napas/filler tanpa kata bermakna, abaikan
        text_alphanumeric = re.sub(r'[^\w]', '', cleaned_text).strip()
        if not text_alphanumeric:
            continue

        text_for_eval = cleaned_text

        speaker_res = _classify_sentence_speaker(
            text_for_eval,
            last_speaker=last_speaker,
            last_was_question=last_was_question
        )

        speaker = speaker_res[0]
        is_q = speaker_res[1]
        role = speaker_res[2] if len(speaker_res) > 2 else ("doctor" if speaker == "Dokter" else "patient")
        spk_id = speaker_res[3] if len(speaker_res) > 3 else ("speaker_1" if role == "doctor" else "speaker_2")
        conf = speaker_res[4] if len(speaker_res) > 4 else 0.90

        last_speaker = speaker
        last_was_question = is_q

        start_t = seg.get("start", 0.0)
        end_t = seg.get("end", 0.0)

        labeled.append({
            **seg,
            "speaker_id": spk_id,
            "speaker_role": role,
            "speaker": speaker,
            "text": cleaned_text,
            "raw_text": raw_text,
            "start_time": start_t,
            "end_time": end_t,
            "confidence": conf,
            "review_required": True if role == "unknown" else False,
            "source": role
        })

    return labeled


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — KONTEKS MEDIS (Ekstraksi Entity Klinis & EMR Fields)
# ══════════════════════════════════════════════════════════════════════════════

# 4a. Keluhan Utama (Chief Complaint) — urut dari spesifik klinis ke umum
# Kunci trigger menggunakan kata/frasa murni (tanpa kurung/regex khusus) agar aman dengan \b
PRIMARY_SYMPTOMS = [
    # Kepala & Neurologis
    ("sakit kepala sebelah",        "Migrain / Sakit kepala sebelah"),
    ("sakit kepala berdenyut",      "Sakit kepala berdenyut"),
    ("pusing berputar",             "Vertigo / Pusing berputar"),
    ("pusing tujuh keliling",       "Vertigo berat"),
    ("vertigo berat",               "Vertigo berat"),
    ("vertigo",                     "Vertigo"),
    ("migrain",                     "Migrain"),
    ("sakit kepala berat",          "Sakit kepala berat"),
    ("sakit kepala",                "Sakit kepala"),
    ("kepala saya sakit",           "Sakit kepala"),
    ("kepala pusing",               "Pusing / Sakit kepala"),
    ("pusing kliyengan",            "Pusing kliyengan"),
    ("kliyengan",                   "Pusing kliyengan"),
    ("pusing",                      "Pusing"),

    # Saluran Kemih (prioritas tinggi karena sangat spesifik)
    ("anyang-anyangan",             "Anyang-anyangan (disuria)"),
    ("kencing perih",               "Anyang-anyangan / Kencing perih"),
    ("pipis perih",                 "Anyang-anyangan / Kencing perih"),
    ("pipis sakit",                 "Anyang-anyangan / Kencing perih"),
    ("kencing panas",               "Anyang-anyangan (disuria)"),
    ("sering buang air kecil",      "Sering buang air kecil (beser)"),
    ("beser",                       "Sering buang air kecil (beser)"),

    # Perut, Lambung & Pencernaan
    ("ulu hati perih",              "Nyeri ulu hati / Gastritis"),
    ("ulu hati sakit",              "Nyeri ulu hati / Gastritis"),
    ("nyeri ulu hati",              "Nyeri ulu hati / Gastritis"),
    ("maag kambuh",                 "Nyeri ulu hati / Gastritis"),
    ("sakit maag",                  "Nyeri ulu hati / Gastritis"),
    ("asam lambung",                "Asam lambung / GERD"),
    ("gerd",                        "Asam lambung / GERD"),
    ("perut saya melilit",          "Perut melilit"),
    ("perut melilit",               "Perut melilit"),
    ("mulas",                       "Mulas / Nyeri perut"),
    ("mules",                       "Mulas / Nyeri perut"),
    ("sakit perut",                 "Sakit perut"),
    ("perut sakit",                 "Sakit perut"),
    ("perut kembung",               "Perut kembung"),
    ("mual dan muntah",             "Mual dan muntah"),
    ("mual muntah",                 "Mual dan muntah"),
    ("muntah berkali-kali",         "Muntah berulang"),
    ("muntah-muntah",               "Muntah berulang"),
    ("muntah",                      "Muntah"),
    ("mual",                        "Mual"),
    ("mencret",                     "Diare"),
    ("bab cair",                    "Diare"),
    ("bab encer",                   "Diare"),
    ("buang-buang air",             "Diare"),
    ("diare",                       "Diare"),
    ("konstipasi",                  "Konstipasi / Susah BAB"),
    ("susah bab",                   "Konstipasi / Susah BAB"),
    ("sembelit",                    "Konstipasi / Susah BAB"),

    # Demam & Suhu
    ("demam tinggi",                "Demam tinggi"),
    ("panas tinggi",                "Demam tinggi"),
    ("demam menggigil",             "Demam menggigil"),
    ("greges-greges",               "Meriang / Demam ringan"),
    ("greges",                      "Meriang / Demam ringan"),
    ("sumeng",                      "Demam ringan"),
    ("meriang",                     "Meriang / Demam ringan"),
    ("demam",                       "Demam"),
    ("panas badan",                 "Demam / Panas badan"),
    ("badan panas",                 "Demam / Panas badan"),
    ("badan agak panas",            "Demam ringan"),
    ("panas ringan",                "Demam ringan"),
    ("panas",                       "Demam / Panas"),
    ("badan tidak enak",            "Badan tidak enak"),
    ("masuk angin",                 "Masuk angin"),

    # Pernapasan & THT
    ("batuk berdahak",              "Batuk berdahak"),
    ("batuk kering",                "Batuk kering"),
    ("batuk darah",                 "Batuk darah"),
    ("batuk berulang",              "Batuk berulang"),
    ("batuk",                       "Batuk"),
    ("sesak napas",                 "Sesak napas"),
    ("sesak",                       "Sesak napas"),
    ("pilek",                       "Pilek"),
    ("flu",                         "Flu / Pilek"),
    ("hidung tersumbat",            "Hidung tersumbat / Pilek"),
    ("hidung mampet",               "Hidung tersumbat / Pilek"),
    ("sakit tenggorokan",           "Sakit tenggorokan"),
    ("tenggorokan sakit",           "Sakit tenggorokan"),
    ("nelen sakit",                 "Sakit tenggorokan / Nyeri menelan"),
    ("sakit menelan",               "Sakit tenggorokan / Nyeri menelan"),
    ("tenggorokan gatal",           "Tenggorokan gatal"),
    ("suara serak",                 "Suara serak"),

    # Nyeri, Sendi & Otot
    ("nyeri dada",                  "Nyeri dada"),
    ("sakit dada",                  "Nyeri dada"),
    ("dada sakit",                  "Nyeri dada"),
    ("dada nyeri",                  "Nyeri dada"),
    ("jantung berdebar",            "Jantung berdebar"),
    ("nyeri pinggang",              "Nyeri pinggang"),
    ("sakit boyok",                 "Nyeri pinggang"),
    ("encok",                       "Nyeri pinggang"),
    ("nyeri sendi",                 "Nyeri sendi"),
    ("sendi ngilu",                 "Nyeri sendi / Ngilu"),
    ("pegal linu",                  "Pegal linu"),
    ("pegel linu",                  "Pegal linu"),
    ("pegal-pegal",                 "Pegal-pegal"),
    ("kesemutan",                   "Kesemutan / Kebas"),
    ("kebas",                       "Kesemutan / Kebas"),
    ("kram otot",                   "Kram otot"),
    ("badan lemas",                 "Badan lemas"),
    ("lemas",                       "Badan lemas"),

    # Alergi & Kulit
    ("biduran",                     "Biduran / Alergi gatal"),
    ("kalikata",                    "Biduran / Alergi gatal"),
    ("bentol-bentol",               "Biduran / Alergi gatal"),

    # Pemeriksaan / Penunjang Medis (Eksplisit)
    ("cek darah lengkap",           "Pemeriksaan laboratorium darah"),
    ("cek lab",                     "Pemeriksaan laboratorium darah"),
    ("cek darah",                   "Pemeriksaan laboratorium darah"),
    ("laboratorium",                "Pemeriksaan laboratorium darah"),
    ("tes urine",                   "Pemeriksaan tes urine"),
    ("rontgen dada",                "Pemeriksaan rontgen dada"),
    ("foto thorax",                 "Pemeriksaan foto thorax"),
    ("foto rontgen",                "Pemeriksaan rontgen"),
    ("rontgen",                     "Pemeriksaan rontgen"),
    ("ct scan",                     "Pemeriksaan CT Scan"),
    ("usg",                         "Pemeriksaan USG"),
    ("ekg",                         "Pemeriksaan EKG"),
    ("medical check up",            "Medical Check Up"),
    ("kontrol rutin",               "Kontrol rutin"),
    ("tekanan darah tinggi",        "Kontrol hipertensi"),
    ("darah tinggi",                "Kontrol hipertensi"),
    ("gula darah",                  "Kontrol diabetes"),
    ("kencing manis",               "Kontrol diabetes"),
]

# 4b. Keluhan Tambahan / Gejala Penyerta
SECONDARY_PATTERNS = [
    (r'\bpanas\b|\bdemam\b',                             'Panas'),
    (r'\bbatuk\b',                                      'Batuk'),
    (r'\bsesak\s+napas\b|\bsesak\b',                    'Sesak napas'),
    (r'\bpilek\b',                                      'Pilek'),
    (r'\bhidung\s+tersumbat\b|\bhidung\s+mampet\b',     'Hidung tersumbat'),
    (r'\bnyeri\s+pipi\b|\bpipi\s+(?:sakit|nyeri)\b',   'Nyeri pipi'),
    (r'\bbadan\s+tidak\s+enak\b',                       'Badan tidak enak'),
    (r'\bpanas\s+ringan\b|\bbadan\s+agak\s+panas\b',    'Panas ringan'),
    (r'\bmeriang\b|\bgreges\b',                         'Meriang'),
    (r'\bmual\b|\benek\b|\beneg\b',                     'Mual'),
    (r'\bmuntah\b',                                     'Muntah'),
    (r'\bperut\s+(?:tidak|nggak|gak)\s+sakit\b|\bsakit\s+perut\b|\bnyeri\s+perut\b|\bperut\s+(?:sakit|nyeri)\b', 'Sakit perut'),
    (r'\bflu\b',                                        'Flu'),
    (r'\bkembung\b|\bperut\s+kembung\b',                'Perut kembung'),
    (r'\bmulas\b|\bmules\b',                            'Mulas'),
    (r'\bperut\s+melilit\b',                            'Perut melilit'),
    (r'\bdiare\b|\bmencret\b|\bbab\s+cair\b',           'Diare'),
    (r'\bsembelit\b|\bsusah\s+bab\b',                   'Sembelit'),
    (r'\bpegal[-\s]?pegal\s+linu\b|\bpegal[-\s]?linu\b|\bpegal\b|\bpegel\b', 'Pegal linu'),
    (r'\blemas\b|\blesu\b|\bbadan\s+lemas\b',           'Badan lemas'),
    (r'\btenggorokan\s+gatal\b',                        'Tenggorokan gatal'),
    (r'\bsakit\s+tenggorokan\b|\btenggorokan\s+sakit\b','Sakit tenggorokan'),
    (r'\bnelen\s+sakit\b|\bsakit\s+menelan\b',          'Nyeri menelan'),
    (r'\bsuara\s+serak\b',                              'Suara serak'),
    (r'\bmenggigil\b',                                   'Menggigil'),
    (r'\bsusah\s+tidur\b|\binsomnia\b',                 'Susah tidur'),
    (r'\bkeringat\s+dingin\b|\bkeringat\s+malam\b',     'Keringat dingin/malam'),
    (r'\bkepala\s+berat\b|\bpusing\b|\bkliyengan\b',    'Pusing'),
    (r'\bnafsu\s+makan\s+(?:menurun|berkurang|hilang)\b', 'Nafsu makan menurun'),
    (r'\bpinggang\s+(?:sakit|nyeri)\b',                 'Nyeri pinggang'),
    (r'\banyang[-\s]?anyangan\b',                       'Anyang-anyangan'),
    (r'\bkencing\s+perih\b|\bpipis\s+perih\b',          'Kencing perih'),
    (r'\bnyeri\s+ulu\s+hati\b|\bmaag\b',                'Nyeri ulu hati'),
    (r'\basam\s+lambung\b',                             'Asam lambung'),
    (r'\bnyeri\s+sendi\b|\bsendi\s+ngilu\b|\bngilu\b',  'Nyeri sendi'),
    (r'\bsesak\s+ringan\b|\bagak\s+sesak\b',            'Sesak ringan'),
    (r'\bsudah\s+membaik\b|\bmendingan\b',              'Sudah membaik'),
    (r'\bmasuk\s+angin\b',                              'Masuk angin'),
    (r'\bkesemutan\b|\bkebas\b',                        'Kesemutan / kebas'),
]

# 4c. Pola Durasi / Lama Sakit (Mendukung Angka Digit & Angka Kata Bahasa Indonesia)
NUM_WORDS = r'(?:\d+|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|dua\s+belas|beberapa|separuh|setengah)'
UNIT_WORDS = r'(?:hari|minggu|bulan|tahun|jam)'
DUR_PREFIX = r'(?:dari|sejak|sudah|selama|sekitar|kurang\s+lebih|kira[-\s]?kira|hampir|baru|udah)'

DURATION_PATTERNS = [
    # "dari/sejak/sudah/sekitar N hari/minggu/bulan (lalu/ini/terakhir)"
    r'\b((?:' + DUR_PREFIX + r')\s+' + NUM_WORDS + r'\s+' + UNIT_WORDS + r'(?:\s*(?:yang\s+lalu|lalu|ini|terakhir))?)\b',
    # "dari/sejak/sudah kemarin malam/pagi/siang/sore/lusa"
    r'\b((?:dari|sejak|sudah)\s+kemar[ei]n\s+(?:mal[aeiu]m|pagi|siang|sore|lusa))\b',
    # "kemarin malam/pagi/siang/sore/lusa"
    r'\b(kemar[ei]n\s+(?:mal[aeiu]m|pagi|siang|sore|lusa))\b',
    # "dari/sejak kemarin lusa"
    r'\b((?:dari|sejak|sudah)\s+kemar[ei]n\s+lusa)\b',
    # "kemarin lusa"
    r'\b(kemar[ei]n\s+lusa)\b',
    # "dari/sejak kemarin"
    r'\b((?:dari|sejak|sudah)\s+kemar[ei]n)\b',
    # "kemarin-kemarin"
    r'\b(kemar[ei]n-kemar[ei]n)\b',
    # "dari/sejak/sudah tadi malam/pagi/siang/sore"
    r'\b((?:dari|sejak|sudah)\s+tadi\s*(?:mal[aeiu]m|pagi|siang|sore)?)\b',
    # "tadi malam / pagi / siang / sore"
    r'\b(tadi\s*(?:mal[aeiu]m|pagi|siang|sore))\b',
    # "dari/sejak semalam"
    r'\b((?:dari|sejak|sudah)\s+semalam)\b',
    # "semalam"
    r'\b(semalam)\b',
    # "sudah / sekitar seminggu / sebulan / seharian"
    r'\b((?:' + DUR_PREFIX + r'\s+)?(?:seminggu|sebulan|seharian|setahun)(?:\s*(?:yang\s+lalu|lalu|ini))?)\b',
    # "seminggu / sebulan yang lalu"
    r'\b((?:seminggu|sebulan)\s+(?:yang\s+lalu|lalu|ini))\b',
    # "N hari yang lalu / lalu / ini"
    r'\b(' + NUM_WORDS + r'\s+' + UNIT_WORDS + r'\s+(?:yang\s+lalu|lalu|ini|terakhir))\b',
    # "N hari ini"
    r'\b(' + NUM_WORDS + r'\s+hari\s+ini)\b',
    # "N hari / N minggu" (berdiri sendiri)
    r'\b(' + NUM_WORDS + r'\s+' + UNIT_WORDS + r')\b',
    # "sejak tadi" / "dari tadi"
    r'\b((?:dari|sejak)\s+tadi)\b',
]


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI PEMROSESAN LAYER & KOREKSI TERLINDUNGI (P0 / P1)
# ══════════════════════════════════════════════════════════════════════════════

def correct_with_trace(text: str) -> dict:
    """
    Koreksi teks STT dengan pelacakan jejak (traceability) & proteksi entitas klinis.
    - Memisahkan original_text dan corrected_text
    - Melindungi nama obat, dosis, angka tanda vital, dan diagnosis dari mutasi
    - Menghasilkan daftar corrections: [{"original", "corrected", "confidence", "review_required", "reason"}]
    """
    if not text or not text.strip():
        return {"original_text": text or "", "corrected_text": text or "", "corrections": []}

    original_text = text
    current_text = text
    corrections = []

    # 1. Kumpulkan rentang protected spans (dilarang dimutasi secara sembarangan)
    protected_spans = []
    for drug in PROTECTED_DRUG_NAMES:
        for m in re.finditer(r'\b' + re.escape(drug) + r'\b', current_text, re.I):
            protected_spans.append(m.span())
    for m in DOSAGE_PATTERN.finditer(current_text):
        protected_spans.append(m.span())
    for m in VITAL_NUM_PATTERN.finditer(current_text):
        protected_spans.append(m.span())
    for diag in PROTECTED_DIAGNOSES:
        for m in re.finditer(r'\b' + re.escape(diag) + r'\b', current_text, re.I):
            protected_spans.append(m.span())

    def is_span_protected(start, end):
        for p_start, p_end in protected_spans:
            if max(start, p_start) < min(end, p_end):
                return True
        return False

    # 2. Terapkan Layer 1: Koreksi Fonetik dengan pelacakan jejak
    for pat, rep in PHONETIC_CORRECTIONS.items():
        for m in list(re.finditer(pat, current_text, flags=re.I)):
            orig_span = m.span()
            orig_val = m.group(0)
            if not is_span_protected(orig_span[0], orig_span[1]):
                if orig_val.strip().lower() != rep.strip().lower() and orig_val.strip():
                    corrections.append({
                        "original": orig_val,
                        "corrected": rep,
                        "confidence": 0.88,
                        "review_required": True,
                        "reason": f"Koreksi fonetik akustik mic: '{orig_val}' -> '{rep}'"
                    })
        current_text = re.sub(pat, rep, current_text, flags=re.I)

    # 3. Terapkan Layer 2: Normalisasi Bahasa & Sehari-hari
    for pat, rep in DIALOG_NORMALIZATIONS.items():
        current_text = re.sub(pat, rep, current_text, flags=re.I)

    for pat, rep in COLLOQUIAL_TO_MEDICAL.items():
        for m in list(re.finditer(pat, current_text, flags=re.I)):
            orig_val = m.group(0)
            if orig_val.strip().lower() != rep.strip().lower() and orig_val.strip():
                corrections.append({
                    "original": orig_val,
                    "corrected": rep,
                    "confidence": 0.92,
                    "review_required": False,
                    "reason": f"Normalisasi istilah sehari-hari ke klinis: '{orig_val}' -> '{rep}'"
                })
        current_text = re.sub(pat, rep, current_text, flags=re.I)

    for pat, rep in INFORMAL_TO_FORMAL.items():
        for m in list(re.finditer(pat, current_text, flags=re.I)):
            orig_val = m.group(0)
            if orig_val.strip().lower() != rep.strip().lower() and orig_val.strip():
                if rep.strip(): # bukan penghapusan filler
                    corrections.append({
                        "original": orig_val,
                        "corrected": rep,
                        "confidence": 0.95,
                        "review_required": False,
                        "reason": f"Standardisasi bahasa tutur informal ke baku: '{orig_val}' -> '{rep}'"
                    })
        current_text = re.sub(pat, rep, current_text, flags=re.I)

    # 4. Bersihkan kata gagap dan spasi dengan melindungi kata ulang bahasa Indonesia
    current_text = _clean_stutters(current_text)

    current_text = re.sub(r'^[,\s]+', '', current_text)
    current_text = re.sub(r'\s*,\s*,+', ', ', current_text)
    current_text = re.sub(r'\s*,\s*([.?!])', r'\1', current_text)
    current_text = re.sub(r'\s+([,.:;?!])', r'\1', current_text)
    current_text = re.sub(r'\s+', ' ', current_text).strip()
    if current_text:
        current_text = current_text[0].upper() + current_text[1:]

    return {
        "original_text": original_text,
        "corrected_text": current_text,
        "corrections": corrections
    }


VALID_INDONESIAN_REDUPLICATIONS = {
    "tiba-tiba", "kira-kira", "kadang-kadang", "sama-sama", "batuk-batuk", "gatal-gatal",
    "bentol-bentol", "muntah-muntah", "kunang-kunang", "pelan-pelan", "hati-hati", "pagi-pagi",
    "malam-malam", "siang-siang", "sore-sore", "hari-hari", "kemarin-kemarin", "putar-putar",
    "muter-muter", "nyut-nyut", "cenut-cenut", "cekot-cekot", "kram-kram", "ayang-ayangan",
    "bintik-bintik", "panas-dingin", "ngos-ngosan", "megap-megap", "remang-remang"
}

def _clean_stutters(text: str) -> str:
    """
    Membersihkan kata gagap audio (stutter 3x+) tanpa merusak kata ulang bahasa Indonesia yang sah.
    Contoh: 'saya saya saya lemas' -> 'saya lemas'
    'kira kira 2 hari' -> 'kira-kira 2 hari'
    'batuk batuk' -> 'batuk-batuk'
    """
    text = re.sub(r'\b([a-zA-Z]{2,})\s+\1\s+\1\b', r'\1', text, flags=re.I)
    text = re.sub(r'\b([a-zA-Z]{2,})\s+\1\s+\1\b', r'\1', text, flags=re.I)

    def _repl_redup(m):
        w = m.group(1)
        w_low = w.lower()
        pair = f"{w_low}-{w_low}"
        if pair in VALID_INDONESIAN_REDUPLICATIONS or w_low in [
            "tiba", "kira", "kadang", "sama", "batuk", "gatal", "bentol", "muntah",
            "pelan", "hati", "pagi", "malam", "siang", "sore", "hari", "putar",
            "muter", "kram", "bintik", "ayang", "remang"
        ]:
            return f"{w}-{w}"
        return w

    text = re.sub(r'\b([a-zA-Z]{2,})\s+\1\b', _repl_redup, text, flags=re.I)
    return text


def _apply_layer1(text: str) -> str:
    """Layer 1: Koreksi fonetik audio Whisper."""
    for pat, rep in PHONETIC_CORRECTIONS.items():
        text = re.sub(pat, rep, text, flags=re.I)
    return text


def _apply_layer2(text: str) -> str:
    """Layer 2: Normalisasi Bahasa Indonesia (slang/informal → baku + istilah medis)."""
    for pat, rep in DIALOG_NORMALIZATIONS.items():
        text = re.sub(pat, rep, text, flags=re.I)
    for pat, rep in COLLOQUIAL_TO_MEDICAL.items():
        text = re.sub(pat, rep, text, flags=re.I)
    for pat, rep in INFORMAL_TO_FORMAL.items():
        text = re.sub(pat, rep, text, flags=re.I)
    text = _clean_stutters(text)
    text = re.sub(r'^[,\s]+', '', text)
    text = re.sub(r'\s*,\s*,+', ', ', text)
    text = re.sub(r'\s*,\s*([.?!])', r'\1', text)
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def _extract_duration(normalized: str) -> tuple:
    """
    Layer 4: Ekstrak durasi / lama sakit dengan deteksi koreksi ucapan (self-correction).
    Contoh: 'Saya panas sejak kemarin. Eh, bukan kemarin, sejak tadi pagi.' -> ('Sejak Tadi Pagi', 1)
    """
    candidates = []
    for i, pat in enumerate(DURATION_PATTERNS):
        for m in re.finditer(pat, normalized, re.I):
            matched_str = m.group(1).strip()
            start, end = m.span(1)
            # Cek apakah didahului negasi/ralat seperti: 'bukan', 'eh bukan', 'ralat', 'keliru'
            pre_text = normalized[max(0, start - 25):start].lower()
            is_negated = bool(re.search(r'\b(?:bukan|eh\s+bukan|ralat|gak\s+jadi|keliru|salah)\s*$', pre_text))
            candidates.append({
                "start": start,
                "end": end,
                "text": matched_str,
                "negated": is_negated,
                "priority": i
            })

    if not candidates:
        return "", 0

    candidates.sort(key=lambda x: x["start"])
    filtered = []
    for c in candidates:
        if not any(c["start"] >= f["start"] and c["end"] <= f["end"] and c != f for f in filtered):
            filtered.append(c)

    negated_keywords = set()
    for d in filtered:
        if d["negated"]:
            for kw in ["kemarin", "lusa", "tadi", "hari", "minggu", "bulan", "jam", "semalam", "pagi", "malam", "siang", "sore"]:
                if kw in d["text"].lower():
                    negated_keywords.add(kw)

    valid_candidates = []
    for d in filtered:
        if d["negated"]:
            continue
        has_negated_match = any(kw in d["text"].lower() for kw in negated_keywords)
        if has_negated_match and any(n["start"] > d["start"] for n in filtered if n["negated"]):
            continue
        valid_candidates.append(d)

    if not valid_candidates:
        return "", 0

    winner = valid_candidates[-1]
    d = winner["text"]
    d = re.sub(r'mal[aeiu]m', 'malam', d, flags=re.I)
    d = re.sub(r'kemar[ei]n', 'kemarin', d, flags=re.I)
    conf = 2 if winner["priority"] < 5 else 1
    return d.title(), conf


WORD_TO_NUMBER = {
    "satu": 1, "sehari": 1, "seminggu": 1, "sebulan": 1, "setahun": 1, "seharian": 1, "semalam": 1,
    "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7,
    "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11, "dua belas": 12
}


def _extract_structured_duration(normalized: str, labeled_segments: list = None) -> dict | None:
    """Ekstraksi durasi dan waktu klinis terstruktur (P1)."""
    dur_str, _ = _extract_duration(normalized)
    if not dur_str:
        return None

    # Tentukan sumber pembicara
    source = "patient"
    if labeled_segments:
        for seg in labeled_segments:
            if dur_str.lower() in seg.get("text", "").lower():
                source = seg.get("source", seg.get("speaker_role", "patient"))
                break

    # Parse numeric value & unit
    d_low = dur_str.lower()
    val = None
    unit = None
    is_approx = bool(re.search(r'\b(?:sekitar|kurang\s+lebih|kira[- ]kira|beberapa|hampir)\b', d_low))

    # Unit detection
    if "hari" in d_low or "seharian" in d_low:
        unit = "day"
    elif "minggu" in d_low or "seminggu" in d_low:
        unit = "week"
    elif "bulan" in d_low or "sebulan" in d_low:
        unit = "month"
    elif "tahun" in d_low or "setahun" in d_low:
        unit = "year"
    elif "jam" in d_low:
        unit = "hour"
    elif "kemarin" in d_low or "semalam" in d_low:
        unit = "day"
        val = 1
        is_approx = True

    # Numeric value detection
    if val is None:
        m_dig = re.search(r'\b(\d+)\b', d_low)
        if m_dig:
            val = int(m_dig.group(1))
        else:
            for w, n in WORD_TO_NUMBER.items():
                if re.search(r'\b' + re.escape(w) + r'\b', d_low):
                    val = n
                    break

    event_time = None
    if "kemarin malam" in d_low:
        event_time = "Kemarin malam"
    elif "kemarin lusa" in d_low:
        event_time = "Kemarin lusa"
    elif "kemarin" in d_low:
        event_time = "Kemarin"
    elif "tadi pagi" in d_low:
        event_time = "Tadi pagi"
    elif "tadi malam" in d_low or "semalam" in d_low:
        event_time = "Semalam"

    return {
        "original_text": dur_str,
        "value": val,
        "unit": unit,
        "is_approximate": is_approx,
        "source": source,
        "event_time": event_time
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 HELPER — NEGASI, REDUNDANSI & ENTITAS KLINIS
# ══════════════════════════════════════════════════════════════════════════════

NEGATION_PATTERNS = [
    # Pola langsung: "tidak ada / nggak ada / bebas / tanpa [gejala]"
    r'\b(?:tidak\s+ada|nggak\s+ada|gak\s+ada|ngga\s+ada|belum\s+ada|tanpa|bebas)\s+(?:rasa\s+|keluhan\s+)?(?:\w+\s+){0,2}%s\b',
    # Pola kata tidak: "tidak / nggak / gak [gejala]"
    r'\b(?:tidak|nggak|gak|ngga)\s+(?:\w+\s+){0,1}%s\b',
    # Pola disangkal: "[gejala] (telah )?disangkal / tidak ada"
    r'\b%s\s+(?:\w+\s+){0,2}(?:disangkal|tidak\s+ada|nggak\s+ada|nihil|negatif)\b',
    # Pola bukan: "bukan [gejala]"
    r'\bbukan\s+%s\b',
]

def _is_symptom_negated(term: str, sentence: str) -> bool:
    """Cek apakah istilah gejala (term) dinegasikan di dalam kalimat tertentu."""
    clean_s = sentence.lower()
    t = term.lower()
    if any(neg in t for neg in ["tidak", "nggak", "gak", "ngga", "bukan", "tanpa", "bebas", "belum"]):
        return True
    if not re.search(r'\b' + re.escape(t) + r'\b', clean_s):
        return False

    for pat_template in NEGATION_PATTERNS:
        pat = pat_template % re.escape(t)
        m = re.search(pat, clean_s)
        if m:
            span_text = m.group(0)
            if re.search(r'\b(?:tidak\s+hanya|bukan\s+hanya|bukan\s+cuma)\b', span_text):
                continue
            return True
    return False


REDUNDANCY_MAP = {
    "Batuk": ["batuk berdahak", "batuk kering", "batuk darah", "batuk berulang"],
    "Pusing": ["pusing kliyengan", "vertigo", "migrain", "sakit kepala berdenyut", "sakit kepala sebelah", "sakit kepala berat"],
    "Demam": ["demam tinggi", "demam menggigil", "meriang / demam ringan", "demam / panas", "demam ringan"],
    "Nyeri ulu hati": ["nyeri ulu hati / gastritis", "asam lambung / gerd"],
    "Diare": ["buang air besar cair", "bab cair", "diare"],
    "Anyang-anyangan": ["anyang-anyangan (disuria)", "anyang-anyangan / kencing perih"],
    "Kencing perih": ["anyang-anyangan (disuria)", "anyang-anyangan / kencing perih"],
    "Nyeri pinggang": ["sakit boyok", "encok"],
    "Sesak napas": ["sesak napas / asma", "sesak berat"],
}

def _is_redundant(label: str, existing_list: list[str]) -> bool:
    """Cek apakah gejala bersifat redundan / duplikat dari gejala yang lebih spesifik."""
    l_lower = label.lower()
    for ex in existing_list:
        if l_lower == ex.lower():
            return True
    for general, specifics in REDUNDANCY_MAP.items():
        if l_lower == general.lower():
            for ex in existing_list:
                for sp in specifics:
                    if sp in ex.lower():
                        return True
    return False


def _extract_vitals(text: str) -> dict:
    """Ekstraksi Tanda-Tanda Vital (TTV) legacy format string."""
    vitals = {}
    m_bp = re.search(r'\b(?:tensi(?:nya)?|tekanan\s+darah)(?:\s+pasien)?(?:\s+adalah)?\s*[:=]?\s*(\d{2,3})\s*(?:[/]|per)\s*(\d{2,3})\b', text, re.I)
    if not m_bp:
        m_bp = re.search(r'\b(\d{2,3})\s*(?:[/]|per)\s*(\d{2,3})\s*(?:mm\s*hg)\b', text, re.I)
    if not m_bp:
        m_bp = re.search(r'\b(\d{2,3}\s*/\s*\d{2,3})\b', text, re.I)
    if m_bp:
        if len(m_bp.groups()) >= 2 and m_bp.group(2):
            vitals["tensi"] = f"{m_bp.group(1)}/{m_bp.group(2)} mmHg"
        else:
            bp_val = re.sub(r'\s+', '', m_bp.group(1))
            vitals["tensi"] = f"{bp_val} mmHg"

    m_temp = re.search(r'\b(?:suhu(?:nya)?|demam)(?:\s+badan)?(?:\s+pasien)?\s*[:=]?\s*(\d{2}(?:[.,]\d+)?)\s*(?:derajat(?:\s*c(?:el[cs]ius)?)?|°\s*c|c\b)?', text, re.I)
    if not m_temp:
        m_temp = re.search(r'\b(\d{2}[.,]\d+)\s*(?:derajat|°\s*c)\b', text, re.I)
    if m_temp:
        val = m_temp.group(1).replace(',', '.')
        try:
            val_f = float(val)
            if 34.0 <= val_f <= 43.0:
                vitals["suhu"] = f"{val}°C"
        except ValueError:
            pass

    m_spo2 = re.search(r'\b(?:saturasi(?:nya)?|spo2)(?:\s+oksigen)?\s*[:=]?\s*(\d{2,3})\s*(?:%|persen)?\b', text, re.I)
    if m_spo2:
        val = int(m_spo2.group(1))
        if 50 <= val <= 100:
            vitals["spo2"] = f"{val}%"

    m_hr = re.search(r'\b(?:nadi(?:nya)?|denyut\s+nadi)\s*[:=]?\s*(\d{2,3})\s*(?:x/menit|bpm|kali/menit|kali\s+per\s+menit)?\b', text, re.I)
    if m_hr:
        val = int(m_hr.group(1))
        if 40 <= val <= 200:
            vitals["nadi"] = f"{val} x/menit"

    return vitals


def _extract_extended_vitals(text: str, labeled_segments: list = None) -> dict:
    """Ekstraksi Tanda Vital Lengkap dengan Validasi Fisiologis Medis (P1)."""
    ext_vitals = {
        "blood_pressure": None,
        "temperature": None,
        "spo2": None,
        "pulse": None,
        "respiratory_rate": None,
        "weight": None,
        "height": None,
        "pain_scale": None
    }

    # Cari sumber pembicara yang menyebutkan TTV
    def get_source_for_term(pattern):
        if labeled_segments:
            for seg in labeled_segments:
                if re.search(pattern, seg.get("text", ""), re.I):
                    return seg.get("source", seg.get("speaker_role", "doctor"))
        return "doctor"

    # 1. Tekanan Darah (mmHg)
    m_bp = re.search(r'\b(?:tensi(?:nya)?|tekanan\s+darah)(?:\s+pasien)?(?:\s+adalah)?\s*[:=]?\s*(\d{2,3})\s*(?:[/]|per)\s*(\d{2,3})\b', text, re.I)
    if not m_bp:
        m_bp = re.search(r'\b(\d{2,3})\s*(?:[/]|per)\s*(\d{2,3})\s*(?:mm\s*hg)\b', text, re.I)
    if not m_bp:
        m_bp = re.search(r'\b(\d{2,3})\s*[/]\s*(\d{2,3})\b', text, re.I)
    if m_bp:
        sys = int(m_bp.group(1))
        dia = int(m_bp.group(2))
        src = get_source_for_term(r'\b\d{2,3}\s*(?:[/]|per)\s*\d{2,3}\b')
        rev = bool(sys < 70 or sys > 240 or dia < 40 or dia > 140)
        ext_vitals["blood_pressure"] = {
            "value": f"{sys}/{dia}",
            "systolic": sys,
            "diastolic": dia,
            "unit": "mmHg",
            "source": src,
            "review_required": rev
        }

    # 2. Suhu Tubuh (°C)
    m_temp = re.search(r'\b(?:suhu(?:nya)?|demam)(?:\s+badan)?(?:\s+pasien)?\s*[:=]?\s*(\d{2}(?:[.,]\d+)?)\s*(?:derajat(?:\s*c(?:el[cs]ius)?)?|°\s*c|c\b)?', text, re.I)
    if not m_temp:
        m_temp = re.search(r'\b(\d{2}[.,]\d+)\s*(?:derajat|°\s*c)\b', text, re.I)
    if m_temp:
        val_str = m_temp.group(1).replace(',', '.')
        try:
            val_f = float(val_str)
            src = get_source_for_term(r'\b' + re.escape(m_temp.group(1)) + r'\b')
            rev = bool(val_f < 36.0 or val_f >= 38.0)
            ext_vitals["temperature"] = {
                "value": val_f,
                "unit": "°C",
                "source": src,
                "review_required": rev
            }
        except ValueError:
            pass

    # 3. SpO2 (%)
    m_spo2 = re.search(r'\b(?:saturasi(?:nya)?|spo2)(?:\s+oksigen)?\s*[:=]?\s*(\d{2,3})\s*(?:%|persen)?\b', text, re.I)
    if m_spo2:
        val = int(m_spo2.group(1))
        src = get_source_for_term(r'\b(?:saturasi|spo2)\b')
        rev = bool(val < 70 or val > 100)
        ext_vitals["spo2"] = {
            "value": val,
            "unit": "%",
            "source": src,
            "review_required": rev
        }

    # 4. Denyut Nadi (bpm)
    m_hr = re.search(r'\b(?:nadi(?:nya)?|denyut\s+nadi|heart\s+rate|hr)\s*[:=]?\s*(\d{2,3})\s*(?:x/menit|bpm|kali/menit|kali\s+per\s+menit)?\b', text, re.I)
    if m_hr:
        val = int(m_hr.group(1))
        src = get_source_for_term(r'\b(?:nadi|denyut)\b')
        rev = bool(val < 40 or val > 180)
        ext_vitals["pulse"] = {
            "value": val,
            "unit": "bpm",
            "source": src,
            "review_required": rev
        }

    # 5. Frekuensi Napas / Respiratory Rate (x/min)
    m_rr = re.search(r'\b(?:rr|respirasi|frekuensi\s+na[fp]as|perna[fp]asan|na[fp]as(?:nya)?)\s*[:=]?\s*(\d{1,2})\s*(?:x/menit|kali/menit|kali\s+per\s+menit|cpm)?\b', text, re.I)
    if m_rr:
        val = int(m_rr.group(1))
        src = get_source_for_term(r'\b(?:rr|respirasi|frekuensi\s+na[fp]as|na[fp]as)\b')
        rev = bool(val < 10 or val > 50)
        ext_vitals["respiratory_rate"] = {
            "value": val,
            "unit": "x/min",
            "source": src,
            "review_required": rev
        }

    # 6. Berat Badan (kg)
    m_bb = re.search(r'\b(?:berat\s+badan|bb)\s*[:=]?\s*(\d{1,3}(?:[.,]\d+)?)\s*(?:kg|kilo(?:gram)?)\b', text, re.I)
    if m_bb:
        val_str = m_bb.group(1).replace(',', '.')
        try:
            val_f = float(val_str)
            src = get_source_for_term(r'\b(?:berat\s+badan|bb)\b')
            rev = bool(val_f < 2.0 or val_f > 250.0)
            ext_vitals["weight"] = {
                "value": val_f,
                "unit": "kg",
                "source": src,
                "review_required": rev
            }
        except ValueError:
            pass

    # 7. Tinggi Badan (cm)
    m_tb = re.search(r'\b(?:tinggi\s+badan|tb)\s*[:=]?\s*(\d{2,3}(?:[.,]\d+)?)\s*(?:cm|sentimeter)\b', text, re.I)
    if m_tb:
        val_str = m_tb.group(1).replace(',', '.')
        try:
            val_f = float(val_str)
            src = get_source_for_term(r'\b(?:tinggi\s+badan|tb)\b')
            rev = bool(val_f < 40.0 or val_f > 220.0)
            ext_vitals["height"] = {
                "value": val_f,
                "unit": "cm",
                "source": src,
                "review_required": rev
            }
        except ValueError:
            pass

    # 8. Skala Nyeri (0-10)
    m_pain = re.search(r'\b(?:skala\s+nyeri|nyeri(?:nya)?\s+skala)\s*[:=]?\s*(\d{1,2})(?:\s*[/]\s*10)?\b', text, re.I)
    if m_pain:
        val = int(m_pain.group(1))
        src = get_source_for_term(r'\bskala\s+nyeri\b')
        rev = bool(val < 0 or val > 10)
        ext_vitals["pain_scale"] = {
            "value": val,
            "unit": "/10",
            "source": src,
            "review_required": rev
        }

    return ext_vitals


def _extract_allergies(text: str, labeled_segments: list = None) -> dict:
    """Ekstraksi riwayat alergi legacy (ada_alergi, alergen, disangkal)."""
    res = {"ada_alergi": False, "alergen": [], "disangkal": False}

    patient_sentences = []
    if labeled_segments:
        patient_sentences = [s.get("text", "") for s in labeled_segments if s.get("speaker_role") in ["patient", "companion"] or s.get("speaker") == "Pasien"]

    search_sentences = patient_sentences if patient_sentences else [text]
    combined_patient_text = " ".join(search_sentences)

    disangkal_patterns = [
        r'\b(?:tidak\s+ada|nggak\s+ada|gak\s+ada|belum\s+pernah\s+ada)\s+alergi\b',
        r'\balergi\s+(?:\w+\s+){0,2}disangkal\b',
        r'\balergi[a]?\s+aman\b',
        r'\baman\s+sih\s+(?:saya\s+)?dok\b',
        r'\baman\s+dok\b',
        r'\baman\s+saya\s+dok\b',
    ]
    if any(re.search(p, combined_patient_text, re.I) for p in disangkal_patterns):
        res["disangkal"] = True
        return res

    for sent in search_sentences:
        m = re.findall(r'\balergi\s+(?:sama\s+|pada\s+|terhadap\s+|obat\s+)?([a-zA-Z\s]+?)(?=[,.]|$|\bdok\b|\bdan\b)', sent, re.I)
        for match in m:
            item = match.strip().lower()
            item = re.sub(r'^(?:obat|makanan|ada|aman|sih|saya)\s+', '', item).strip()
            if item and item not in ['tidak', 'nggak', 'gak', 'belum', 'apa', 'ya', 'ibu', 'ada', 'obatnya ibu ada', 'aman']:
                res["ada_alergi"] = True
                res["alergen"].append(item.title())

    return res


def _extract_standardized_allergies(text: str, labeled_segments: list = None) -> dict:
    """
    Ekstraksi riwayat alergi klinis standar (P0).
    Status: reported | denied | unknown | not_asked | uncertain
    """
    # Cek apakah topik alergi sama sekali disinggung di dialog
    has_allergy_topic = bool(re.search(r'\balergi\b', text, re.I))
    if not has_allergy_topic:
        return {
            "status": "not_asked",
            "items": [],
            "reactions": [],
            "source": None,
            "review_required": False,
            "category": None
        }

    patient_sentences = []
    doctor_sentences = []
    if labeled_segments:
        for s in labeled_segments:
            role = s.get("speaker_role", s.get("source", "patient"))
            if role in ["patient", "companion"]:
                patient_sentences.append(s.get("text", ""))
            elif role == "doctor":
                doctor_sentences.append(s.get("text", ""))
    else:
        patient_sentences = [text]

    combined_patient = " ".join(patient_sentences)

    # 1. Pasien menyatakan tidak tahu / belum tahu
    if re.search(r'\b(?:kurang\s+tahu|tidak\s+tahu|nggak\s+tahu|belum\s+tahu|tidak\s+ingat|lupa)\s+(?:\w+\s+){0,4}alergi\b|\balergi(?:\s+obat)?\s+(?:kurang|tidak|nggak)\s+tahu\b', combined_patient, re.I):
        return {
            "status": "unknown",
            "items": [],
            "reactions": [],
            "source": "patient",
            "review_required": True,
            "category": None
        }

    # 2. Pasien menyangkal alergi
    disangkal_patterns = [
        r'\b(?:tidak\s+punya|nggak\s+punya|gak\s+punya|tidak\s+ada|nggak\s+ada|gak\s+ada|belum\s+pernah\s+ada)\s+(?:riwayat\s+)?alergi\b',
        r'\b(?:riwayat\s+)?alergi\s+(?:obat\s+|makanan\s+)?(?:tidak\s+ada|nggak\s+ada|disangkal|nihil|negatif)\b',
        r'\b(?:tidak\s+ada|nggak\s+ada|tidak\s+punya|nggak\s+punya)\s+(?:alergi|alergi\s+obat|riwayat\s+alergi)\b',
        r'\btanpa\s+riwayat\s+alergi\b',
        r'\balergi\s+(?:\w+\s+){0,2}disangkal\b',
        r'\balergi[a]?\s+aman\b',
        r'\baman\s+sih\s+(?:saya\s+)?dok\b',
        r'\baman\s+dok\b',
        r'\baman\s+saya\s+dok\b',
    ]
    if any(re.search(p, combined_patient, re.I) for p in disangkal_patterns):
        return {
            "status": "denied",
            "items": [],
            "reactions": [],
            "source": "patient",
            "review_required": False,
            "category": None
        }

    # 3. Pasien melaporkan alergi spesifik
    allergens = []
    reactions = []
    category = None

    for sent in patient_sentences:
        # 3a. Deteksi khusus alergi obat umum vs spesifik
        if re.search(r'\balergi\s+(?:sama\s+|pada\s+)?obat\b', sent, re.I):
            m_sp = re.search(r'\balergi\s+(?:sama\s+|pada\s+)?obat\s+([a-zA-Z]+)\b', sent, re.I)
            if m_sp and m_sp.group(1).lower() not in ['dok', 'dokter', 'apa', 'ya', 'sih', 'tapi', 'dan']:
                specific_drug = m_sp.group(1).title()
                if specific_drug not in allergens:
                    allergens.append(specific_drug)
                category = "drug"
            else:
                if "Obat (belum spesifik)" not in allergens:
                    allergens.append("Obat (belum spesifik)")
                category = "drug"
        else:
            m = re.findall(r'\balergi\s+(?:sama\s+|pada\s+|terhadap\s+)?([a-zA-Z\s]+?)(?=[,.]|$|\bdok\b|\bdan\b)', sent, re.I)
            for match in m:
                item = match.strip().lower()
                item = re.sub(r'^(?:makanan|ada|aman|sih|saya)\s+', '', item).strip()
                if item and item not in ['tidak', 'nggak', 'gak', 'belum', 'apa', 'ya', 'ibu', 'ada', 'obatnya ibu ada', 'aman', 'dok', 'dokter']:
                    item_tit = item.title()
                    if item_tit not in allergens:
                        allergens.append(item_tit)
                    if any(d in item for d in ['amox', 'penisilin', 'paracetamol', 'sulfa', 'aspirin', 'antibiotik', 'obat']):
                        category = "drug"
                    elif any(f in item for f in ['udang', 'telur', 'kacang', 'seafood', 'ikan', 'susu']):
                        category = "food"
                    else:
                        category = "other"

        # Cari reaksi alergi yang menyertai
        m_reac = re.search(r'\b(?:bentol[-\s]?bentol|gatal[-\s]?gatal|biduran|bengkak|sesak\s+napas|ruam)\b', sent, re.I)
        if m_reac:
            reac_cap = m_reac.group(0).capitalize()
            if reac_cap not in reactions:
                reactions.append(reac_cap)

    if allergens:
        is_unspecified = any("belum spesifik" in it for it in allergens)
        return {
            "status": "reported",
            "items": allergens,
            "reactions": reactions,
            "source": "patient",
            "review_required": is_unspecified,
            "category": category or "drug"
        }

    # 4. Dokter bertanya alergi tetapi respon pasien ambigu / belum ada
    return {
        "status": "uncertain",
        "items": [],
        "reactions": [],
        "source": "doctor",
        "review_required": True,
        "category": None
    }


def _extract_chief_complaint(normalized: str, labeled_segments: list = None) -> tuple:
    """Layer 4: Ekstrak keluhan utama legacy string."""
    patient_sentences = []
    if labeled_segments:
        patient_sentences = [s.get("text", "") for s in labeled_segments if s.get("speaker_role") in ["patient", "companion"] or s.get("speaker") == "Pasien"]

    if patient_sentences:
        for sent in patient_sentences:
            for trigger, label in PRIMARY_SYMPTOMS:
                if re.search(r'\b' + re.escape(trigger) + r'\b', sent, re.I):
                    if not _is_symptom_negated(trigger, sent):
                        return label, 2

    if '?' in normalized:
        parts = normalized.split('?')
        post_q = " ".join(parts[1:])
        for trigger, label in PRIMARY_SYMPTOMS:
            if re.search(r'\b' + re.escape(trigger) + r'\b', post_q, re.I):
                if not _is_symptom_negated(trigger, post_q):
                    return label, 2

    sentences = _split_into_dialogue_sentences(normalized)
    for sent in sentences:
        for trigger, label in PRIMARY_SYMPTOMS:
            if re.search(r'\b' + re.escape(trigger) + r'\b', sent, re.I):
                if not _is_symptom_negated(trigger, sent):
                    return label, 1

    return "", 0


ANATOMICAL_BODY_PARTS = {
    "sakit kepala": "Kepala", "pusing": "Kepala", "vertigo": "Kepala / Saraf", "migrain": "Kepala",
    "batuk": "Saluran Pernapasan", "sesak": "Saluran Pernapasan / Dada", "pilek": "Saluran Pernapasan / Hidung", "flu": "Saluran Pernapasan",
    "tenggorokan": "Tenggorokan / THT", "nyeri dada": "Dada / Kardiovaskular",
    "perut": "Saluran Cerna / Abdomen", "sakit perut": "Saluran Cerna / Abdomen", "mual": "Saluran Cerna", "muntah": "Saluran Cerna", "diare": "Saluran Cerna",
    "maag": "Saluran Cerna / Lambung", "ulu hati": "Saluran Cerna / Lambung", "sembelit": "Saluran Cerna",
    "anyang-anyangan": "Saluran Kemih", "kencing": "Saluran Kemih",
    "pinggang": "Pinggang / Punggung", "sendi": "Sendi & Otot", "pegal": "Muskuloskeletal",
    "demam": "Sistemik / Seluruh Tubuh", "panas": "Sistemik / Seluruh Tubuh", "meriang": "Sistemik / Seluruh Tubuh"
}


def _extract_structured_chief_complaint(normalized: str, labeled_segments: list = None) -> dict:
    """Ekstraksi keluhan utama terstruktur klinis (P1)."""
    chief_str, pts = _extract_chief_complaint(normalized, labeled_segments=labeled_segments)
    if not chief_str:
        return {
            "value": None,
            "body_part": None,
            "onset": None,
            "duration": None,
            "severity": None,
            "source": None,
            "confidence": None,
            "review_required": True
        }

    # Deteksi body part
    b_part = "Sistemik / Seluruh Tubuh"
    c_low = chief_str.lower()
    for k, bp in ANATOMICAL_BODY_PARTS.items():
        if k in c_low:
            b_part = bp
            break

    # Deteksi sumber keluhan (patient vs companion)
    src = "patient"
    onset_val = None
    if labeled_segments:
        for seg in labeled_segments:
            txt_low = seg.get("text", "").lower()
            if any(trig in txt_low for trig, lbl in PRIMARY_SYMPTOMS if lbl == chief_str):
                src = seg.get("source", seg.get("speaker_role", "patient"))
                break

    # Durasi & keparahan
    dur_dict = _extract_structured_duration(normalized, labeled_segments=labeled_segments)
    dur_val = dur_dict.get("original_text") if dur_dict else None
    if dur_dict:
        onset_val = dur_dict.get("event_time")

    severity = None
    if any(w in normalized.lower() for w in ["berat", "parah", "tidak tertahankan", "sangat", "tinggi"]):
        severity = "berat"
    elif any(w in normalized.lower() for w in ["ringan", "agak", "sedikit", "mulai membaik"]):
        severity = "ringan"

    return {
        "value": chief_str,
        "body_part": b_part,
        "onset": onset_val,
        "duration": dur_val,
        "severity": severity,
        "source": src,
        "confidence": 0.95 if pts == 2 else 0.85,
        "review_required": False
    }


def _extract_secondary(normalized: str, chief: str, labeled_segments: list = None) -> list:
    """Layer 4: Ekstrak gejala tambahan legacy list[str]."""
    found = []
    patient_sentences = []
    if labeled_segments:
        patient_sentences = [s.get("text", "") for s in labeled_segments if s.get("speaker_role") in ["patient", "companion"] or s.get("speaker") == "Pasien"]

    sentences_to_search = patient_sentences if patient_sentences else _split_into_dialogue_sentences(normalized)

    for pat, label in SECONDARY_PATTERNS:
        for sent in sentences_to_search:
            m = re.search(pat, sent, re.I)
            if m:
                matched_term = m.group(0)
                if not _is_symptom_negated(matched_term, sent):
                    if label.lower() not in chief.lower() and not _is_redundant(label, [chief] + found):
                        found.append(label)
                        break

    return found


def _extract_structured_symptoms(normalized: str, chief: str, labeled_segments: list = None) -> list:
    """Ekstraksi daftar gejala terstruktur klinis dengan status present vs absent dan sumber pembicara akurat."""
    symptoms = []
    seen = set()

    search_tuples = []
    if labeled_segments:
        for s in labeled_segments:
            role = s.get("speaker_role", "patient")
            spk = s.get("speaker", "")
            if role in ["patient", "companion"] or spk in ["Pasien", "Pendamping"]:
                src = "companion" if (role == "companion" or spk == "Pendamping") else "patient"
                search_tuples.append((s.get("text", ""), src))
    if not search_tuples:
        for s in _split_into_dialogue_sentences(normalized):
            search_tuples.append((s, "patient"))

    def _is_same_symptom(l1, l2):
        if not l1 or not l2:
            return False
        l1_low, l2_low = l1.lower(), l2.lower()
        if l1_low == l2_low:
            return True
        if l1_low in l2_low or l2_low in l1_low:
            return True
        if ("demam" in l1_low or "panas" in l1_low) and ("demam" in l2_low or "panas" in l2_low):
            return True
        if ("pusing" in l1_low or "sakit kepala" in l1_low) and ("pusing" in l2_low or "sakit kepala" in l2_low):
            return True
        return False

    # 1. Gejala keluhan utama
    if chief:
        b_part = "Sistemik / Seluruh Tubuh"
        for k, bp in ANATOMICAL_BODY_PARTS.items():
            if k in chief.lower():
                b_part = bp
                break

        chief_src = "patient"
        for sent, src in search_tuples:
            if chief.lower() in sent.lower() or any(w.strip() in sent.lower() for w in chief.lower().split('/')):
                chief_src = src
                break

        symptoms.append({
            "name": chief,
            "status": "present",
            "source": chief_src,
            "body_part": b_part,
            "frequency": None
        })
        seen.add(chief.lower())

    # 2. Gejala sekunder (present vs absent / negated)
    for pat, label in SECONDARY_PATTERNS:
        for sent, sent_src in search_tuples:
            m = re.search(pat, sent, re.I)
            if m:
                matched_term = m.group(0)
                negated = _is_symptom_negated(matched_term, sent)
                status = "absent" if negated else "present"
                key = (label.lower(), status)
                if key not in seen and not _is_same_symptom(label, chief):
                    b_part = "Sistemik"
                    for k, bp in ANATOMICAL_BODY_PARTS.items():
                        if k in label.lower():
                            b_part = bp
                            break

                    symptoms.append({
                        "name": label,
                        "status": status,
                        "source": sent_src,
                        "body_part": b_part,
                        "frequency": None
                    })
                    seen.add(key)
                break

    # 3. Analisis Negasi Antar-Giliran Bicara (Cross-Turn Dialogue Anamnesis)
    # Contoh: Dokter: "Apakah ada batuk?" -> Pasien: "Tidak ada dok."
    segments_to_check = labeled_segments or _label_speakers([], normalized)
    if segments_to_check and len(segments_to_check) >= 2:
        for i in range(len(segments_to_check) - 1):
            s1 = segments_to_check[i]
            s2 = segments_to_check[i + 1]
            r1 = s1.get("speaker_role", s1.get("source", ""))
            r2 = s2.get("speaker_role", s2.get("source", ""))
            t1 = s1.get("text", "")
            t2 = s2.get("text", "")

            # Jika dokter bertanya gejala dan pasien menyangkal (tidak ada / nggak ada / tidak)
            if r1 in ["doctor", "Dokter"] and r2 in ["patient", "companion", "Pasien", "Pendamping"]:
                is_denial = bool(re.search(r'\b(?:tidak\s+ada|nggak\s+ada|gak\s+ada|ngga\s+ada|belum\s+ada|tidak|nggak|gak|nihil)\b', t2, re.I))
                if is_denial:
                    for pat, label in SECONDARY_PATTERNS:
                        if re.search(pat, t1, re.I):
                            key = (label.lower(), "absent")
                            if key not in seen and not _is_same_symptom(label, chief):
                                b_part = "Sistemik"
                                for k, bp in ANATOMICAL_BODY_PARTS.items():
                                    if k in label.lower():
                                        b_part = bp
                                        break
                                symptoms.append({
                                    "name": label,
                                    "status": "absent",
                                    "source": "patient",
                                    "body_part": b_part,
                                    "frequency": None
                                })
                                seen.add(key)

    return symptoms


def _extract_ancillary_entities(text: str) -> tuple:
    """Ekstraksi riwayat penyakit, obat, dan prosedur pemeriksaan yang disebutkan."""
    history = []
    meds = []
    procs = []
    uncat = []

    # Riwayat penyakit
    for diag in PROTECTED_DIAGNOSES:
        if re.search(r'\b(?:riwayat|punya|pernah)\s+(?:\w+\s+){0,2}' + re.escape(diag) + r'\b', text, re.I):
            history.append(diag.title())

    # Obat yang disebutkan
    for drug in PROTECTED_DRUG_NAMES:
        if re.search(r'\b' + re.escape(drug) + r'\b', text, re.I):
            meds.append(drug.title())

    # Prosedur penunjang
    proc_patterns = [
        (r'\brontgen(?:\s+dada)?\b|\bfoto\s+thorax\b', 'Rontgen Dada'),
        (r'\bcek\s+lab(?:oratorium)?\b|\blaborat\b', 'Laboratorium Darah'),
        (r'\bct\s*scan\b', 'CT Scan'),
        (r'\busg\b', 'USG'),
        (r'\bekg\b', 'EKG'),
        (r'\btes\s+urine\b', 'Tes Urine'),
    ]
    for pat, name in proc_patterns:
        if re.search(pat, text, re.I):
            procs.append(name)

    return history, meds, procs, uncat


class ClinicalDataValidator:
    """Lapisan Validasi Klinis dan Konsistensi Data (P0 / P1)."""
    @staticmethod
    def validate(emr_data: dict) -> dict:
        issues = []
        chief = emr_data.get("chief_complaint") or {}
        chief_val = chief.get("value")

        # 1. Keluhan utama tidak boleh kosong jika pasien menyampaikan keluhan
        symptoms = emr_data.get("symptoms", [])
        present_syms = [s for s in symptoms if s.get("status") == "present"]
        if not chief_val and present_syms:
            issues.append({
                "field": "chief_complaint",
                "value": None,
                "reason": "Pasien menyampaikan gejala namun keluhan utama belum ditentukan secara definitif.",
                "severity": "warning",
                "action": "Tinjau gejala dan tentukan keluhan utama sebelum finalisasi EMR."
            })

        # 2. Validasi fisiologis tanda vital
        vitals = emr_data.get("vitals") or {}
        for vk, vo in vitals.items():
            if vo and isinstance(vo, dict) and vo.get("review_required"):
                is_extreme = False
                if vk == "blood_pressure":
                    val_str = str(vo.get("value", ""))
                    if "/" in val_str:
                        try:
                            s, d = map(int, val_str.split("/"))
                            if s > 260 or s < 50 or d > 160 or d < 30:
                                is_extreme = True
                        except Exception:
                            pass
                elif vk == "temperature":
                    try:
                        t = float(vo.get("value", 0))
                        if t < 33.0 or t > 43.0:
                            is_extreme = True
                    except Exception:
                        pass

                severity = "error" if (is_extreme or vo.get("severity") == "error") else "warning"
                issues.append({
                    "field": f"vitals.{vk}",
                    "value": vo.get("value"),
                    "reason": f"Nilai tanda vital {vk} ({vo.get('value')} {vo.get('unit')}) berada di luar batas fisiologis wajar.",
                    "severity": severity,
                    "action": "Verifikasi nilai tanda vital dengan dokter atau perawat pemeriksa."
                })

        # 3. Validasi status alergi
        allergies = emr_data.get("allergies") or {}
        if allergies.get("status") in ["unknown", "uncertain"]:
            issues.append({
                "field": "allergies",
                "value": allergies.get("status"),
                "reason": "Status alergi obat pasien belum terkonfirmasi pasti.",
                "severity": "warning",
                "action": "Konfirmasi riwayat alergi obat kepada pasien sebelum memberikan terapi."
            })

        # 4. Validasi penelusuran durasi ke teks sumber
        dur = emr_data.get("duration")
        if dur and dur.get("original_text"):
            orig_text = dur.get("original_text", "").lower()
            source_text = (emr_data.get("original_text") or emr_data.get("raw_transcript") or "").lower()
            words_dur = [w for w in orig_text.split() if len(w) > 2]
            if source_text and words_dur and not any(w in source_text for w in words_dur):
                issues.append({
                    "field": "duration",
                    "value": dur.get("original_text"),
                    "reason": "Durasi waktu tidak ditemukan pada transkrip sumber ucapan pasien.",
                    "severity": "error",
                    "action": "Hapus atau sesuaikan durasi dengan rekaman asli."
                })

        has_error = any(i["severity"] == "error" for i in issues)
        return {
            "valid": not has_error,
            "issues": issues,
            "total_issues": len(issues)
        }


def calculate_decoupled_confidence(stt_prob: float, segments: list,
                                   chief_data: dict, corrections: list,
                                   validation_issues: list) -> tuple:
    """Hitung confidence multi-dimensi & identifikasi alasan review klinis (P1)."""
    reasons = []

    # 1. Transcription confidence
    trans_conf = round(max(0.70, min(0.99, stt_prob if stt_prob > 0 else 0.90)), 2)

    # 2. Speaker confidence
    spk_confs = [s.get("confidence", 0.85) for s in segments] if segments else [0.85]
    speaker_conf = round(sum(spk_confs) / len(spk_confs), 2)
    if any(s.get("speaker_role") == "unknown" for s in segments):
        reasons.append("Identitas pembicara pada beberapa segmen belum terkonfirmasi pasti")

    # 3. Clinical extraction confidence
    if chief_data.get("value"):
        clin_conf = 0.94 if not chief_data.get("review_required") else 0.82
    else:
        clin_conf = 0.50
        reasons.append("Keluhan utama tidak ditemukan dalam tuturan pasien")

    # 4. Normalization confidence
    if corrections:
        norm_conf = round(sum(c.get("confidence", 0.90) for c in corrections) / len(corrections), 2)
        uncertain_corrections = [c for c in corrections if c.get("review_required")]
        if uncertain_corrections:
            reasons.append(f"Terdapat {len(uncertain_corrections)} kata yang dikoreksi fonetik dan memerlukan tinjauan")
    else:
        norm_conf = 0.98

    for iss in validation_issues:
        reasons.append(iss["reason"])

    review_required = len(reasons) > 0 or not chief_data.get("value")

    confidence_dict = {
        "transcription": trans_conf,
        "speaker": speaker_conf,
        "clinical_extraction": clin_conf,
        "normalization": norm_conf
    }

    review_dict = {
        "required": review_required,
        "reasons": reasons
    }

    return confidence_dict, review_dict


def _confidence_score(chief: str, chief_pts: int, dur_pts: int,
                      sec_count: int, word_count: int) -> dict:
    """Legacy confidence scoring untuk backward compatibility."""
    warnings = []
    score = 0
    if chief:
        score += chief_pts * 30
    else:
        warnings.append("Keluhan utama tidak terdeteksi — pastikan pasien menyebutkan gejalanya")

    score += dur_pts * 15
    score += min(sec_count * 10, 20)

    if word_count < 6:
        score -= 25
        warnings.append("Transcript terlalu pendek — coba rekam lebih lama")
    elif word_count > 30:
        score += 10

    score = max(0, min(100, score))
    level = "TINGGI" if score >= 70 else "SEDANG" if score >= 40 else "RENDAH"
    if level == "RENDAH" and chief:
        warnings.append("Confidence rendah — coba rekam ulang dengan lebih jelas")

    return {"level": level, "score": score, "warnings": warnings}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTOR CLASS (Unified Hospital-Grade Architecture)
# ══════════════════════════════════════════════════════════════════════════════

class MedicalComplaintExtractor:
    """
    Pipeline Klinis Lengkap: STT → Koreksi Terlacak → Diarization Multi-Role →
    Ekstraksi Terstruktur → Validasi Klinis → Draf EMR Siap Tinjau Tenaga Medis.
    """

    def extract(self, transcript: str, segments: list = None,
                audio_meta: dict = None) -> dict:
        if not transcript or not transcript.strip():
            return self._empty()

        meta = audio_meta or {}
        stt_prob = meta.get("language_probability", 0.95)

        # ── 1. Koreksi Terlacak & Normalisasi (P0) ────────────────────────
        corr_res = correct_with_trace(transcript)
        original_text = corr_res["original_text"]
        corrected_text = corr_res["corrected_text"]
        corrections = corr_res["corrections"]

        # ── 2. Speaker Diarization Multi-Role (P0) ────────────────────────
        labeled_segments = _label_speakers(segments, fallback_text=corrected_text)

        # ── 3. Ekstraksi Klinis Terstruktur (P1) ──────────────────────────
        struct_chief = _extract_structured_chief_complaint(corrected_text, labeled_segments=labeled_segments)
        chief_str = struct_chief["value"] or ""
        chief_pts = 2 if chief_str else 0

        duration_dict = _extract_structured_duration(corrected_text, labeled_segments=labeled_segments)
        duration_str = duration_dict["original_text"] if duration_dict else ""
        dur_pts = 2 if duration_str else 0

        struct_symptoms = _extract_structured_symptoms(corrected_text, chief_str, labeled_segments=labeled_segments)
        secondary_list = _extract_secondary(corrected_text, chief_str, labeled_segments=labeled_segments)

        extended_vitals = _extract_extended_vitals(corrected_text, labeled_segments=labeled_segments)
        legacy_vitals = _extract_vitals(corrected_text)

        standardized_allergies = _extract_standardized_allergies(corrected_text, labeled_segments=labeled_segments)
        legacy_allergies = _extract_allergies(corrected_text, labeled_segments=labeled_segments)

        med_history, meds_mentioned, procs_mentioned, uncat = _extract_ancillary_entities(corrected_text)

        # Susun catatan klinis teks untuk keluhan tambahan EMR
        structured_secondary_text = list(secondary_list)
        if legacy_vitals.get("tensi"):
            structured_secondary_text.append(f"Tensi: {legacy_vitals['tensi']}")
        if legacy_vitals.get("suhu"):
            structured_secondary_text.append(f"Suhu: {legacy_vitals['suhu']}")
        if legacy_vitals.get("spo2"):
            structured_secondary_text.append(f"SpO2: {legacy_vitals['spo2']}")
        if legacy_vitals.get("nadi"):
            structured_secondary_text.append(f"Nadi: {legacy_vitals['nadi']}")
        if standardized_allergies.get("status") == "reported" and standardized_allergies.get("items"):
            structured_secondary_text.append(f"Alergi: {', '.join(standardized_allergies['items'])}")
        elif standardized_allergies.get("status") == "denied":
            structured_secondary_text.append("Alergi: Disangkal")

        # ── 4. Validasi Klinis & Konsistensi Data (P0 / P1) ────────────────
        partial_emr = {
            "original_text": original_text,
            "corrected_text": corrected_text,
            "chief_complaint": struct_chief,
            "symptoms": struct_symptoms,
            "duration": duration_dict,
            "vitals": extended_vitals,
            "allergies": standardized_allergies,
        }
        val_res = ClinicalDataValidator.validate(partial_emr)

        # ── 5. Confidence Scoring Multi-Dimensi (P1) ──────────────────────
        legacy_conf = _confidence_score(
            chief_str, chief_pts, dur_pts, len(secondary_list), len(transcript.split())
        )
        decoupled_conf, review_info = calculate_decoupled_confidence(
            stt_prob, labeled_segments, struct_chief, corrections, val_res["issues"]
        )
        decoupled_conf["level"] = legacy_conf.get("level", "SEDANG")
        decoupled_conf["score"] = legacy_conf.get("score", 70)
        decoupled_conf["warnings"] = legacy_conf.get("warnings", [])

        # Status alur persetujuan tenaga medis
        review_status = "needs_review" if (review_info["required"] or not val_res["valid"]) else "draft"

        session_id = meta.get("session_id") or f"EMR-{int(datetime.datetime.now().timestamp())}"

        return {
            # ── Struktur Utama EMR Konsisten (Section 10) ─────────────────
            "session_metadata": {
                "session_id": session_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "audio_file": meta.get("audio_file", ""),
                "audio_duration": meta.get("audio_duration", 0.0),
                "language": meta.get("detected_language", "id"),
            },
            "original_text": original_text,
            "corrected_text": corrected_text,
            "corrections": corrections,
            "segments": labeled_segments,
            "chief_complaint": struct_chief,
            "symptoms": struct_symptoms,
            "duration": duration_dict,
            "vitals": extended_vitals,
            "allergies": standardized_allergies,
            "medical_history": med_history,
            "medications_mentioned": meds_mentioned,
            "procedures_mentioned": procs_mentioned,
            "uncategorized": uncat,
            "confidence": decoupled_conf,
            "review": review_info,
            "clinical_validation": val_res,
            "review_status": review_status,

            # ── Legacy Keys (100% Backward Compatibility) ─────────────────
            "raw_transcript": original_text,
            "layer1_phonetic": _apply_layer1(transcript),
            "cleaned_transcript": corrected_text,
            "labeled_segments": labeled_segments,
            "keluhan_utama": chief_str,
            "keluhan_tambahan": ", ".join(structured_secondary_text) if structured_secondary_text else "",
            "lama_sakit": duration_str,
            "fields": {
                "keluhan_utama": chief_str,
                "keluhan_tambahan": ", ".join(structured_secondary_text) if structured_secondary_text else "",
                "lama_sakit": duration_str,
            },
            "clinical_entities": {
                "vitals": legacy_vitals,
                "extended_vitals": extended_vitals,
                "allergies": legacy_allergies,
                "standardized_allergies": standardized_allergies,
                "patient_turns": len([s for s in labeled_segments if s.get("speaker_role") in ["patient", "companion"] or s.get("speaker") == "Pasien"]),
                "doctor_turns": len([s for s in labeled_segments if s.get("speaker_role") == "doctor" or s.get("speaker") == "Dokter"]),
            },
            "pipeline_layers": {
                "L1_phonetic": _apply_layer1(transcript),
                "L2_indonesia": corrected_text,
                "L3_speakers": len(labeled_segments),
                "L4_medical": {
                    "chief": chief_str,
                    "secondary": secondary_list,
                    "structured_secondary": structured_secondary_text,
                    "duration": duration_str,
                    "vitals": legacy_vitals,
                    "allergies": legacy_allergies,
                }
            },
            "legacy_confidence": legacy_conf,
            "ml_predicted_class": "KLINIS" if chief_str else "KOSONG",
        }

    def _empty(self) -> dict:
        return {
            "session_metadata": {
                "session_id": f"EMR-EMPTY",
                "timestamp": datetime.datetime.now().isoformat(),
                "audio_file": "",
                "audio_duration": 0.0,
                "language": "id"
            },
            "original_text": "",
            "corrected_text": "",
            "corrections": [],
            "segments": [],
            "chief_complaint": {
                "value": None, "body_part": None, "onset": None,
                "duration": None, "severity": None, "source": None,
                "confidence": None, "review_required": True
            },
            "symptoms": [],
            "duration": None,
            "vitals": {
                "blood_pressure": None, "temperature": None, "spo2": None,
                "pulse": None, "respiratory_rate": None, "weight": None,
                "height": None, "pain_scale": None
            },
            "allergies": {
                "status": "not_asked", "items": [], "reactions": [],
                "source": None, "review_required": False, "category": None
            },
            "medical_history": [],
            "medications_mentioned": [],
            "procedures_mentioned": [],
            "uncategorized": [],
            "confidence": {
                "transcription": 0.0, "speaker": 0.0,
                "clinical_extraction": 0.0, "normalization": 0.0,
                "level": "RENDAH", "score": 0,
                "warnings": ["Tidak ada audio / transcript kosong"]
            },
            "review": {
                "required": True, "reasons": ["Transcript kosong atau tidak ada audio."]
            },
            "clinical_validation": {
                "valid": False, "issues": [{"field": "audio", "value": None, "reason": "Audio kosong", "severity": "error", "action": "Rekam audio"}],
                "total_issues": 1
            },
            "review_status": "failed",

            # Legacy fields
            "raw_transcript": "",
            "layer1_phonetic": "",
            "cleaned_transcript": "",
            "labeled_segments": [],
            "keluhan_utama": "",
            "keluhan_tambahan": "",
            "lama_sakit": "",
            "fields": {
                "keluhan_utama": "",
                "keluhan_tambahan": "",
                "lama_sakit": "",
            },
            "clinical_entities": {
                "vitals": {}, "allergies": {}, "patient_turns": 0, "doctor_turns": 0
            },
            "pipeline_layers": {},
            "ml_predicted_class": "KOSONG",
        }

