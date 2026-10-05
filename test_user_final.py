from transcriber import SpeechTranscriber
from medical_extractor import MedicalComplaintExtractor

t = SpeechTranscriber()
e = MedicalComplaintExtractor()

print("=== MENGUJI FILE REKAMAN USER TERAKHIR ===")
res_stt = t.transcribe("uploads/rec_1790233487.webm")
print("Transkrip Faster-Whisper :", res_stt["text"])

res_ext = e.extract(res_stt["text"])
print("\n[HASIL FORMULIR REKAM MEDIS]")
print("-> KELUHAN UTAMA             :", res_ext["keluhan_utama"])
print("-> KELUHAN TAMBAHAN/ANAMNESA :", res_ext["keluhan_tambahan"])
print("-> LAMA SAKIT / DURASI       :", res_ext["lama_sakit"])
