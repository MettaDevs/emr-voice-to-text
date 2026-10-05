import asyncio
import os
import edge_tts

SAMPLES = [
    {
        "filename": "sample_1_kepala_pipi.mp3",
        "title": "Kasus 1 (Contoh dari Atasan): Sakit Kepala + Pipi",
        "doctor_text": "Selamat pagi pak, keluhannya apa?",
        "patient_text": "Eee kepala saya sakit, terus di atas pipi saya sakit.",
    },
    {
        "filename": "sample_2_perut_meriang.mp3",
        "title": "Kasus 2: Sakit Perut + Meriang & Mual",
        "doctor_text": "Ada keluhan apa bu hari ini?",
        "patient_text": "Anu dok, perut saya melilit dari kemarin malam, mmm terus badan agak meriang dan mual.",
    },
    {
        "filename": "sample_3_batuk_sesak.mp3",
        "title": "Kasus 3: Batuk Kering + Tenggorokan Gatal",
        "doctor_text": "Bisa diceritakan keluhan utamanya apa?",
        "patient_text": "Eee ini dok, batuk kering sudah tiga hari, terus tenggorokan rasanya gatal dan agak sesak.",
    }
]

async def create_dialogue_audio(doctor_text, patient_text, output_file):
    doc_audio = output_file.replace(".mp3", "_doc.mp3")
    pat_audio = output_file.replace(".mp3", "_pat.mp3")
    
    # Dokter menggunakan suara pria (id-ID-ArdiNeural)
    doc_communicate = edge_tts.Communicate(doctor_text, "id-ID-ArdiNeural", rate="+5%")
    await doc_communicate.save(doc_audio)
    
    # Pasien menggunakan suara wanita (id-ID-GadisNeural)
    pat_communicate = edge_tts.Communicate(patient_text, "id-ID-GadisNeural", rate="-5%")
    await pat_communicate.save(pat_audio)
    
    # Gabungkan file MP3
    with open(output_file, "wb") as outfile:
        with open(doc_audio, "rb") as f1:
            outfile.write(f1.read())
        with open(pat_audio, "rb") as f2:
            outfile.write(f2.read())
            
    # Bersihkan file temporer
    if os.path.exists(doc_audio):
        os.remove(doc_audio)
    if os.path.exists(pat_audio):
        os.remove(pat_audio)

async def main():
    target_dir = os.path.join(os.path.dirname(__file__), "audio_samples")
    os.makedirs(target_dir, exist_ok=True)
    print("Membuat contoh audio percakapan dokter-pasien via Edge-TTS (Bahasa Indonesia)...")
    for s in SAMPLES:
        path = os.path.join(target_dir, s["filename"])
        print(f"-> Generating {s['filename']} ({s['title']})...")
        await create_dialogue_audio(s["doctor_text"], s["patient_text"], path)
    print("Semua contoh audio berhasil dibuat di:", target_dir)

if __name__ == "__main__":
    asyncio.run(main())
