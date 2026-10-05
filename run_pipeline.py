import os
import time
from transcriber import SpeechTranscriber
from medical_extractor import MedicalComplaintExtractor

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_e2e_pipeline():
    print("=" * 80)
    print("      PROTOTYPE VOICE-TO-EMR (SPEECH-TO-TEXT + INFORMATION EXTRACTION)")
    print("      Model STT: Faster-Whisper | Layer 2: Noise Filter & Clinical Splitter")
    print("=" * 80)

    # 1. Inisialisasi Model
    transcriber = SpeechTranscriber(model_size="base", device="cpu")
    extractor = MedicalComplaintExtractor()

    samples_dir = os.path.join(os.path.dirname(__file__), "audio_samples")
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith(".mp3")]

    print(f"\nDitemukan {len(sample_files)} contoh file audio percakapan klinis.")

    for idx, audio_file in enumerate(sample_files, 1):
        audio_path = os.path.join(samples_dir, audio_file)
        print("\n" + "-" * 80)
        print(f"CASE {idx}: {audio_file}")
        print("-" * 80)

        # Langkah 1: Transkripsi Suara (Speech-to-Text)
        start_time = time.time()
        print("[Tahap 1] Mentranskripsi suara dengan Faster-Whisper...")
        stt_result = transcriber.transcribe(audio_path)
        raw_text = stt_result["text"]
        elapsed = round(time.time() - start_time, 2)

        print(f"  -> Durasi Audio : {stt_result['duration']} detik")
        print(f"  -> Waktu Proses : {elapsed} detik")
        print(f"  -> Transkrip Asli (Mentah): \"{raw_text}\"")

        # Langkah 2: Pemrosesan Layer di Bawah Model (Filter Noise & Split Field)
        print("\n[Tahap 2] Memproses Layer di bawah model (Pembersihan Noise & Ekstraksi)...")
        ext_result = extractor.extract(raw_text)

        print(f"  -> Transkrip Bersih (Tanpa Noise 'eee/eh/anu') : \"{ext_result['cleaned_transcript']}\"")
        print("\n  HASIL PEMISAHAN FIELD FORMULIR REKAM MEDIS:")
        print("  +" + "-" * 76 + "+")
        print(f"  | [KELUHAN UTAMA *]             : {ext_result['keluhan_utama']:<44} |")
        print(f"  | [KELUHAN TAMBAHAN / ANAMNESA] : {ext_result['keluhan_tambahan']:<44} |")
        print("  +" + "-" * 76 + "+")

    print("\n" + "=" * 80)
    print("                     SEMUA KASUS BERHASIL DIPROSES!")
    print("=" * 80)

if __name__ == "__main__":
    run_e2e_pipeline()
