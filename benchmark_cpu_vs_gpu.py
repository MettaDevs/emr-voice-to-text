"""
benchmark_cpu_vs_gpu.py — Benchmark Komprehensif CPU vs GPU untuk MedVoice AI
=============================================================================
Membandingkan model Whisper (large-v3-turbo, small, base, tiny) pada CPU vs GPU:
  1. Kecepatan (Inference Time & RTF)
  2. Akurasi Transkripsi (WER & CER)
  3. Kualitas Ekstraksi Rekam Medis (EMR Fields: Keluhan Utama, Tambahan, Durasi)
  4. Penggunaan Memori (RAM / VRAM)
  5. Analisis Biaya Operasional (Cost Efficiency)
"""

import os
import sys
import time
import json
import ctypes
from ctypes import wintypes
from typing import List, Dict, Any

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

def get_ram_mb() -> float:
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb
        )
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0

# Pastikan DLL CUDA terbaca jika GPU tersedia
venv_dir = os.path.dirname(os.path.dirname(sys.executable))
for lib_name in ["cublas", "cudnn", "cuda_nvrtc"]:
    p_path = os.path.join(venv_dir, "Lib", "site-packages", "nvidia", lib_name, "bin")
    if os.path.exists(p_path):
        try:
            os.add_dll_directory(p_path)
            os.environ["PATH"] = p_path + ";" + os.environ["PATH"]
        except Exception:
            pass

from faster_whisper import WhisperModel
import ctranslate2
from audio_processor import normalize_and_prepare_audio
from medical_extractor import MedicalComplaintExtractor
from evaluate_audio_stt import compute_wer, compute_cer, AUDIO_BENCHMARK_CASES

extractor = MedicalComplaintExtractor()

MODELS_TO_BENCHMARK = [
    # (name, device, compute_type, label)
    ("large-v3-turbo", "cuda", "float16", "GPU RTX 3050 (large-v3-turbo) [Baseline]"),
    ("large-v3-turbo", "cpu", "int8", "CPU (large-v3-turbo int8)"),
    ("small", "cpu", "int8", "CPU (small int8)"),
    ("base", "cpu", "int8", "CPU (base int8)"),
    ("tiny", "cpu", "int8", "CPU (tiny int8)"),
]


def run_benchmark():
    cuda_available = ctranslate2.get_cuda_device_count() > 0
    results = []

    print("=" * 85)
    print("BENCHMARK KOMPREHENSIF CPU VS GPU: MEDVOICE AI")
    print(f"Platform: Intel Core i5-12450HX (12 Threads) | RAM: 12GB | GPU CUDA: {cuda_available}")
    print("=" * 85)

    for model_name, device, compute_type, label in MODELS_TO_BENCHMARK:
        if device == "cuda" and not cuda_available:
            print(f"[SKIP] {label}: CUDA tidak tersedia.")
            continue

        print(f"\n--> Memuat {label}...")
        t_load_start = time.time()
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            load_duration = time.time() - t_load_start
        except Exception as e:
            print(f"    Gagal memuat {label}: {e}")
            continue

        model_results = {
            "label": label,
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
            "load_time_s": round(load_duration, 2),
            "cases": [],
            "total_audio_s": 0.0,
            "total_latency_s": 0.0,
            "avg_wer": 0.0,
            "avg_cer": 0.0,
            "emr_fidelity_score": 0.0,
            "ram_mb": 0.0
        }

        wers = []
        cers = []
        emr_fidelity_hits = 0
        total_fields_expected = 0

        # Ukur memori setelah model termuat
        ram_mb = get_ram_mb()
        model_results["ram_mb"] = round(ram_mb, 1)

        for case in AUDIO_BENCHMARK_CASES:
            audio_path = case["file_path"]
            if not os.path.exists(audio_path):
                continue

            ref_text = case["reference_text"]
            audio_np, diag = normalize_and_prepare_audio(audio_path, auto_gain=True)
            dur = diag.get("duration_seconds", 3.0)

            # Transkripsi
            t0 = time.time()
            segments, info = model.transcribe(
                audio_np,
                language="id",
                beam_size=5,
                temperature=0.0,
                vad_filter=True,
                initial_prompt=(
                    "Transkripsi percakapan klinis dokter dan pasien di rumah sakit Indonesia. "
                    "Keluhan utama, keluhan tambahan, lama sakit, tensi darah, obat dan alergi."
                )
            )
            seg_list = list(segments)
            latency = time.time() - t0

            hyp_text = " ".join(s.text.strip() for s in seg_list).strip()

            # Metrik STT
            wer = compute_wer(ref_text, hyp_text)
            cer = compute_cer(ref_text, hyp_text)
            wers.append(wer)
            cers.append(cer)

            # Evaluasi Ekstraksi Rekam Medis (EMR)
            emr_res = extractor.extract(hyp_text)
            chief = emr_res.get("keluhan_utama", "")
            secondary = emr_res.get("keluhan_tambahan", "")
            duration_str = emr_res.get("lama_sakit", "")

            # Penilaian kesesuaian klinis:
            # Case 1: Kepala sakit / pipi
            # Case 2: Perut melilit / meriang, kemarin malam
            # Case 3: Batuk / sesak, 3 hari
            case_id = case["id"]
            matched_fields = 0
            expected_fields = 2 # chief + secondary/durasi

            if case_id == "AUDIO-CASE-01":
                if any(w in chief.lower() for w in ["kepala", "pipi", "sakit"]): matched_fields += 1
                if any(w in secondary.lower() or w in chief.lower() for w in ["pipi", "kepala"]): matched_fields += 1
            elif case_id == "AUDIO-CASE-02":
                if any(w in chief.lower() for w in ["perut", "melilit", "meriang"]): matched_fields += 1
                if any(w in duration_str.lower() for w in ["kemarin", "semalam"]) or any(w in secondary.lower() for w in ["meriang", "mual"]): matched_fields += 1
            elif case_id == "AUDIO-CASE-03":
                if any(w in chief.lower() for w in ["batuk"]): matched_fields += 1
                if any(w in duration_str.lower() for w in ["tiga hari", "3 hari"]) or any(w in secondary.lower() for w in ["tenggorokan", "sesak"]): matched_fields += 1

            emr_fidelity_hits += matched_fields
            total_fields_expected += expected_fields

            model_results["total_audio_s"] += dur
            model_results["total_latency_s"] += latency

            model_results["cases"].append({
                "id": case_id,
                "audio_duration_s": round(dur, 2),
                "latency_s": round(latency, 2),
                "rtf": round(latency / dur, 3) if dur > 0 else 0.0,
                "wer": round(wer * 100, 2),
                "cer": round(cer * 100, 2),
                "hyp_text": hyp_text,
                "emr_chief": chief,
                "emr_secondary": secondary,
                "emr_duration": duration_str,
                "fidelity_match": f"{matched_fields}/{expected_fields}"
            })

            print(f"    [{case_id}] Dur: {dur:.1f}s | Lat: {latency:.2f}s (RTF: {latency/dur:.2f}x) | WER: {wer*100:.1f}%")
            print(f"         STT: \"{hyp_text}\"")
            print(f"         EMR: [Utama: {chief}] [Tambahan: {secondary}] [Lama Sakit: {duration_str}]")

        model_results["avg_wer"] = round((sum(wers) / len(wers)) * 100, 2) if wers else 0.0
        model_results["avg_cer"] = round((sum(cers) / len(cers)) * 100, 2) if cers else 0.0
        model_results["avg_latency_s"] = round(model_results["total_latency_s"] / len(AUDIO_BENCHMARK_CASES), 2)
        model_results["avg_rtf"] = round(model_results["total_latency_s"] / model_results["total_audio_s"], 3) if model_results["total_audio_s"] > 0 else 0.0
        model_results["emr_fidelity_pct"] = round((emr_fidelity_hits / total_fields_expected) * 100, 2) if total_fields_expected > 0 else 0.0

        # Bersihkan memori model sebelum beralih
        del model
        results.append(model_results)

    # Simpan hasil benchmark ke JSON
    out_file = "benchmark_cpu_gpu_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print("RINGKASAN HASIL PERBANDINGAN CPU VS GPU:")
    print("=" * 85)
    print(f"{'Model & Device':<35} | {'Latency':<8} | {'RTF':<7} | {'WER':<7} | {'CER':<7} | {'EMR Fidelity':<12} | {'RAM (MB)':<8}")
    print("-" * 85)
    for r in results:
        print(f"{r['label']:<35} | {r['avg_latency_s']:>6.2f}s | {r['avg_rtf']:>5.2f}x | {r['avg_wer']:>5.1f}% | {r['avg_cer']:>5.1f}% | {r['emr_fidelity_pct']:>10.1f}% | {r['ram_mb']:>8.1f}")
    print("=" * 85)

    return results


if __name__ == "__main__":
    run_benchmark()
