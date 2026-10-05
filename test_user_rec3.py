import os, sys
venv = os.path.dirname(os.path.dirname(sys.executable))
for p in ['cublas', 'cudnn', 'cuda_nvrtc']:
    p_path = os.path.join(venv, 'Lib', 'site-packages', 'nvidia', p, 'bin')
    if os.path.exists(p_path):
        os.add_dll_directory(p_path)
        os.environ['PATH'] = p_path + ';' + os.environ['PATH']

from faster_whisper import WhisperModel
model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
audio = 'uploads/rec_1790231747.webm'

print("--- UJI 1: TANPA INITIAL PROMPT ---")
segs, info = model.transcribe(audio, beam_size=5, vad_filter=True)
for s in segs:
    print(f"[{s.start:.2f}s - {s.end:.2f}s]: {s.text}")

print("\n--- UJI 2: DENGAN PROMPT MEDIS BERSIH (TANPA ANGKA) ---")
clean_prompt = "Konsultasi medis dokter dan pasien. Istilah: rontgen, keluhan, sakit, kemarin, test."
segs, info = model.transcribe(audio, initial_prompt=clean_prompt, beam_size=5, vad_filter=True)
for s in segs:
    print(f"[{s.start:.2f}s - {s.end:.2f}s]: {s.text}")
