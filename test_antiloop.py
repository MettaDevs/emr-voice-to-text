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

# Prompt bersih tanpa angka
prompt = "Percakapan dokter dan pasien di klinik rumah sakit. Istilah medis: rontgen, test, check up, keluhan, sakit, kemarin."

segs, info = model.transcribe(
    audio,
    language="id",
    initial_prompt=prompt,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    compression_ratio_threshold=2.2,
    condition_on_previous_text=False,
    beam_size=5,
    vad_filter=True
)

print("HASIL TRANSKRIPSI ANTI-LOOP:")
for s in segs:
    print(f"- {s.text}")
