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

prompt = "Percakapan dokter dan pasien di rumah sakit. Bilingual Indonesian English medical consultation. Test, check up, rontgen, I go to school, kemarin."

segs, info = model.transcribe(
    audio,
    language=None,
    initial_prompt=prompt,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    compression_ratio_threshold=2.2,
    condition_on_previous_text=False,
    beam_size=5,
    vad_filter=True
)

print("HASIL (language=None):", info.language)
for s in segs:
    print(f"- {s.text}")
