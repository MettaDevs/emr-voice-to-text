import os, sys
venv = os.path.dirname(os.path.dirname(sys.executable))
for p in ['cublas', 'cudnn', 'cuda_nvrtc']:
    p_path = os.path.join(venv, 'Lib', 'site-packages', 'nvidia', p, 'bin')
    if os.path.exists(p_path):
        os.add_dll_directory(p_path)
        os.environ['PATH'] = p_path + ';' + os.environ['PATH']

from faster_whisper import WhisperModel
model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
audio = 'uploads/rec_1790232205.webm'

print("--- ANALISIS AUDIO rec_1790232205.webm ---")
segs, info = model.transcribe(audio, beam_size=5)
print("Language Detected:", info.language, f"({info.language_probability:.2f})")
for s in segs:
    print(f"[{s.start:.2f}s - {s.end:.2f}s]: {s.text}")
