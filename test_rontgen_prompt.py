import os, sys
venv = os.path.dirname(os.path.dirname(sys.executable))
for p in ['cublas', 'cudnn', 'cuda_nvrtc']:
    p_path = os.path.join(venv, 'Lib', 'site-packages', 'nvidia', p, 'bin')
    if os.path.exists(p_path):
        os.add_dll_directory(p_path)
        os.environ['PATH'] = p_path + ';' + os.environ['PATH']

from faster_whisper import WhisperModel
model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
prompt = "Pemeriksaan foto rontgen dada dan rontgen thorax di rumah sakit."
segs, info = model.transcribe("audio_samples/test_rontgen.mp3", language="id", initial_prompt=prompt, beam_size=5)
print("Hasil Transkripsi dengan Prompt:", [s.text for s in segs])
