import os, sys
venv = os.path.dirname(os.path.dirname(sys.executable))
for p in ['cublas', 'cudnn', 'cuda_nvrtc']:
    p_path = os.path.join(venv, 'Lib', 'site-packages', 'nvidia', p, 'bin')
    if os.path.exists(p_path):
        os.add_dll_directory(p_path)
        os.environ['PATH'] = p_path + ';' + os.environ['PATH']

from faster_whisper import WhisperModel
model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
audio = 'uploads/rec_1790233487.webm'

print("=== TEST 1: Strict Indonesian with Medical Symptom Priors ===")
prompt_id = "Percakapan konsultasi dokter dan pasien di klinik. Pasien mengeluh pilek, batuk, flu, demam, badan anget, meriang, tidak enak badan sejak kemarin malam."
segs, info = model.transcribe(
    audio,
    language="id",
    initial_prompt=prompt_id,
    beam_size=5,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

print("Language:", info.language)
text = " ".join([s.text for s in segs])
print("Transkripsi:", text)
