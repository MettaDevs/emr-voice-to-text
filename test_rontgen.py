import asyncio, edge_tts, os, sys
async def gen():
    c = edge_tts.Communicate("Dok, kemarin saya sudah rontgen dada di rumah sakit", "id-ID-ArdiNeural")
    await c.save("audio_samples/test_rontgen.mp3")
asyncio.run(gen())

venv = os.path.dirname(os.path.dirname(sys.executable))
for p in ['cublas', 'cudnn', 'cuda_nvrtc']:
    p_path = os.path.join(venv, 'Lib', 'site-packages', 'nvidia', p, 'bin')
    if os.path.exists(p_path):
        os.add_dll_directory(p_path)
        os.environ['PATH'] = p_path + ';' + os.environ['PATH']

from faster_whisper import WhisperModel
model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
segs, info = model.transcribe("audio_samples/test_rontgen.mp3", language="id", beam_size=5)
print("Hasil Transkripsi:", [s.text for s in segs])
