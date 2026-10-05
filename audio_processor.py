"""
audio_processor.py — Hospital Voice-to-EMR Audio Quality & Processing Engine
=============================================================================
Menangani inspeksi kualitas audio, konversi format, kalkulasi RMS (dBFS),
deteksi hening / mic mute, deteksi volume rendah dengan adaptive software gain boost,
deteksi audio clipping, dan standardisasi ke 16kHz mono PCM untuk Whisper.
"""

import os
import math
import numpy as np
import av
import wave
import scipy.signal
from typing import Dict, Any, Tuple, Optional


class AudioQualityStatus:
    OPTIMAL = "OPTIMAL"
    LOW_VOLUME = "LOW_VOLUME"
    VERY_LOW_VOLUME = "VERY_LOW_VOLUME"
    SILENT = "SILENT"
    CLIPPING = "CLIPPING"
    CORRUPTED = "CORRUPTED"
    EMPTY = "EMPTY"


def inspect_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Inspeksi menyeluruh kualitas dan format file audio sebelum STT.
    Menghitung durasi, sample rate, channel, RMS dBFS, peak amplitude, dan status kualitas.
    """
    if not os.path.exists(file_path):
        return {
            "valid": False,
            "status": AudioQualityStatus.CORRUPTED,
            "error": f"File tidak ditemukan: {file_path}",
            "file_size_bytes": 0,
            "duration_seconds": 0.0,
            "rms_dbfs": -99.0,
            "peak_amplitude": 0.0,
            "warning": "File audio tidak dapat diakses."
        }

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return {
            "valid": False,
            "status": AudioQualityStatus.EMPTY,
            "error": "File audio berukuran 0 byte (kosong).",
            "file_size_bytes": 0,
            "duration_seconds": 0.0,
            "rms_dbfs": -99.0,
            "peak_amplitude": 0.0,
            "warning": "Mikrofon tidak mengirimkan data audio sama sekali."
        }

    try:
        container = av.open(file_path)
        if not container.streams.audio:
            container.close()
            return {
                "valid": False,
                "status": AudioQualityStatus.CORRUPTED,
                "error": "Tidak ditemukan audio stream dalam container media.",
                "file_size_bytes": file_size,
                "duration_seconds": 0.0,
                "rms_dbfs": -99.0,
                "peak_amplitude": 0.0,
                "warning": "Format file tidak memiliki stream audio yang valid."
            }

        stream = container.streams.audio[0]
        codec_name = stream.codec_context.name
        source_rate = stream.rate or 16000
        source_channels = stream.channels or 1

        # Resample ke 16000 Hz Mono Float32
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        frames_data = []

        for frame in container.decode(stream):
            for resampled_frame in resampler.resample(frame):
                frames_data.append(resampled_frame.to_ndarray())

        container.close()

        if not frames_data:
            return {
                "valid": False,
                "status": AudioQualityStatus.EMPTY,
                "error": "Audio stream tidak menghasilkan frame data.",
                "file_size_bytes": file_size,
                "duration_seconds": 0.0,
                "rms_dbfs": -99.0,
                "peak_amplitude": 0.0,
                "warning": "Data audio kosong setelah proses decode."
            }

        audio_np = np.concatenate(frames_data, axis=1)[0].astype(np.float32)
        total_samples = len(audio_np)
        duration_s = round(total_samples / 16000.0, 3)

        # Hitung metrik akustik
        peak = float(np.max(np.abs(audio_np))) if total_samples > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio_np ** 2))) if total_samples > 0 else 0.0
        rms_dbfs = round(float(20 * np.log10(rms + 1e-9)), 2)

        # Klasifikasi status kualitas audio
        status = AudioQualityStatus.OPTIMAL
        warning = None

        if duration_s < 0.25:
            status = AudioQualityStatus.EMPTY
            warning = "Durasi audio terlalu pendek (< 250ms) untuk pengenalan wicara."
        elif rms_dbfs < -50.0 or peak < 0.005:
            status = AudioQualityStatus.SILENT
            warning = "Audio hening / mikrofon ter-mute. Tidak ada sinyal suara manusia yang cukup."
        elif -50.0 <= rms_dbfs < -38.0:
            status = AudioQualityStatus.VERY_LOW_VOLUME
            warning = "Volume suara sangat rendah. Perlu software gain boost agar Whisper dapat menangkap suara."
        elif -38.0 <= rms_dbfs < -28.0:
            status = AudioQualityStatus.LOW_VOLUME
            warning = "Volume suara agak pelan, namun masih dapat diproses."
        elif peak >= 0.985:
            status = AudioQualityStatus.CLIPPING
            warning = "Terjadi audio clipping / distorsi suara karena mikrofon terlalu dekat atau gain berlebih."

        return {
            "valid": status not in [AudioQualityStatus.CORRUPTED, AudioQualityStatus.EMPTY],
            "status": status,
            "file_size_bytes": file_size,
            "duration_seconds": duration_s,
            "source_codec": codec_name,
            "source_sample_rate": source_rate,
            "source_channels": source_channels,
            "target_sample_rate": 16000,
            "target_channels": 1,
            "rms_dbfs": rms_dbfs,
            "peak_amplitude": round(peak, 4),
            "warning": warning
        }

    except Exception as e:
        return {
            "valid": False,
            "status": AudioQualityStatus.CORRUPTED,
            "error": f"Gagal membaca file audio: {str(e)}",
            "file_size_bytes": file_size,
            "duration_seconds": 0.0,
            "rms_dbfs": -99.0,
            "peak_amplitude": 0.0,
            "warning": "Format file tidak kompatibel atau terpotong saat transmisi."
        }


def normalize_and_prepare_audio(file_path: str,
                                target_wav_path: Optional[str] = None,
                                auto_gain: bool = True) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Mendecode file audio ke 16kHz mono float32, melakukan normalisasi adaptif jika volume terlalu rendah,
    dan menyimpan file PCM WAV terstandarisasi jika target_wav_path ditentukan.
    """
    inspection = inspect_audio_file(file_path)
    if not inspection["valid"]:
        return None, inspection

    try:
        container = av.open(file_path)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        frames_data = []

        for frame in container.decode(stream):
            for resampled_frame in resampler.resample(frame):
                frames_data.append(resampled_frame.to_ndarray())
        container.close()

        audio_np = np.concatenate(frames_data, axis=1)[0].astype(np.float32)

        # ── 1. High-Pass Filter (80Hz Butterworth) ──
        # Menghilangkan sub-bass rumble, hembusan napas/pop mikrofon (<70Hz), dan hum listrik 50Hz/60Hz AC
        if len(audio_np) >= 32:
            sos = scipy.signal.butter(2, 80, 'hp', fs=16000, output='sos')
            audio_np = scipy.signal.sosfilt(sos, audio_np).astype(np.float32)

        applied_gain_db = 0.0
        # ── 2. Speech-Aware Adaptive Gain Boost & Loudness Leveling ──
        # Untuk audio dengan suara pelan (pasien lemah / mikrofon berjarak),
        # hitung RMS pada segmen aktif (speech frames) agar leveling tepat sasaran.
        if auto_gain and inspection["status"] in [AudioQualityStatus.LOW_VOLUME, AudioQualityStatus.VERY_LOW_VOLUME]:
            frame_size = 400  # 25ms @ 16kHz
            if len(audio_np) >= frame_size:
                num_frames = len(audio_np) // frame_size
                frames = audio_np[:num_frames * frame_size].reshape(num_frames, frame_size)
                frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
                active_frames = frame_rms[frame_rms > 0.005]  # Segmen di atas noise floor (-46 dBFS)
                speech_rms = float(np.mean(active_frames)) if len(active_frames) > 0 else float(np.sqrt(np.mean(audio_np ** 2)))
            else:
                speech_rms = float(np.sqrt(np.mean(audio_np ** 2)))

            current_peak = float(np.max(np.abs(audio_np)))
            if current_peak > 0.003 and speech_rms > 0.002:
                # Target speech RMS -19 dBFS (~0.112) untuk sensitivitas optimal Whisper mel-filterbank
                target_speech_rms = 0.112
                gain_factor = min(7.5, max(1.0, target_speech_rms / speech_rms))
                # Batasi gain agar peak tidak melampaui 0.92
                if current_peak * gain_factor > 0.92:
                    gain_factor = max(1.0, 0.92 / current_peak)

                audio_np = audio_np * gain_factor
                applied_gain_db = round(float(20 * math.log10(gain_factor)), 2)
            elif current_peak > 0.005:
                target_peak = 0.707
                gain_factor = min(7.5, target_peak / current_peak)
                audio_np = audio_np * gain_factor
                applied_gain_db = round(float(20 * math.log10(gain_factor)), 2)

        # ── 3. Soft Limiter untuk Mencegah Distorsi / Clipping ──
        current_peak = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
        if current_peak > 0.95:
            audio_np = audio_np * (0.95 / current_peak)
        np.clip(audio_np, -0.99, 0.99, out=audio_np)

        inspection["applied_gain_db"] = applied_gain_db
        inspection["samples_count"] = len(audio_np)

        # Simpan ke target 16-bit PCM WAV jika diminta
        if target_wav_path:
            os.makedirs(os.path.dirname(os.path.abspath(target_wav_path)), exist_ok=True)
            # Konversi float32 [-1.0, 1.0] ke int16 [-32768, 32767]
            int16_audio = (audio_np * 32767.0).astype(np.int16)
            with wave.open(target_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(int16_audio.tobytes())
            inspection["normalized_wav_path"] = target_wav_path

        return audio_np, inspection

    except Exception as e:
        inspection["valid"] = False
        inspection["status"] = AudioQualityStatus.CORRUPTED
        inspection["error"] = f"Gagal normalisasi audio: {str(e)}"
        return None, inspection


def save_pcm_wav(audio_data: np.ndarray, output_path: str, sample_rate: int = 16000):
    """Simpan numpy float32 array sebagai standar 16-bit PCM WAV mono."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    clipped = np.clip(audio_data, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def generate_synthetic_tone_wav(output_path: str,
                                duration_s: float = 3.0,
                                freq_hz: float = 440.0,
                                volume_dbfs: float = -20.0,
                                sample_rate: int = 16000,
                                with_pause: bool = False) -> str:
    """
    Menghasilkan file WAV sintetis untuk keperluan testing otomatis (Section 13)
    tanpa menggunakan audio pasien riil.
    """
    total_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, total_samples, endpoint=False)
    amplitude = 10 ** (volume_dbfs / 20.0)

    audio = amplitude * np.sin(2 * np.pi * freq_hz * t)

    # Tambahkan jeda hening di tengah jika diminta (menguji jeda percakapan dokter-pasien)
    if with_pause and duration_s >= 2.0:
        pause_start = int(0.8 * sample_rate)
        pause_end = int(1.6 * sample_rate)  # Jeda 800ms
        audio[pause_start:pause_end] = 0.0

    save_pcm_wav(audio.astype(np.float32), output_path, sample_rate)
    return output_path
