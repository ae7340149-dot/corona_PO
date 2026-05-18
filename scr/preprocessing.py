import librosa
import numpy as np
import torch
import os
import subprocess
import tempfile
import imageio_ffmpeg as im_ffm
import warnings

warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 22050, duration: int = 10):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = self.sample_rate * self.duration
        self.ffmpeg_path = im_ffm.get_ffmpeg_exe()

    def process(self, file_path: str) -> torch.Tensor:
        tmp_wav = None
        try:
            # Создаем временный идеальный WAV через FFmpeg
            fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            cmd = [
                self.ffmpeg_path, '-y', '-i', str(file_path),
                '-ac', '1', '-ar', str(self.sample_rate),
                '-t', str(self.duration), tmp_wav
            ]
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo, check=True)
            
            y, _ = librosa.load(tmp_wav, sr=self.sample_rate, mono=True)
            y_padded = librosa.util.fix_length(data=y, size=self.target_length)
            
            mel_spec = librosa.feature.melspectrogram(y=y_padded, sr=self.sample_rate, n_mels=64, fmax=8000)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Нормализация 0-1
            m_min, m_max = mel_db.min(), mel_db.max()
            spec_norm = (mel_db - m_min) / (m_max - m_min) if (m_max - m_min) > 0 else mel_db
            
            return torch.from_numpy(spec_norm).float().unsqueeze(0)
            
        except Exception:
            return None
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)