import numpy as np
import librosa
from typing import List, Dict, Tuple, Any


class AudioFeatureExtractor:
    """Класс для извлечения аудио-признаков из видеофайлов"""
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 13, frame_length: int = 2048, 
                hop_length: int = 512):
        """
        Инициализация извлечения аудио-признаков
        
        Args:
            sr: частота дискретизации аудио
            n_mfcc: количество MFCC коэффициентов
            frame_length: длина окна для FFT
            hop_length: шаг между окнами
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        
    def load_audio(self, video_path: str) -> np.ndarray:
        """
        Загрузка аудио из видеофайла
        
        Args:
            video_path: путь к видеофайлу
            
        Returns:
            numpy array с аудиоданными
        """
        try:
            # Извлечение аудио из видео
            y, _ = librosa.load(video_path, sr=self.sr)
            return y
        except Exception as e:
            print(f"Ошибка при загрузке аудио: {e}")
            return None
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Извлечение MFCC коэффициентов
        
        Args:
            audio: аудиоданные
            
        Returns:
            numpy array с MFCC коэффициентами
        """
        if audio is None or len(audio) == 0:
            return None
            
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """
        Извлечение спектрального контраста
        
        Args:
            audio: аудиоданные
            
        Returns:
            numpy array со спектральным контрастом
        """
        if audio is None or len(audio) == 0:
            return None
            
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return contrast
    
    def extract_volume_envelope(self, audio: np.ndarray, frame_size: int = 1024) -> np.ndarray:
        """
        Извлечение огибающей громкости
        
        Args:
            audio: аудиоданные
            frame_size: размер кадра для вычисления громкости
            
        Returns:
            numpy array с огибающей громкости
        """
        if audio is None or len(audio) == 0:
            return None
            
        # Вычисление среднеквадратичной энергии в каждом фрейме
        hop_length = frame_size // 2
        rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_length)[0]
        return rms
    
    def extract_audio_fingerprint(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Извлечение всех аудио-признаков для создания "отпечатка"
        
        Args:
            audio: аудиоданные
            
        Returns:
            словарь с различными аудио-признаками
        """
        if audio is None or len(audio) == 0:
            return {}
            
        features = {
            'mfcc': self.extract_mfcc(audio),
            'contrast': self.extract_spectral_contrast(audio),
            'volume': self.extract_volume_envelope(audio)
        }
        
        return features
    
    def extract_audio_features_at_time(self, audio: np.ndarray, time_point: float, 
                                     window_size: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Извлечение аудио-признаков для заданного момента времени
        
        Args:
            audio: аудиоданные
            time_point: временная метка в секундах
            window_size: размер окна в секундах
            
        Returns:
            словарь с аудио-признаками для заданного момента
        """
        if audio is None or len(audio) == 0:
            return {}
            
        # Вычисление индексов начала и конца для заданного времени
        start_idx = int(time_point * self.sr)
        end_idx = min(start_idx + int(window_size * self.sr), len(audio))
        
        # Получение фрагмента аудио
        audio_segment = audio[start_idx:end_idx]
        
        # Если фрагмент слишком короткий, дополним его нулями
        if len(audio_segment) < window_size * self.sr:
            audio_segment = np.pad(audio_segment, 
                                  (0, int(window_size * self.sr) - len(audio_segment)))
        
        # Извлечение признаков из фрагмента
        return self.extract_audio_fingerprint(audio_segment) 