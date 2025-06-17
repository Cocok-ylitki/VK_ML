import numpy as np
import cv2
from typing import List, Dict, Tuple, Any
from ..utils.video_processor import VideoProcessor


class FingerprintExtractor:
    """Класс для извлечения "отпечатков" из видео для поиска заставок"""
    
    def __init__(self, video_path: str, sampling_rate: float = 0.5):
        """
        Инициализация экстрактора признаков
        
        Args:
            video_path: путь к видеофайлу
            sampling_rate: частота сэмплирования кадров (кадров в секунду)
        """
        self.video_path = video_path
        self.sampling_rate = sampling_rate
        self.video_processor = VideoProcessor(video_path)
        
    def extract_fingerprints(self) -> Dict[float, np.ndarray]:
        """
        Извлечение отпечатков для всего видео
        
        Returns:
            словарь {время: отпечаток}
        """
        fingerprints = {}
        duration = self.video_processor.duration
        current_time = 0
        
        while current_time < duration:
            frame = self.video_processor.extract_frame(current_time)
            if frame is not None:
                fingerprint = self.video_processor.get_frame_fingerprint(frame)
                fingerprints[current_time] = fingerprint
            current_time += 1 / self.sampling_rate
            
        return fingerprints
    
    def compute_similarity_matrix(self, fingerprints: Dict[float, np.ndarray]) -> Tuple[np.ndarray, List[float]]:
        """
        Вычисление матрицы сходства между отпечатками
        
        Args:
            fingerprints: словарь отпечатков {время: отпечаток}
            
        Returns:
            матрица сходства и список временных меток
        """
        times = sorted(fingerprints.keys())
        n = len(times)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Вычисление схожести на основе расстояния Хэмминга
                fp1 = fingerprints[times[i]]
                fp2 = fingerprints[times[j]]
                # Меньшее расстояние = большее сходство
                distance = np.sum(fp1 != fp2)
                similarity = 1.0 - distance / len(fp1)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                
        return similarity_matrix, times
    
    def extract_scene_transitions(self, threshold: float = 0.5) -> List[float]:
        """
        Выявление переходов между сценами
        
        Args:
            threshold: порог для определения смены сцены
            
        Returns:
            список временных меток переходов между сценами
        """
        transitions = []
        duration = self.video_processor.duration
        current_time = 0
        prev_hist = None
        
        while current_time < duration:
            frame = self.video_processor.extract_frame(current_time)
            if frame is not None:
                hist = self.video_processor.compute_frame_histogram(frame)
                
                if prev_hist is not None:
                    # Вычисление сходства гистограмм
                    similarity = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                    if similarity < threshold:
                        transitions.append(current_time)
                
                prev_hist = hist
            
            current_time += 1 / self.sampling_rate
            
        return transitions
    
    def release_resources(self):
        """Освобождение ресурсов"""
        self.video_processor.release() 