import cv2
import numpy as np
import os
from typing import Tuple, List, Dict, Any


class VideoProcessor:
    """Класс для обработки видео и извлечения кадров и аудио"""
    
    def __init__(self, video_path: str):
        """
        Инициализация обработчика видео
        
        Args:
            video_path: путь к видеофайлу
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
    
    def extract_frame(self, time_point: float) -> np.ndarray:
        """
        Извлечение кадра по временной метке
        
        Args:
            time_point: время в секундах
            
        Returns:
            numpy array с изображением
        """
        frame_number = int(time_point * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def extract_frames_interval(self, start_time: float, end_time: float, 
                               step: float = 0.5) -> List[np.ndarray]:
        """
        Извлечение последовательности кадров за интервал времени
        
        Args:
            start_time: время начала (сек)
            end_time: время конца (сек)
            step: шаг между кадрами (сек)
            
        Returns:
            список numpy arrays с изображениями
        """
        frames = []
        current_time = start_time
        
        while current_time <= end_time:
            frame = self.extract_frame(current_time)
            if frame is not None:
                frames.append(frame)
            current_time += step
            
        return frames
    
    def compute_frame_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Вычисление цветовой гистограммы для кадра
        
        Args:
            frame: изображение
            
        Returns:
            гистограмма как numpy array
        """
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def get_frame_fingerprint(self, frame: np.ndarray, hash_size: int = 16) -> np.ndarray:
        """
        Получение перцептивного хеша для кадра
        
        Args:
            frame: изображение
            hash_size: размерность хеша
            
        Returns:
            хеш кадра как numpy array
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, (hash_size, hash_size))
        # Вычисление DCT
        dct = cv2.dct(np.float32(frame_resized))
        dct_low_freq = dct[:8, :8]
        # Вычисление медианного значения
        med = np.median(dct_low_freq)
        # Создание бинарного хеша
        hash_bits = dct_low_freq > med
        return hash_bits.flatten()
    
    def release(self):
        """Освобождение ресурсов"""
        if self.cap:
            self.cap.release() 