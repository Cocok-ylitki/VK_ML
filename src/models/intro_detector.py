import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from ..features.fingerprint_extractor import FingerprintExtractor
import json
import os


class IntroDetector:
    """Класс для поиска коротких заставок в видео сериалов"""
    
    def __init__(self, min_intro_length: float = 2.0, max_intro_length: float = 30.0,
                 similarity_threshold: float = 0.8):
        """
        Инициализация детектора заставок
        
        Args:
            min_intro_length: минимальная длительность заставки (сек)
            max_intro_length: максимальная длительность заставки (сек)
            similarity_threshold: порог сходства для определения повторяющихся фрагментов
        """
        self.min_intro_length = min_intro_length
        self.max_intro_length = max_intro_length
        self.similarity_threshold = similarity_threshold
        
    def find_intro_candidates(self, similarity_matrix: np.ndarray, times: List[float]) -> List[Dict[str, Any]]:
        """
        Поиск кандидатов на заставки по матрице сходства
        
        Args:
            similarity_matrix: матрица сходства между кадрами
            times: список временных меток для кадров
            
        Returns:
            список кандидатов на заставки с временами начала и конца
        """
        n = len(times)
        candidates = []
        
        # Поиск диагональных паттернов высокого сходства в матрице
        for i in range(n):
            for j in range(i + int(self.min_intro_length), n):
                # Проверка, что длительность не превышает максимальную
                if times[j] - times[i] > self.max_intro_length:
                    continue
                
                # Проверка сходства последовательности кадров
                sequence_similarity = np.mean(
                    [similarity_matrix[i+k, j+k] for k in range(min(20, n-j))]
                )
                
                if sequence_similarity > self.similarity_threshold:
                    # Определение конца последовательности
                    end_offset = 0
                    for k in range(min(int(self.max_intro_length), n-j)):
                        if similarity_matrix[i+k, j+k] > self.similarity_threshold:
                            end_offset = k
                        else:
                            break
                    
                    duration = times[i + end_offset] - times[i]
                    if self.min_intro_length <= duration <= self.max_intro_length:
                        candidates.append({
                            'start_time': times[i],
                            'end_time': times[i + end_offset],
                            'duration': duration,
                            'similarity_score': sequence_similarity
                        })
        
        # Сортировка по оценке сходства
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Удаление перекрывающихся кандидатов
        filtered_candidates = []
        for c in candidates:
            overlap = False
            for fc in filtered_candidates:
                # Проверка перекрытия
                if (c['start_time'] < fc['end_time'] and 
                    c['end_time'] > fc['start_time']):
                    overlap = True
                    break
            
            if not overlap:
                filtered_candidates.append(c)
                
                # Ограничиваем количество кандидатов
                if len(filtered_candidates) >= 5:
                    break
                    
        return filtered_candidates
    
    def detect_intro(self, video_path: str) -> Dict[str, Any]:
        """
        Обнаружение заставки в видео
        
        Args:
            video_path: путь к видеофайлу
            
        Returns:
            информация о найденной заставке
        """
        # Извлечение отпечатков
        extractor = FingerprintExtractor(video_path)
        fingerprints = extractor.extract_fingerprints()
        
        # Вычисление матрицы сходства
        similarity_matrix, times = extractor.compute_similarity_matrix(fingerprints)
        
        # Поиск кандидатов
        candidates = self.find_intro_candidates(similarity_matrix, times)
        
        # Освобождение ресурсов
        extractor.release_resources()
        
        if not candidates:
            return {'found': False}
        
        # Выбор лучшего кандидата
        best_candidate = candidates[0]
        
        return {
            'found': True,
            'start_time': best_candidate['start_time'],
            'end_time': best_candidate['end_time'],
            'duration': best_candidate['duration'],
            'confidence': best_candidate['similarity_score']
        }
    
    def batch_detect_intros(self, video_directory: str, output_file: str = 'detected_intros.json') -> Dict[str, Dict[str, Any]]:
        """
        Пакетное обнаружение заставок в директории с видео
        
        Args:
            video_directory: директория с видеофайлами
            output_file: путь для сохранения результатов
            
        Returns:
            словарь с результатами детекции для каждого файла
        """
        results = {}
        
        for filename in os.listdir(video_directory):
            if filename.endswith(('.mp4', '.avi', '.mkv')):
                video_path = os.path.join(video_directory, filename)
                print(f"Обработка {filename}...")
                
                result = self.detect_intro(video_path)
                results[filename] = result
        
        # Сохранение результатов
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        return results 