import os
import json
from typing import Dict, List, Any, Tuple


class DataLoader:
    """Класс для загрузки и обработки данных для обучения и тестирования"""
    
    def __init__(self, data_dir: str, annotations_file: str = None):
        """
        Инициализация загрузчика данных
        
        Args:
            data_dir: директория с видеофайлами
            annotations_file: путь к файлу с аннотациями заставок (JSON)
        """
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.annotations = {}
        
        if annotations_file and os.path.exists(annotations_file):
            self._load_annotations()
    
    def _load_annotations(self):
        """Загрузка аннотаций из JSON-файла"""
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
    
    def get_video_files(self) -> List[str]:
        """
        Получение списка видеофайлов в директории
        
        Returns:
            список путей к видеофайлам
        """
        video_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv')):
                    video_files.append(os.path.join(root, file))
        return video_files
    
    def get_train_test_split(self, test_size: float = 0.2) -> Tuple[List[str], List[str]]:
        """
        Разделение данных на обучающую и тестовую выборки
        
        Args:
            test_size: доля тестовой выборки
            
        Returns:
            кортеж (обучающая выборка, тестовая выборка)
        """
        video_files = self.get_video_files()
        split_idx = int(len(video_files) * (1 - test_size))
        return video_files[:split_idx], video_files[split_idx:]
    
    def get_annotation(self, video_file: str) -> Dict[str, Any]:
        """
        Получение аннотации для видеофайла
        
        Args:
            video_file: путь к видеофайлу
            
        Returns:
            словарь с аннотацией или пустой словарь, если аннотация не найдена
        """
        file_name = os.path.basename(video_file)
        return self.annotations.get(file_name, {})
    
    def evaluate_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Оценка точности предсказаний
        
        Args:
            predictions: словарь с предсказаниями {имя_файла: {start_time, end_time, ...}}
            
        Returns:
            метрики точности предсказаний
        """
        total = 0
        correct = 0
        iou_scores = []
        
        for file_name, pred in predictions.items():
            if file_name in self.annotations and pred.get('found', False):
                total += 1
                
                # Получение истинных временных меток
                gt_start = self.annotations[file_name].get('start_time', 0)
                gt_end = self.annotations[file_name].get('end_time', 0)
                
                # Получение предсказанных временных меток
                pred_start = pred.get('start_time', 0)
                pred_end = pred.get('end_time', 0)
                
                # Вычисление IoU (Intersection over Union)
                intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
                union = max(pred_end, gt_end) - min(pred_start, gt_start)
                iou = intersection / union if union > 0 else 0
                
                iou_scores.append(iou)
                
                # Считаем предсказание правильным, если IoU > 0.5
                if iou > 0.5:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'correct': correct,
            'total': total
        } 