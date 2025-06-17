#!/usr/bin/env python
import os
import sys
import argparse
import json
from pathlib import Path

# Добавление директории проекта в sys.path
project_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_dir)

from src.models.intro_detector import IntroDetector
from src.utils.data_loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Детектор коротких заставок в сериалах')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Директория с видео для обработки')
    parser.add_argument('--annotations', type=str, default=None,
                      help='JSON файл с аннотациями заставок (для оценки)')
    parser.add_argument('--output', type=str, default='detected_intros.json',
                      help='Путь для сохранения результатов')
    parser.add_argument('--min_length', type=float, default=2.0,
                      help='Минимальная длительность заставки (сек)')
    parser.add_argument('--max_length', type=float, default=30.0,
                      help='Максимальная длительность заставки (сек)')
    parser.add_argument('--threshold', type=float, default=0.8,
                      help='Порог сходства для определения заставок')
    parser.add_argument('--evaluate', action='store_true',
                      help='Вычислить метрики оценки, если указан файл аннотаций')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Проверка наличия директории с данными
    if not os.path.exists(args.data_dir):
        print(f"Ошибка: директория {args.data_dir} не существует")
        return
    
    # Инициализация загрузчика данных
    data_loader = DataLoader(args.data_dir, args.annotations)
    
    # Инициализация детектора заставок
    detector = IntroDetector(
        min_intro_length=args.min_length,
        max_intro_length=args.max_length,
        similarity_threshold=args.threshold
    )
    
    # Получение списка видео для обработки
    video_files = data_loader.get_video_files()
    
    if not video_files:
        print(f"В директории {args.data_dir} не найдено видеофайлов")
        return
    
    print(f"Найдено {len(video_files)} видеофайлов для обработки")
    
    # Обнаружение заставок
    results = {}
    
    for i, video_path in enumerate(video_files, 1):
        file_name = os.path.basename(video_path)
        print(f"[{i}/{len(video_files)}] Обработка {file_name}...")
        
        try:
            result = detector.detect_intro(video_path)
            results[file_name] = result
            
            if result['found']:
                print(f"  Заставка найдена: {result['start_time']:.2f}с - {result['end_time']:.2f}с "
                     f"(длительность: {result['duration']:.2f}с, уверенность: {result['confidence']:.3f})")
            else:
                print("  Заставка не обнаружена")
        except Exception as e:
            print(f"  Ошибка при обработке файла: {e}")
    
    # Сохранение результатов
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Результаты сохранены в {args.output}")
    
    # Оценка результатов
    if args.evaluate and args.annotations:
        metrics = data_loader.evaluate_predictions(results)
        print("\nРезультаты оценки:")
        print(f"  Точность: {metrics['accuracy']:.4f}")
        print(f"  Средний IoU: {metrics['mean_iou']:.4f}")
        print(f"  Правильно определено: {metrics['correct']}/{metrics['total']}")
    

if __name__ == "__main__":
    main() 