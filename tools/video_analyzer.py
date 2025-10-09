# tools/video_analyzer.py
import av
import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Импорт YOLOX
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.datasets import COCO_CLASSES
import torch

class VideoAnalyzer:
    def __init__(self, model_name="yolox-tiny", weights_path="weights/yolox_tiny.pth", device="cpu"):
        """Инициализация анализатора видео с YOLOX"""
        self.model, self.exp = self.setup_yolox_model(model_name, weights_path, device)
        self.device = device
        
    def setup_yolox_model(self, model_name, ckpt_path, device):
        """Загрузка и настройка модели YOLOX"""
        print(f" Загрузка модели {model_name}...")
        
        exp = get_exp(None, model_name)
        model = exp.get_model()
        model.eval()
        
        # Загрузка весов
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            print(f" Веса загружены: {ckpt_path}")
        else:
            print(f" Файл весов не найден: {ckpt_path}")
            return None, None
        
        if device == "gpu" and torch.cuda.is_available():
            model = model.cuda()
            print(" Используется GPU")
        else:
            print(" Используется CPU")
        
        return model, exp

    def preprocess_image(self, img, input_size):
        """Предобработка изображения для YOLOX"""
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
        
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        
        tensor = torch.from_numpy(padded_img).unsqueeze(0)
        if self.device == "gpu" and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        return tensor, r

    def detect_objects(self, img, conf_threshold=0.25):
        """Детекция объектов на изображении"""
        if self.model is None:
            return None, 1.0
            
        input_size = self.exp.test_size
        tensor, ratio = self.preprocess_image(img, input_size)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            outputs = postprocess(outputs, self.exp.num_classes, conf_threshold, 0.45)
        
        return outputs[0], ratio

    def draw_bounding_boxes(self, image, outputs, ratio, conf_threshold=0.25):
        """Отрисовка прямоугольников вокруг объектов"""
        if outputs is None:
            return image
        
        bbox_image = image.copy()
        
        for output in outputs:
            if output[4] > conf_threshold:
                x1, y1, x2, y2 = output[:4]
                cls_id = int(output[5])
                
                # Масштабирование координат
                x1 = int(x1 / ratio)
                y1 = int(y1 / ratio)
                x2 = int(x2 / ratio)
                y2 = int(y2 / ratio)
                
                # Отрисовка прямоугольника
                color = (0, 255, 0)  # Зеленый
                cv2.rectangle(bbox_image, (x1, y1), (x2, y2), color, 2)
                
                # Подпись класса
                label = f"{COCO_CLASSES[cls_id]}: {output[4]:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(bbox_image, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(bbox_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return bbox_image

    def draw_circles(self, image, outputs, ratio, conf_threshold=0.25):
        """Отрисовка окружностей вокруг объектов"""
        if outputs is None:
            return image
        
        circle_image = image.copy()
        
        for output in outputs:
            if output[4] > conf_threshold:
                x1, y1, x2, y2 = output[:4]
                cls_id = int(output[5])
                
                # Вычисление центра
                center_x = int(((x1 + x2) / 2) / ratio)
                center_y = int(((y1 + y2) / 2) / ratio)
                
                # Вычисление радиуса
                width = (x2 - x1) / ratio
                height = (y2 - y1) / ratio
                radius = int(np.sqrt(width**2 + height**2) / 4)
                
                # Отрисовка окружности
                color = (0, 0, 255)  # Красный
                cv2.circle(circle_image, (center_x, center_y), radius, color, 2)
                
                # Подпись класса
                label = f"{COCO_CLASSES[cls_id]}: {output[4]:.2f}"
                cv2.putText(circle_image, label, (center_x - 30, center_y - radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return circle_image

    def analyze_video(self, input_video, output_dir="YOLOX_outputs", 
                     extract_frames=None, create_video=True, 
                     visualization="both", conf_threshold=0.25):
        """
        Полный анализ видео
        
        Args:
            input_video: путь к входному видео
            output_dir: папка для результатов
            extract_frames: список кадров для сохранения [100, 200, 300]
            create_video: создавать ли обработанное видео
            visualization: "bbox", "circle", "both"
            conf_threshold: порог уверенности
        """
        
        if self.model is None:
            print("  Модель не загружена!")
            return
        
        # Создаем папки для результатов
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Открываем видео
        container = av.open(input_video)
        video_stream = container.streams.video[0]
        
        # Параметры видео
        width = video_stream.width
        height = video_stream.height
        try:
            fps = float(video_stream.average_rate) if video_stream.average_rate else float(video_stream.rate)
        except Exception:
            fps = 30.0  # fallback
        
        print(f"   Анализ видео:")
        print(f"   Файл: {input_video}")
        print(f"   Размер: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Визуализация: {visualization}")
        print(f"   Минимальная уверенность для детекции : {conf_threshold}")
        
        # Подготовка для создания видео
        if create_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_output_path = os.path.join(output_dir, f"processed_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        # Обработка кадров
        processed_frames = 0
        frames_to_extract = set(extract_frames) if extract_frames else set()
        
        print(" Обработка кадров...")
        
        for frame_idx, frame in enumerate(container.decode(video=0)):
            # Конвертируем кадр
            img = frame.to_ndarray(format='bgr24')
            
            # Детекция объектов
            outputs, ratio = self.detect_objects(img, conf_threshold)
            
            # Визуализация
            if visualization in ["bbox", "both"]:
                img_bbox = self.draw_bounding_boxes(img, outputs, ratio, conf_threshold)
            if visualization in ["circle", "both"]:
                img_circle = self.draw_circles(img, outputs, ratio, conf_threshold)
            
            # Сохранение отдельных кадров
            if frame_idx in frames_to_extract:
                if visualization in ["bbox", "both"]:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(frame_path, img_bbox)
                    print(f" Сохранен кадр: {frame_path}")
                
                if visualization in ["circle", "both"]:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(frame_path, img_circle)
                    print(f" Сохранен кадр: {frame_path}")
            
            # Запись в видео
            if create_video:
                if visualization == "bbox":
                    video_writer.write(img_bbox)
                elif visualization == "circle":
                    video_writer.write(img_circle)
                else:  # both - используем bbox для видео
                    video_writer.write(img_bbox)
            
            processed_frames += 1
            if processed_frames % 30 == 0:
                print(f"   Обработано: {processed_frames} кадров")
        
        # Завершение
        container.close()
        if create_video:
            video_writer.release()
            print(f" Видео сохранено: {video_output_path}")
        
        print(f" Анализ завершен! Обработано кадров: {processed_frames}")
        print(f" Результаты в: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Универсальный анализатор видео с YOLOX')
    parser.add_argument('-i', '--input', required=True, help='Входное видео')
    parser.add_argument('-o', '--output', default='YOLOX_outputs', help='Папка для результатов')
    parser.add_argument('--frames', help='Кадры для сохранения (например: 100,200,300)')
    parser.add_argument('--model', default='yolox-tiny', help='Модель YOLOX')
    parser.add_argument('--weights', default='weights/yolox_tiny.pth', help='Путь к весам')
    parser.add_argument('--conf', type=float, default=0.25, help='Минимальная уверенность для детекции ')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'], help='Устройство')
    parser.add_argument('--visualization', default='both', choices=['bbox', 'circle', 'both'], 
                       help='Тип визуализации')
    parser.add_argument('--no-video', action='store_true', help='Не создавать видео')
    
    args = parser.parse_args()
    
    # Парсим кадры для сохранения
    extract_frames = None
    if args.frames:
        try:
            extract_frames = [int(x.strip()) for x in args.frames.split(',')]
        except ValueError:
            print(" Ошибка в формате кадров")
            return
    
    # Создаем анализатор
    analyzer = VideoAnalyzer(args.model, args.weights, args.device)
    
    # Запускаем анализ
    analyzer.analyze_video(
        input_video=args.input,
        output_dir=args.output,
        extract_frames=extract_frames,
        create_video=not args.no_video,
        visualization=args.visualization,
        conf_threshold=args.conf
    )

if __name__ == "__main__":
    main()