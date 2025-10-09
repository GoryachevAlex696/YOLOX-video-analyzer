# ShorYOLOX Video Analyzertlify

Скрипт на python читает видео используя библиотеку pyav. Каждый кадр анализируется используя библиотеку yolox. 

---

## Быстрый старт

1. Клонировать репозиторий:
```bash
git clone https://github.com/GoryachevAlex696/YOLOX-video-analyzer.git
```
2.   Перейти в папку проекта:
```bash
cd YOLOX-video-analyzer
```

3. Сборка Docker-образа:
```bash
docker build -t yolox-analyzer .
```

3. Проверка что образ создался:
```bash
docker images
```

---

## Зпуск скрипта (через docker-compose)

- Три кадра сохранить как картинки (объект в окружности):
```bash
docker compose run yolox-analyzer python tools/video_analyzer.py -i assets/develop_streem.ts --frames=100,200,224 --no-video
```

- Проанализировать всё видео с выделением объектов на видео (объект в окружности):
```bash
docker compose run yolox-analyzer python tools/video_analyzer.py -i "assets/develop_streem.ts" --visualization=circle
```

- Полный анализ: видео (объект в прямоугольнике) + кадры (объект в окружности)
```bash
docker compose run yolox-analyzer python tools/video_analyzer.py -i "assets/develop_streem.ts" --frames=100,200,224
```

---

## Комбинации параметров:

| Сценарий | Параметры | Результат |
|----------|-----------|-----------|
| **Быстрая обработка** | `-i "assets/develop_streem.ts" --no-video` | Только кадры (без видео) |
| **Только прямоугольники** | `-i "assets/develop_streem.ts" --visualization=bbox` | Только прямоугольные детекции |
| **Только окружности** | `-i "assets/develop_streem.ts" --visualization=circle` | Только круговые детекции |
| **Выборочные кадры** | `-i "assets/develop_streem.ts" --frames=100,200,300` | Сохраняет только указанные кадры |
| **Точная детекция** | `-i "assets/develop_streem.ts" --conf=0.5` | Меньше ложных срабатываний |
| **Максимум объектов** | `-i "assets/develop_streem.ts" --conf=0.1` | Больше обнаружений (возможны ложные) |


