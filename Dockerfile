# Dockerfile
FROM python:3.9-bullseye


# Системные зависимости (включаем dev-пакеты для pyav)
RUN apt-get update && apt-get install -y \
libgl1-mesa-glx \
libglib2.0-0 \
ffmpeg \
wget \
build-essential \
pkg-config \
libavformat-dev \
libavcodec-dev \
libavdevice-dev \
libavutil-dev \
libswscale-dev \
libsm6 \
libxrender1 \
git \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . .


# Создаём каталоги для монтирования
RUN mkdir -p /app/weights /app/assets /app/YOLOX_outputs


# (Опционально) скачиваем веса при сборке — если ссылка актуальна
# Если вы планируете монтировать ./weights из хоста, эту строку можно удалить
RUN wget -q -O /app/weights/yolox_tiny.pth \
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth || true


# Устанавливаем torch (CPU) заранее — уменьшает конфликты зависимостей
RUN pip install --no-cache-dir --timeout=300 \
torch torchvision --index-url https://download.pytorch.org/whl/cpu


# Устанавливаем остальные Python-зависимости
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt


# Если в корне проекта присутствует setup.py/pyproject.toml (например, сам репозиторий YOLOX),
# можно установить пакет в editable режиме. Уберите эту строку, если её не нужно выполнять.
RUN pip install --no-cache-dir -v -e . || true


ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1


VOLUME ["/app/assets", "/app/YOLOX_outputs", "/app/weights"]


CMD ["python", "tools/video_analyzer.py", "--help"]