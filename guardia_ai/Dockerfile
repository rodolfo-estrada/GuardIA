FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GUARDIA_API_HOST=0.0.0.0 \
    GUARDIA_API_PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY app.py detector.py utils.py README.md ./
COPY yolov8n.pt ./yolov8n.pt

RUN mkdir -p capturas logs videos

EXPOSE 8000

CMD ["python", "app.py", "--mode", "api", "--host", "0.0.0.0"]
