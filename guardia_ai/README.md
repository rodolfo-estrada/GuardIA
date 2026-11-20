# GuardIA MVP

Prototipo de videovigilancia con inteligencia artificial para detectar personas en tiempo real usando YOLOv8, OpenCV y Python.

## Requisitos

- Python 3.10 o 3.11
- Webcam integrada o un archivo de video local
- GPU NVIDIA opcional para acelerar inferencia con PyTorch y CUDA

## Estructura

```text
IA_GuardIA/
|-- app.py
|-- detector.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- capturas/
|-- logs/
`-- videos/
```

## Instalacion

1. Crear un entorno virtual:

```powershell
python -m venv venv
```

2. Activarlo en Windows:

```powershell
venv\Scripts\activate
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Ejecucion

### Webcam

```powershell
python app.py
```

### API para React

```powershell
python app.py --mode api --host 127.0.0.1 --port 8000
```

### Video local

```powershell
python app.py --source videos\prueba.mp4
```

### Ajustar confianza o modelo

```powershell
python app.py --confidence 0.60
python app.py --model yolov8s.pt
```

### Elegir otra camara

```powershell
python app.py --camera-index 1
```

### Reducir alertas repetidas

```powershell
python app.py --event-reset-seconds 8
```

## Comportamiento

- Detecta solo la clase `person`
- Dibuja bounding boxes y confianza
- Muestra estado en pantalla, FPS y dispositivo (`cpu` o `cuda`)
- Guarda evidencias en `capturas/`
- Registra eventos en `logs/eventos.csv` con confianza maxima y numero de personas
- Agrupa una presencia continua en un solo evento para evitar spam de alertas
- Sale al presionar `q`

## Endpoints principales

- `GET /health`
- `GET /status`
- `GET /cameras`
- `POST /cameras`
- `GET /alerts`
- `POST /detections/start`
- `POST /detections/stop`
- `GET /captures/<archivo>`

## Integracion con el panel React

Configura en el panel:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Flujo recomendado:

1. Iniciar la API de GuardIA.
2. Desde React, cargar camaras con `GET /cameras`.
3. Iniciar IA con `POST /detections/start`.
4. Mostrar alertas reales con `GET /alerts`.
5. Consumir evidencias desde `/captures/...`.

## Despliegue

La API ya puede desplegarse en servidor como servicio HTTP.

Archivos agregados para eso:

- `Dockerfile`
- `.dockerignore`
- `DEPLOYMENT.md`

Build:

```powershell
docker build -t guardia-ia:latest .
```

Run:

```powershell
docker run --rm -p 8000:8000 `
  -e GUARDIA_CORS_ALLOW_ORIGINS=http://localhost:5173 `
  -v ${PWD}\capturas:/app/capturas `
  -v ${PWD}\logs:/app/logs `
  guardia-ia:latest
```

Para publicar la API en servidor y consumirla desde panel web y app movil, revisa [DEPLOYMENT.md](./DEPLOYMENT.md).

## Notas

- El modelo por defecto es `yolov8n.pt`, pensado para correr bien en una laptop con RTX 3050.
- Si no existe, el proyecto crea automaticamente `capturas/`, `logs/` y `videos/`.
- La descarga inicial del modelo puede tardar unos momentos si no esta en cache local.
