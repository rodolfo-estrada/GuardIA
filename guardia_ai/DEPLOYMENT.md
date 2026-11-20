# Deployment Guide

## Objetivo

Desplegar la API de GuardIA en un servidor para que pueda ser consumida por:

- panel web React
- app movil

## Punto importante

Si subes la IA a un servidor remoto, `webcam 0` deja de tener sentido.

En servidor, la fuente de video debe ser una de estas:

- camara IP con URL `rtsp://...`
- stream HTTP/MJPEG
- video local montado en el servidor

Para produccion, la arquitectura correcta es:

1. El servidor corre la API de GuardIA.
2. Las camaras publican stream accesible para el servidor.
3. El panel web y la app movil consumen la API publica de GuardIA.

## Opcion recomendada

Usar Docker en un VPS o instancia cloud.

Ejemplos viables:

- Ubuntu Server en DigitalOcean
- AWS EC2
- Azure VM
- Google Compute Engine

## Variables de entorno

- `GUARDIA_API_HOST`
- `GUARDIA_API_PORT`
- `GUARDIA_CORS_ALLOW_ORIGINS`

Ejemplo:

```bash
export GUARDIA_API_HOST=0.0.0.0
export GUARDIA_API_PORT=8000
export GUARDIA_CORS_ALLOW_ORIGINS=https://tu-panel.com,https://tu-app.com
```

## Build local de Docker

```bash
docker build -t guardia-ia:latest .
```

## Run local de Docker

```bash
docker run --rm -p 8000:8000 \
  -e GUARDIA_CORS_ALLOW_ORIGINS=http://localhost:5173 \
  -v $(pwd)/capturas:/app/capturas \
  -v $(pwd)/logs:/app/logs \
  guardia-ia:latest
```

En Windows PowerShell:

```powershell
docker build -t guardia-ia:latest .

docker run --rm -p 8000:8000 `
  -e GUARDIA_CORS_ALLOW_ORIGINS=http://localhost:5173 `
  -v ${PWD}\capturas:/app/capturas `
  -v ${PWD}\logs:/app/logs `
  guardia-ia:latest
```

## Deploy en servidor Linux

1. Instalar Docker.
2. Copiar el proyecto al servidor.
3. Construir la imagen:

```bash
docker build -t guardia-ia:latest .
```

4. Ejecutar el contenedor:

```bash
docker run -d --name guardia-ia \
  --restart unless-stopped \
  -p 8000:8000 \
  -e GUARDIA_CORS_ALLOW_ORIGINS=https://panel.tudominio.com,https://api.tuapp.com \
  -v /opt/guardia/capturas:/app/capturas \
  -v /opt/guardia/logs:/app/logs \
  guardia-ia:latest
```

## Nginx reverse proxy

Publica la API con HTTPS usando Nginx delante del contenedor.

Ejemplo minimo:

```nginx
server {
    listen 80;
    server_name ia.tudominio.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Luego agrega HTTPS con Let's Encrypt.

## URLs del panel y app movil

Cuando la API ya este publica, usa algo como esto:

```env
VITE_API_BASE_URL=https://ia.tudominio.com
```

La app movil debe consumir esa misma base URL.

## Endpoints utiles

- `GET /`
- `GET /health`
- `GET /status`
- `GET /cameras`
- `POST /cameras`
- `PATCH /cameras/{camera_id}`
- `GET /alerts`
- `POST /detections/start`
- `POST /detections/stop`
- `GET /stream/{camera_id}`
- `GET /captures/<archivo>`

## Riesgos reales de produccion

- no uses `webcam 0` en servidor cloud
- si el stream de camara no es accesible desde el servidor, la IA no podra leerlo
- `allow_origins=*` sirve para pruebas, no para produccion
- el modelo puede consumir bastante CPU si no hay GPU

## Siguiente paso recomendado

Antes de subirlo:

1. Registrar una camara IP real en vez de webcam local.
2. Probar `GET /stream/{camera_id}` desde el servidor.
3. Apuntar el panel React y la app movil al dominio publico de la API.
