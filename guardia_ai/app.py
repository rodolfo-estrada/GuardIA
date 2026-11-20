"""Punto de entrada para el MVP de GuardIA."""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
import threading
import time
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from detector import DetectionEvent, PersonDetector
from utils import ensure_directories, read_recent_events

DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_STREAM_JPEG_QUALITY = 90
DEFAULT_API_HOST = os.getenv("GUARDIA_API_HOST", "127.0.0.1")
# Railway and other PaaS providers inject PORT at runtime. Keep the
# project-specific variable as a local/deployment fallback.
DEFAULT_API_PORT = int(os.getenv("PORT", os.getenv("GUARDIA_API_PORT", "8000")))
DEFAULT_CORS_ORIGINS = os.getenv("GUARDIA_CORS_ALLOW_ORIGINS", "*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototipo GuardIA para deteccion de personas con YOLOv8."
    )
    parser.add_argument(
        "--mode",
        choices=("cli", "api"),
        default="cli",
        help="Modo de ejecucion: ventana local (`cli`) o API para integrar con React (`api`).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Fuente de video. Usa 0 para webcam o una ruta local de video.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Modelo YOLOv8 a utilizar. Ejemplo: yolov8n.pt o yolov8s.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Umbral minimo de confianza para la clase person.",
    )
    parser.add_argument(
        "--save-cooldown",
        type=float,
        default=2.0,
        help="Segundos minimos entre capturas consecutivas.",
    )
    parser.add_argument(
        "--event-reset-seconds",
        type=float,
        default=5.0,
        help="Segundos sin deteccion requeridos para considerar un nuevo evento.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indice de webcam a usar cuando --source sea un numero.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_API_HOST,
        help="Host para exponer la API en modo api.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_API_PORT,
        help="Puerto para exponer la API en modo api.",
    )
    return parser


def normalize_source(raw_source: str, camera_index: int) -> int | str:
    if raw_source == "0":
        return camera_index
    return int(raw_source) if raw_source.isdigit() else raw_source


def configure_capture(capture: cv2.VideoCapture, source: int | str) -> None:
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, 30)

        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        capture.set(cv2.CAP_PROP_FOURCC, mjpg)


class CameraRecord(BaseModel):
    id: str
    codigo: str
    nombre: str
    zone_name: str
    estado: Literal["online", "offline", "mantenimiento"]
    protocolo: Literal["RTSP", "ONVIF"]
    resolucion: Literal["720p", "1080p", "4K"]
    url_stream: str
    perfiles_ia: list[str]
    retencion_dias: int


class CameraCreate(BaseModel):
    name: str
    zone: str
    streamUrl: str = ""
    protocol: Literal["RTSP", "ONVIF"] = "RTSP"
    resolution: Literal["720p", "1080p", "4K"] = "1080p"
    aiProfiles: list[str] = Field(default_factory=lambda: ["person-detection"])
    retentionDays: int = 7


class DetectionControl(BaseModel):
    cameraId: str
    model: str = "yolov8n.pt"
    confidence: float = 0.50
    saveCooldown: float = 2.0
    eventResetSeconds: float = 5.0


class DetectionStatus(BaseModel):
    running: bool
    cameraId: str | None
    source: str | None
    startedAt: float | None
    device: str | None
    model: str | None
    personCount: int = 0
    motionLevel: float = 0.0


class WorkflowResponse(BaseModel):
    ok: bool


class CameraUpdate(BaseModel):
    name: str
    zone: str
    streamUrl: str = ""
    protocol: Literal["RTSP", "ONVIF"] = "RTSP"
    resolution: Literal["720p", "1080p", "4K"] = "1080p"
    aiProfiles: list[str] = Field(default_factory=lambda: ["person-detection"])
    retentionDays: int = 7


class SharedCameraFeed:
    def __init__(self, source: int | str) -> None:
        self.source = source
        self.capture: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.frame_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_id = 0
        self.clients = 0
        self.is_virtual = (source == "phone_stream")

    def acquire(self) -> None:
        with self.lock:
            self.clients += 1
            if self.is_virtual:
                # Virtual feeds receive frames via push_frame(); no capture thread needed.
                return
            if self.thread is not None and self.thread.is_alive():
                return

            self.stop_event.clear()
            self.capture = cv2.VideoCapture(self.source)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = None
                self.clients -= 1
                raise RuntimeError(f"No se pudo abrir la fuente compartida: {self.source}")
            configure_capture(self.capture, self.source)

            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def release(self) -> None:
        thread_to_join: threading.Thread | None = None
        with self.lock:
            self.clients = max(0, self.clients - 1)
            if self.clients == 0:
                self.stop_event.set()
                self.frame_event.set()
                thread_to_join = self.thread

        if thread_to_join is not None:
            thread_to_join.join(timeout=1.0)

        with self.lock:
            if self.clients == 0:
                self.thread = None
                self.latest_frame = None
                self.frame_id = 0
                self.frame_event.clear()

    def push_frame(self, frame) -> None:
        """Push a frame from an external source (e.g. mobile phone upload)."""
        with self.lock:
            self.latest_frame = frame
            self.frame_id += 1
            self.frame_event.set()

    def get_frame(self, timeout: float = 1.0):
        if not self.frame_event.wait(timeout=timeout):
            return None

        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy(), self.frame_id

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                if self.capture is None:
                    break

                success, frame = self.capture.read()
                if not success:
                    time.sleep(0.05)
                    continue

                with self.lock:
                    self.latest_frame = frame
                    self.frame_id += 1
                    self.frame_event.set()
        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None


class GuardIAService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.cameras: list[CameraRecord] = [
            CameraRecord(
                id="cam-local-1",
                codigo="CAM-001",
                nombre="Webcam principal",
                zone_name="Acceso local",
                estado="online",
                protocolo="RTSP",
                resolucion="1080p",
                url_stream="0",
                perfiles_ia=["person-detection"],
                retencion_dias=7,
            ),
            CameraRecord(
                id="cam-phone",
                codigo="CAM-002",
                nombre="Cámara Móvil",
                zone_name="Operador móvil",
                estado="online",
                protocolo="RTSP",
                resolucion="720p",
                url_stream="phone_stream",
                perfiles_ia=["person-detection"],
                retencion_dias=7,
            ),
        ]
        self.detector_thread: threading.Thread | None = None
        self.detector_stop = threading.Event()
        self.detector_lock = threading.Lock()
        self.detector_status = DetectionStatus(
            running=False,
            cameraId=None,
            source=None,
            startedAt=None,
            device=None,
            model=None,
        )
        self.shared_feeds: dict[str, SharedCameraFeed] = {}

    def list_cameras(self) -> list[dict]:
        return [camera.model_dump() for camera in self.cameras]

    def create_camera(self, payload: CameraCreate) -> str:
        camera_id = f"cam-{uuid.uuid4().hex[:8]}"
        codigo = f"CAM-{len(self.cameras) + 1:03d}"
        record = CameraRecord(
            id=camera_id,
            codigo=codigo,
            nombre=payload.name,
            zone_name=payload.zone,
            estado="online",
            protocolo=payload.protocol,
            resolucion=payload.resolution,
            url_stream=payload.streamUrl or "0",
            perfiles_ia=payload.aiProfiles,
            retencion_dias=payload.retentionDays,
        )
        self.cameras.append(record)
        return camera_id

    def get_camera(self, camera_id: str) -> CameraRecord:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        raise KeyError(camera_id)

    def update_camera(self, camera_id: str, payload: CameraUpdate) -> None:
        for index, camera in enumerate(self.cameras):
            if camera.id != camera_id:
                continue

            self.cameras[index] = CameraRecord(
                id=camera.id,
                codigo=camera.codigo,
                nombre=payload.name,
                zone_name=payload.zone,
                estado="online" if payload.streamUrl or camera.url_stream else camera.estado,
                protocolo=payload.protocol,
                resolucion=payload.resolution,
                url_stream=payload.streamUrl or "0",
                perfiles_ia=payload.aiProfiles,
                retencion_dias=payload.retentionDays,
            )
            return

        raise KeyError(camera_id)

    def start_detection(self, payload: DetectionControl) -> None:
        with self.detector_lock:
            if self.detector_status.running:
                raise RuntimeError("La deteccion ya esta en ejecucion.")

            camera = self.get_camera(payload.cameraId)
            source = normalize_source(camera.url_stream, 0)
            self.detector_stop.clear()
            self.detector_status = DetectionStatus(
                running=True,
                cameraId=camera.id,
                source=str(source),
                startedAt=time.time(),
                device=None,
                model=payload.model,
            )

            self.detector_thread = threading.Thread(
                target=self._run_detector,
                args=(source, payload),
                daemon=True,
            )
            self.detector_thread.start()

    def stop_detection(self) -> None:
        with self.detector_lock:
            self.detector_stop.set()
            self.detector_status = DetectionStatus(
                running=False,
                cameraId=None,
                source=None,
                startedAt=None,
                device=None,
                model=None,
            )

    def _on_detection_event(self, event: DetectionEvent) -> None:
        with self.detector_lock:
            self.detector_status.device = event.device
            self.detector_status.personCount = event.detections_count
            self.detector_status.motionLevel = min(96.0, max(65.0, 60.0 + event.detections_count * 15.0))

    def _run_detector(self, source: int | str, payload: DetectionControl) -> None:
        feed = self._get_or_create_feed(source)
        feed.acquire()
        detector = PersonDetector(
            model_path=payload.model,
            confidence_threshold=payload.confidence,
            project_root=self.project_root,
            save_cooldown_seconds=payload.saveCooldown,
            event_reset_seconds=payload.eventResetSeconds,
            display=False,
            on_event=self._on_detection_event,
            stop_event=self.detector_stop,
        )
        try:
            source_label = "webcam" if source == 0 else str(source)
            detector.run_with_frame_provider(
                frame_provider=lambda timeout=1.0: self._next_shared_frame(feed, timeout),
                source_label=source_label,
            )
        except Exception as exc:
            print(f"Error en modo API: {exc}", file=sys.stderr)
        finally:
            feed.release()
            with self.detector_lock:
                self.detector_status = DetectionStatus(
                    running=False,
                    cameraId=None,
                    source=None,
                    startedAt=None,
                    device=detector.device,
                    model=payload.model,
                )
                self.detector_stop.clear()

    def list_alerts(self) -> list[dict]:
        events = read_recent_events(self.project_root / "logs" / "eventos.csv")
        camera_occurrences: dict[str, int] = defaultdict(int)
        items = []
        for index, event in enumerate(reversed(events), start=1):
            item_id = f"ALT-{index:04d}"
            confidence = float(event["confianza"])
            detections_count = int(event.get("detecciones", 1))
            camera_name = "Webcam principal" if event["fuente"] == "webcam" else event["fuente"]
            camera_occurrences[camera_name] += 1
            occurrence_count = camera_occurrences[camera_name]
            detected_at = datetime.fromisoformat(f"{event['fecha']}T{event['hora']}")

            if detections_count >= 2 or confidence >= 0.85 or occurrence_count >= 3:
                severity = "alta"
            elif confidence >= 0.65 or occurrence_count == 2:
                severity = "media"
            else:
                severity = "baja"

            items.append(
                {
                    "id": item_id,
                    "titulo": f"Persona detectada x{occurrence_count}" if occurrence_count > 1 else "Persona detectada",
                    "tipo_evento": "deteccion_persona",
                    "severidad": severity,
                    "estado": "nueva",
                    "confianza_ia": confidence,
                    "descripcion": (
                        f"Se detectaron {detections_count} persona(s) en {camera_name}. "
                        f"Evento #{occurrence_count} registrado para esta camara."
                    ),
                    "detectada_en": detected_at.isoformat(),
                    "resuelta_en": None,
                    "camera_name": camera_name,
                    "zone_name": "Acceso local",
                    "assignee_name": "Sin asignar",
                    "assignee_uid": None,
                    "latest_note": f"Evidencia: {event['captura']}",
                    "latest_action_type": "registro",
                    "latest_action_at": detected_at.isoformat(),
                    "capture_url": f"/captures/{Path(event['captura']).name}",
                    "occurrence_count": occurrence_count,
                    "detection_count": detections_count,
                }
            )
        return items

    def build_stream_source(self, camera_id: str) -> int | str:
        camera = self.get_camera(camera_id)
        return normalize_source(camera.url_stream, 0)

    def stream_frames(self, camera_id: str):
        source = self.build_stream_source(camera_id)
        feed = self._get_or_create_feed(source)
        feed.acquire()
        last_frame_id = -1

        try:
            while True:
                frame_data = feed.get_frame(timeout=1.0)
                if frame_data is None:
                    continue

                frame, frame_id = frame_data
                if frame_id == last_frame_id:
                    continue
                last_frame_id = frame_id

                success, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), DEFAULT_STREAM_JPEG_QUALITY],
                )
                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                )
        finally:
            feed.release()

    def _get_or_create_feed(self, source: int | str) -> SharedCameraFeed:
        key = str(source)
        if key not in self.shared_feeds:
            self.shared_feeds[key] = SharedCameraFeed(source)
        return self.shared_feeds[key]

    @staticmethod
    def _next_shared_frame(feed: SharedCameraFeed, timeout: float = 1.0):
        frame_data = feed.get_frame(timeout=timeout)
        if frame_data is None:
            return None
        frame, _ = frame_data
        return frame


def build_api(project_root: Path) -> FastAPI:
    service = GuardIAService(project_root)
    app = FastAPI(title="GuardIA IA API", version="0.1.0")
    allow_origins = (
        ["*"]
        if DEFAULT_CORS_ORIGINS.strip() == "*"
        else [origin.strip() for origin in DEFAULT_CORS_ORIGINS.split(",") if origin.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/captures", StaticFiles(directory=project_root / "capturas"), name="captures")

    @app.get("/")
    def root() -> dict:
        return {"ok": True, "service": "guardia-ia", "version": "0.1.0"}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": "guardia-ia"}

    @app.get("/status")
    def status() -> dict:
        return service.detector_status.model_dump()

    @app.get("/cameras")
    def cameras() -> dict:
        return {"items": service.list_cameras()}

    @app.post("/cameras")
    def create_camera(payload: CameraCreate) -> dict:
        camera_id = service.create_camera(payload)
        return {"ok": True, "id": camera_id}

    @app.patch("/cameras/{camera_id}")
    def update_camera(camera_id: str, payload: CameraUpdate) -> WorkflowResponse:
        try:
            service.update_camera(camera_id, payload)
            return WorkflowResponse(ok=True)
        except KeyError:
            raise HTTPException(status_code=404, detail="Camara no encontrada.")

    @app.get("/alerts")
    def alerts() -> dict:
        return {"items": service.list_alerts()}

    @app.post("/detections/start")
    def start_detection(payload: DetectionControl) -> WorkflowResponse:
        try:
            service.start_detection(payload)
            return WorkflowResponse(ok=True)
        except KeyError:
            raise HTTPException(status_code=404, detail="Camara no encontrada.")
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/detections/stop")
    def stop_detection() -> WorkflowResponse:
        service.stop_detection()
        return WorkflowResponse(ok=True)

    @app.get("/stream/{camera_id}")
    def stream(camera_id: str):
        try:
            generator = service.stream_frames(camera_id)
            return StreamingResponse(
                generator,
                media_type="multipart/x-mixed-replace; boundary=frame",
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Camara no encontrada.")
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/cameras/{camera_id}/upload_frame")
    async def upload_frame(camera_id: str, file: UploadFile = File(...)):
        """Receive a JPEG frame from the mobile app and push it into the camera feed."""
        try:
            camera = service.get_camera(camera_id)
        except KeyError:
            # Register camera dynamically if it doesn't exist
            suffix = camera_id.replace("cam-phone-", "").replace("cam-phone", "").strip("-")
            name_suffix = f" ({suffix})" if suffix else " (Móvil)"
            camera = CameraRecord(
                id=camera_id,
                codigo=f"CAM-{len(service.cameras) + 1:03d}",
                nombre=f"Cámara Móvil{name_suffix}",
                zone_name="Operador móvil",
                estado="online",
                protocolo="RTSP",
                resolucion="720p",
                url_stream=f"phone_stream_{camera_id}",
                perfiles_ia=["person-detection"],
                retencion_dias=7,
            )
            service.cameras.append(camera)

        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        source = normalize_source(camera.url_stream, 0)
        feed = service._get_or_create_feed(source)
        # Ensure the feed is acquired (at least one virtual client)
        if feed.clients == 0:
            feed.acquire()
        feed.push_frame(frame)
        return {"ok": True}

    return app


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ensure_directories(project_root)

    if args.mode == "api":
        api = build_api(project_root)
        uvicorn.run(api, host=args.host, port=args.port)
        return 0

    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        project_root=project_root,
        save_cooldown_seconds=args.save_cooldown,
        event_reset_seconds=args.event_reset_seconds,
    )

    try:
        detector.run(source=normalize_source(args.source, args.camera_index))
    except KeyboardInterrupt:
        print("\nInterrupcion detectada. Cerrando GuardIA...")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
