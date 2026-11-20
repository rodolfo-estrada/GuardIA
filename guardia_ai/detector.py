"""Logica principal de deteccion usando YOLOv8 y OpenCV."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import threading

import cv2
import torch
from ultralytics import YOLO

from utils import (
    build_capture_filename,
    ensure_directories,
    log_event,
    save_frame,
)

DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720


@dataclass
class DetectionEvent:
    timestamp: float
    source: str
    confidence: float
    detections_count: int
    capture_path: Path
    device: str


class PersonDetector:
    """Encapsula la carga del modelo y el bucle de deteccion."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        project_root: Path,
        save_cooldown_seconds: float = 2.0,
        event_reset_seconds: float = 5.0,
        display: bool = True,
        on_event: Callable[[DetectionEvent], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.project_root = project_root
        self.save_cooldown_seconds = save_cooldown_seconds
        self.event_reset_seconds = event_reset_seconds
        self.display = display
        self.on_event = on_event
        self.stop_event = stop_event
        self.last_capture_time = 0.0
        self.last_detection_time = 0.0
        self.event_active = False
        self.frames_processed = 0
        self.total_detections = 0
        self.last_status_message = "Iniciando detector..."
        self.last_status_expiration = 0.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ensure_directories(project_root)
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar el modelo '{self.model_path}'."
            ) from exc

    def run(self, source: int | str) -> None:
        if isinstance(source, int):
            # En Windows, usar DirectShow (CAP_DSHOW) para evitar errores de Microsoft Media Foundation (MSMF)
            capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
        self._configure_capture(capture, source)

        source_label = "webcam" if source == 0 else str(source)
        window_name = "GuardIA - Deteccion de Personas"
        frame_time = time.time()

        try:
            while True:
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                success, frame = capture.read()
                if not success:
                    if source == 0:
                        raise RuntimeError("No se pudo leer un frame desde la webcam.")
                    print("Fin del video o frame no disponible.")
                    break

                current_time = time.time()
                fps = 1.0 / max(current_time - frame_time, 1e-6)
                frame_time = current_time

                annotated_frame, detections = self.process_frame(frame, source_label)
                self._draw_hud(
                    annotated_frame,
                    source_label=source_label,
                    fps=fps,
                    detections_count=len(detections),
                )
                if self.display:
                    cv2.imshow(window_name, annotated_frame)

                if self.display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            if self.display:
                cv2.destroyAllWindows()

    def run_with_frame_provider(self, frame_provider, source_label: str) -> None:
        window_name = "GuardIA - Deteccion de Personas"
        frame_time = time.time()

        try:
            while True:
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                frame = frame_provider(timeout=1.0)
                if frame is None:
                    continue

                current_time = time.time()
                fps = 1.0 / max(current_time - frame_time, 1e-6)
                frame_time = current_time

                annotated_frame, detections = self.process_frame(frame, source_label)
                self._draw_hud(
                    annotated_frame,
                    source_label=source_label,
                    fps=fps,
                    detections_count=len(detections),
                )
                if self.display:
                    cv2.imshow(window_name, annotated_frame)

                if self.display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            if self.display:
                cv2.destroyAllWindows()

    @staticmethod
    def _configure_capture(capture: cv2.VideoCapture, source: int | str) -> None:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if isinstance(source, int):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
            capture.set(cv2.CAP_PROP_FPS, 30)

            mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            capture.set(cv2.CAP_PROP_FOURCC, mjpg)

    def process_frame(self, frame, source_label: str):
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        annotated_frame = frame.copy()
        detections = []
        self.frames_processed += 1

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    {
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                    }
                )
                self._draw_detection(annotated_frame, x1, y1, x2, y2, confidence)

        max_confidence = max((detection["confidence"] for detection in detections), default=0.0)

        if detections:
            self.total_detections += len(detections)
            self.last_detection_time = time.time()
            self._set_status(
                f"Persona detectada | conteo={len(detections)} | max={max_confidence:.2f}"
            )
        else:
            if time.time() > self.last_status_expiration:
                self.last_status_message = "Monitoreando..."

        # Emitir evento en tiempo real en cada cuadro analizado
        if self.on_event:
            self.on_event(
                DetectionEvent(
                    timestamp=time.time(),
                    source=source_label,
                    confidence=max_confidence,
                    detections_count=len(detections),
                    capture_path=Path(""),
                    device=self.device,
                )
            )

        if detections and self._should_create_event():
            capture_name = build_capture_filename()
            capture_path = self.project_root / "capturas" / capture_name
            save_frame(capture_path, annotated_frame)
            max_confidence = max(detection["confidence"] for detection in detections)
            log_event(
                self.project_root / "logs" / "eventos.csv",
                confidence=max_confidence,
                source=source_label,
                capture_path=capture_path,
                detections_count=len(detections),
            )
            if self.on_event is not None:
                self.on_event(
                    DetectionEvent(
                        timestamp=time.time(),
                        source=source_label,
                        confidence=max_confidence,
                        detections_count=len(detections),
                        capture_path=capture_path,
                        device=self.device,
                    )
                )
            self._set_status(f"Captura guardada: {capture_name}")
            self.last_capture_time = time.time()
            self.event_active = True

        if not detections and self.event_active and self._should_reset_event():
            self.event_active = False
            self._set_status("Evento cerrado. Sistema en espera.", duration_seconds=2.5)

        return annotated_frame, detections

    def _should_create_event(self) -> bool:
        enough_time_since_capture = (time.time() - self.last_capture_time) >= self.save_cooldown_seconds
        return (not self.event_active) and enough_time_since_capture

    def _should_reset_event(self) -> bool:
        return (time.time() - self.last_detection_time) >= self.event_reset_seconds

    def _set_status(self, message: str, duration_seconds: float = 2.0) -> None:
        self.last_status_message = message
        self.last_status_expiration = time.time() + duration_seconds

    def _draw_hud(
        self,
        frame,
        source_label: str,
        fps: float,
        detections_count: int,
    ) -> None:
        hud_lines = [
            f"Fuente: {source_label}",
            f"Dispositivo: {self.device}",
            f"FPS: {fps:.1f}",
            f"Detecciones actuales: {detections_count}",
            f"Detecciones acumuladas: {self.total_detections}",
            f"Cooldown captura: {self.save_cooldown_seconds:.1f}s",
            self.last_status_message,
        ]

        overlay_height = 28 + (len(hud_lines) * 24)
        cv2.rectangle(frame, (10, 10), (420, overlay_height), (20, 20, 20), -1)

        for index, line in enumerate(hud_lines):
            color = (255, 255, 255)
            if index == len(hud_lines) - 1:
                color = (0, 255, 255)
            cv2.putText(
                frame,
                line,
                (20, 35 + (index * 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_detection(frame, x1: int, y1: int, x2: int, y2: int, confidence: float) -> None:
        label = f"person {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.rectangle(frame, (x1, max(y1 - 30, 0)), (x2, y1), (0, 200, 0), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
