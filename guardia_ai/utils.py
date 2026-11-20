"""Funciones auxiliares para GuardIA."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2


def ensure_directories(project_root: Path) -> None:
    for directory_name in ("capturas", "logs", "videos"):
        (project_root / directory_name).mkdir(parents=True, exist_ok=True)


def build_capture_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"captura_{timestamp}.jpg"


def save_frame(destination: Path, frame) -> None:
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"No se pudo guardar la captura en: {destination}")


def log_event(
    log_path: Path,
    confidence: float,
    source: str,
    capture_path: Path,
    detections_count: int,
) -> None:
    file_exists = log_path.exists()
    timestamp = datetime.now()

    with log_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                ["fecha", "hora", "confianza", "detecciones", "fuente", "captura"]
            )

        writer.writerow(
            [
                timestamp.strftime("%Y-%m-%d"),
                timestamp.strftime("%H:%M:%S"),
                f"{confidence:.4f}",
                detections_count,
                source,
                str(capture_path),
            ]
        )


def read_recent_events(log_path: Path) -> list[dict[str, str]]:
    if not log_path.exists():
        return []

    with log_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)
