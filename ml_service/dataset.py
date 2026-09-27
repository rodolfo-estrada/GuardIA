import numpy as np
import pandas as pd

def generate_security_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de datos realista de videovigilancia inteligente para clasificación de nivel de amenaza.
    
    Características (Features):
    - motion_level: Nivel de movimiento detectado en video (0 a 100%)
    - ai_confidence: Porcentaje de confianza del modelo visual previo (40 a 99%)
    - person_count: Concurrencia de personas detectadas (0 a 15)
    - vehicle_count: Concurrencia de vehículos (0 a 8)
    - duration_seconds: Duración continuada de la anomalía (1 a 180 s)
    - is_night: 1 si ocurrió en horario nocturno (22:00 - 06:00), 0 diurno
    - zone_risk_level: Nivel de riesgo asignado a la zona de la cámara (1: Baja, 2: Media, 3: Alta)
    - anomaly_score: Puntuación sintética de comportamiento anómalo (0 a 100)
    
    Target:
    - threat_level: 'Baja', 'Media', 'Alta', 'Critica'
    """
    np.random.seed(random_state)

    motion_level = np.random.uniform(5, 98, n_samples)
    ai_confidence = np.random.uniform(45, 99, n_samples)
    person_count = np.random.poisson(2, n_samples)
    vehicle_count = np.random.poisson(1, n_samples)
    duration_seconds = np.random.exponential(25, n_samples) + 2
    is_night = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    zone_risk_level = np.random.choice([1, 2, 3], size=n_samples, p=[0.3, 0.4, 0.3])
    
    anomaly_score = (
        0.35 * motion_level * np.where((person_count > 0) | (vehicle_count > 0), 1.0, 0.4) +
        0.15 * (ai_confidence * 0.7) +
        18.0 * np.minimum(person_count, 5) +
        14.0 * np.minimum(vehicle_count, 4) +
        0.05 * np.minimum(duration_seconds, 120) +
        8 * is_night +
        6 * zone_risk_level +
        np.random.normal(0, 5, n_samples)
    )
    
    # Normalizar score sintético entre 0 y 100
    anomaly_score = np.clip((anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min()) * 100, 0, 100)

    # Asignación de etiquetas basadas en el score combinado de amenaza
    threat_level = []
    for score in anomaly_score:
        if score < 30:
            threat_level.append("Baja")
        elif score < 55:
            threat_level.append("Media")
        elif score < 78:
            threat_level.append("Alta")
        else:
            threat_level.append("Critica")

    df = pd.DataFrame({
        "motion_level": np.round(motion_level, 2),
        "ai_confidence": np.round(ai_confidence, 2),
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "duration_seconds": np.round(duration_seconds, 1),
        "is_night": is_night,
        "zone_risk_level": zone_risk_level,
        "anomaly_score": np.round(anomaly_score, 2),
        "threat_level": threat_level
    })

    return df

if __name__ == "__main__":
    df = generate_security_dataset()
    print(f"Dataset generado exitosamente con {len(df)} muestras.")
    print("Distribución de clases:")
    print(df["threat_level"].value_counts())
