import json
import os
import time
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="GuardIA ML Inference API",
    description="API de Inferencia y Evaluación de Modelos de Machine Learning / Deep Learning para GuardIA",
    version="1.0.0"
)

# Configuración CORS para permitir peticiones desde Vite / Express
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.joblib")
EVAL_PATH = os.path.join(BASE_DIR, "evaluation_results.json")

# Carga global de artefactos
best_model = None
scaler = None
label_encoder = None
evaluation_meta = {}

def load_artifacts():
    global best_model, scaler, label_encoder, evaluation_meta
    try:
        if os.path.exists(MODEL_PATH):
            best_model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(ENCODER_PATH):
            label_encoder = joblib.load(ENCODER_PATH)
        if os.path.exists(EVAL_PATH):
            with open(EVAL_PATH, "r", encoding="utf-8") as f:
                evaluation_meta = json.load(f)
        print("Artefactos ML cargados en memoria correctamente.")
    except Exception as e:
        print(f"Error cargando artefactos ML: {e}")

@app.on_event("startup")
def startup_event():
    load_artifacts()

class PredictionInput(BaseModel):
    motion_level: float = Field(..., ge=0, le=100, description="Porcentaje de movimiento (0 a 100)")
    ai_confidence: float = Field(..., ge=0, le=100, description="Confianza del detector visual (0 a 100)")
    person_count: int = Field(..., ge=0, description="Número de personas detectadas")
    vehicle_count: int = Field(..., ge=0, description="Número de vehículos detectados")
    duration_seconds: float = Field(..., ge=0, description="Duración de la anomalía en segundos")
    is_night: int = Field(..., ge=0, le=1, description="1 para horario nocturno, 0 para diurno")
    zone_risk_level: int = Field(..., ge=1, le=3, description="1: Baja, 2: Media, 3: Alta")
    anomaly_score: Optional[float] = Field(None, description="Puntuación previa de anomalía (opcional)")

class PredictionOutput(BaseModel):
    threat_level: str
    confidence: float
    probabilities: Dict[str, float]
    selected_model_name: str
    execution_time_ms: float

class SelectModelInput(BaseModel):
    model_key: str

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": best_model is not None,
        "selected_model": evaluation_meta.get("selected_model_name", "Desconocido")
    }

@app.get("/ml/evaluation")
def get_evaluation_metrics():
    if not evaluation_meta:
        raise HTTPException(status_code=500, detail="Métricas de evaluación no encontradas.")
    return evaluation_meta

@app.post("/ml/select-model")
def select_active_model(data: SelectModelInput):
    global best_model, evaluation_meta
    key = data.model_key
    target_path = os.path.join(BASE_DIR, f"model_{key}.joblib")
    if not os.path.exists(target_path):
        target_path = MODEL_PATH

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Modelo {key} no encontrado.")

    try:
        best_model = joblib.load(target_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo {key}: {e}")

    if "models" in evaluation_meta:
        for m_key in evaluation_meta["models"]:
            evaluation_meta["models"][m_key]["is_selected"] = (m_key == key)
        if key in evaluation_meta["models"]:
            evaluation_meta["selected_model"] = key
            evaluation_meta["selected_model_name"] = evaluation_meta["models"][key]["name"]

        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_meta, f, indent=2, ensure_ascii=False)

    return {
        "ok": True,
        "selected_model": key,
        "selected_model_name": evaluation_meta.get("selected_model_name", key)
    }

@app.post("/ml/predict", response_model=PredictionOutput)
def predict_threat(data: PredictionInput):
    if best_model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Modelo ML no inicializado. Ejecuta train_models.py primero.")
    
    start_time = time.time()
    
    # Calcular anomaly_score si no fue provisto
    if anomaly_score is None:
        motion_factor = 1.0 if (data.person_count > 0 or data.vehicle_count > 0) else 0.4
        raw_score = (
            0.35 * data.motion_level * motion_factor +
            0.15 * (data.ai_confidence * 0.7) +
            18.0 * min(data.person_count, 5) +
            14.0 * min(data.vehicle_count, 4) +
            0.05 * min(data.duration_seconds, 120) +
            8 * data.is_night +
            6 * data.zone_risk_level
        )
        anomaly_score = min(max(raw_score, 0), 100)

    # Construcción del vector de características
    features = np.array([[
        data.motion_level,
        data.ai_confidence,
        data.person_count,
        data.vehicle_count,
        data.duration_seconds,
        data.is_night,
        data.zone_risk_level,
        anomaly_score
    ]])
    
    # Escalamiento e Inferencia
    features_scaled = scaler.transform(features)
    probs = best_model.predict_proba(features_scaled)[0]
    pred_idx = np.argmax(probs)
    
    classes = label_encoder.classes_
    predicted_class = str(classes[pred_idx])
    confidence = float(probs[pred_idx]) * 100.0
    
    prob_dict = {str(classes[i]): round(float(probs[i]) * 100, 2) for i in range(len(classes))}
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    
    return PredictionOutput(
        threat_level=predicted_class,
        confidence=round(confidence, 2),
        probabilities=prob_dict,
        selected_model_name=evaluation_meta.get("selected_model_name", "Gradient Boosting Classifier"),
        execution_time_ms=elapsed_ms
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
