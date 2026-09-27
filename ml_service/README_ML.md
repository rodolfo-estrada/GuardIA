# Módulo de Machine Learning & Deep Learning - GuardIA

Este módulo implementa el entrenamiento, evaluación mediante validación cruzada e inferencia en tiempo real para la clasificación de nivel de amenaza de seguridad.

---

## 🚀 Guía de Uso Rápido

### 1. Entrenar y Evaluar los 3 Algoritmos
Para volver a entrenar los 3 modelos (*Random Forest*, *Gradient Boosting* y *Red Neuronal MLP*), ejecutar:

```bash
cd guardia-panel
python ml_service/train_models.py
```

Esto generará/actualizará los artefactos en `ml_service/`:
- `best_model.joblib`: Modelo entrenado seleccionado automáticamente (mayor F1-Score).
- `scaler.joblib`: Normalizador de características (`StandardScaler`).
- `label_encoder.joblib`: Codificador de etiquetas de clase.
- `evaluation_results.json`: Métricas comparativas (Train/Test y 5-Fold Cross Validation).

---

### 2. Iniciar el Servicio API de Inferencia (Python FastAPI)

```bash
cd guardia-panel
python ml_service/api.py
```

El servicio estará disponible en `http://127.0.0.1:8000`.

- **Métricas de Evaluación**: `GET http://127.0.0.1:8000/ml/evaluation`
- **Inferencia en Tiempo Real**: `POST http://127.0.0.1:8000/ml/predict`

---

### 3. Iniciar la Aplicación Principal (Frontend y Backend Express)

```bash
# Terminal 1: Backend Express
cd guardia-panel/api
npm run dev

# Terminal 2: Frontend React
cd guardia-panel
npm run dev
```

Navega en tu navegador a `http://localhost:5173/ml-analysis` para ver el panel de control funcional.
